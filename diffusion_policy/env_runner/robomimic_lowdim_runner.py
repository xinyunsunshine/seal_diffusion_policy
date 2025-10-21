import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env


class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicLowdimWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[:,:self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction


    def collect_demo_episodes(self, policy: BaseLowdimPolicy, seeds: list, 
                            enable_render=True, output_dir=None):
        """
        Collect demonstration episodes for specific seeds with action/observation recording.
        Uses parallel environments for faster collection.
        
        Args:
            policy: The policy to run
            seeds: List of seeds for episode generation
            enable_render: Whether to record videos
            output_dir: Directory to save videos
            
        Returns:
            List of episode dictionaries with observations, actions, rewards, etc.
        """
        device = policy.device
        episodes = []
        
        if output_dir is None:
            output_dir = self.output_dir
        
        # Create demo videos directory
        demo_video_dir = pathlib.Path(output_dir) / "demo_videos"
        demo_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Plan for parallel collection - batch seeds across available environments
        n_envs = len(self.env_fns)
        n_seeds = len(seeds)
        n_chunks = math.ceil(n_seeds / n_envs)
        
        print(f"Collecting {n_seeds} episodes using {n_envs} parallel environments ({n_chunks} chunks)")
        
        env = self.env
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * n_envs
            end_idx = min(n_seeds, start_idx + n_envs)
            chunk_seeds = seeds[start_idx:end_idx]
            this_n_active_envs = len(chunk_seeds)
            this_local_slice = slice(0, this_n_active_envs)
            
            print(f"Processing chunk {chunk_idx + 1}/{n_chunks}: seeds {chunk_seeds}")
            
            # Prepare initialization functions for this chunk
            init_fns = []
            for i, seed in enumerate(chunk_seeds):
                # Calculate global episode index
                global_ep_idx = start_idx + i
                
                # Setup video path if rendering enabled
                video_path = None
                if enable_render:
                    video_filename = f"demo_episode_{global_ep_idx:04d}_seed_{seed}.mp4"
                    video_path = str(demo_video_dir / video_filename)
                
                def create_init_fn(seed=seed, video_path=video_path):
                    def init_fn(env):
                        # Setup video recording
                        assert isinstance(env.env, VideoRecordingWrapper)
                        env.env.video_recoder.stop()
                        env.env.file_path = video_path
                        
                        # Use seed-based reset for demo collection
                        assert isinstance(env.env.env, RobomimicLowdimWrapper)
                        env.env.env.init_state = None
                        env.seed(seed)
                        
                    return init_fn
                
                init_fns.append(create_init_fn())
            
            # Pad init functions to match number of environments
            n_diff = n_envs - len(init_fns)
            if n_diff > 0:
                init_fns.extend([init_fns[0]] * n_diff)
            
            # Serialize initialization functions
            init_fn_dills = [dill.dumps(fn) for fn in init_fns]
            
            # Initialize environments for this chunk
            env.call_each('run_dill_function', args_list=[(x,) for x in init_fn_dills])
            
            # Collect episodes for this chunk in parallel
            chunk_episodes = self._collect_parallel_demo_episodes(
                env, policy, chunk_seeds, start_idx, this_local_slice
            )
            
            # Get video paths after completion
            all_video_paths = env.render()
            for i, episode in enumerate(chunk_episodes):
                if i < len(all_video_paths) and all_video_paths[i]:
                    episode['video_path'] = all_video_paths[i]
            
            episodes.extend(chunk_episodes)
        
        return episodes

    def _collect_parallel_demo_episodes(self, env, policy, seeds, start_episode_idx, local_slice):
        """Collect episodes in parallel across multiple environments."""
        device = policy.device
        n_active_envs = len(seeds)
        
        # Reset environments and policy
        obs = env.reset()
        policy.reset()
        
        # Storage for episode data per environment
        env_episode_data = []
        for i in range(n_active_envs):
            env_episode_data.append({
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'infos': [],
                'seed': seeds[i],
                'episode_idx': start_episode_idx + i,
                'episode_reward': 0,
                'step_count': 0
            })
        
        # Track which environments are still active
        active_envs = list(range(n_active_envs))
        prev_action = None
        
        pbar = tqdm.tqdm(total=self.max_steps * n_active_envs, 
                        desc=f"Demo collection", 
                        leave=False, 
                        mininterval=1.0)
        
        while len(active_envs) > 0:
            # Create obs dict for active environments
            active_obs = obs[active_envs, :self.n_obs_steps].astype(np.float32)
            np_obs_dict = {'obs': active_obs}
            
            # Device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(device=device))
            
            # Run policy
            with torch.no_grad():
                if hasattr(policy, "use_action_traj") and policy.use_action_traj:
                    action_dict = policy.predict_action(obs_dict, prev_action=prev_action)
                else:
                    action_dict = policy.predict_action(obs_dict)
            
            # Convert back to numpy
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            
            # Handle latency steps
            action = np_action_dict['action'][:, self.n_latency_steps:]
            
            if hasattr(policy, "use_action_traj") and policy.use_action_traj:
                prev_action = action_dict['prev_action']
            
            if not np.all(np.isfinite(action)):
                raise RuntimeError("Nan or Inf action")
            
            # Apply action transformation for environment stepping
            env_action = action
            if self.abs_action:
                env_action = self.undo_transform_action(action)
            
            # Prepare full action array (inactive envs get zero actions)
            full_env_action = np.zeros((len(self.env_fns), action.shape[1], env_action.shape[2]))
            full_env_action[active_envs] = env_action
            
            # Store data for active environments before stepping
            for i, env_idx in enumerate(active_envs):
                episode_data = env_episode_data[env_idx]
                action_sequence = action[i]  # Remove batch dimension for this env
                
                # Store observations and actions for each action step
                for action_step_idx in range(action_sequence.shape[0]):
                    current_obs = obs[env_idx, min(action_step_idx, self.n_obs_steps-1)]
                    episode_data['observations'].append(current_obs.copy())
                    episode_data['actions'].append(action_sequence[action_step_idx].copy())
            
            # Step all environments
            new_obs, reward, done, info = env.step(full_env_action)
            
            # Update episode data for active environments
            new_active_envs = []
            for i, env_idx in enumerate(active_envs):
                episode_data = env_episode_data[env_idx]
                env_reward = reward[env_idx]
                env_done = done[env_idx]
                env_info = info[env_idx]
                
                # Update episode data for each action step
                for action_step_idx in range(action.shape[1]):
                    episode_data['rewards'].append(env_reward)
                    episode_data['dones'].append(env_done)
                    episode_data['infos'].append(env_info)
                    episode_data['step_count'] += 1
                
                episode_data['episode_reward'] += env_reward
                
                # Check if environment should continue
                # Episode ends if: done=True OR max_steps reached OR success achieved (reward > 0)
                episode_should_end = (env_done or 
                                    episode_data['step_count'] >= self.max_steps or
                                    env_reward > 0)  # End immediately on success
                
                if not episode_should_end:
                    new_active_envs.append(env_idx)
                else:
                    # Finalize episode data
                    try:
                        # Get final reward from environment
                        final_reward = env.get_attr('reward')[env_idx]
                        if isinstance(final_reward, (list, np.ndarray)):
                            final_reward = float(final_reward[0]) if len(final_reward) > 0 else float(episode_data['episode_reward'])
                        else:
                            final_reward = float(final_reward)
                    except:
                        final_reward = float(episode_data['episode_reward'])
                    
                    # Use the most recent reward if it's positive (indicates success)
                    if env_reward > 0:
                        final_reward = float(env_reward)
                    
                    episode_data['final_reward'] = final_reward
                    episode_data['success'] = final_reward > 0
                    episode_data['episode_length'] = len(episode_data['actions'])
                    
                    # Ensure all arrays have the same length
                    min_length = min(len(episode_data['observations']), len(episode_data['actions']), 
                                   len(episode_data['rewards']), len(episode_data['dones']), 
                                   len(episode_data['infos']))
                    
                    episode_data['observations'] = episode_data['observations'][:min_length]
                    episode_data['actions'] = np.array(episode_data['actions'][:min_length], dtype=np.float32)
                    episode_data['rewards'] = np.array(episode_data['rewards'][:min_length], dtype=np.float32)
                    episode_data['dones'] = np.array(episode_data['dones'][:min_length], dtype=bool)
                    episode_data['infos'] = episode_data['infos'][:min_length]
                    
                    # Log episode completion
                    success_str = "SUCCESS" if episode_data['success'] else "FAILURE"
                    termination_reason = "SUCCESS" if env_reward > 0 else ("DONE" if env_done else "MAX_STEPS")
                    print(f"Episode {episode_data['episode_idx']} (seed {episode_data['seed']}) completed: {success_str} ({termination_reason}), "
                          f"reward={final_reward:.3f}, steps={episode_data['episode_length']}")
            
            active_envs = new_active_envs
            obs = new_obs
            
            # Update progress bar
            total_completed_steps = sum(len(data['actions']) for data in env_episode_data[:n_active_envs])
            pbar.n = total_completed_steps
            pbar.refresh()
        
        pbar.close()
        
        # Return episode data for this chunk
        return env_episode_data[:n_active_envs]
