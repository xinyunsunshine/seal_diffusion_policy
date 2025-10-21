# filepath: /home/sunsh16e/streaming_flow_policy/collect_demo_data.py
import os
import h5py
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
import pathlib
from tqdm import tqdm
import copy
import math

from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace
from diffusion_policy.env_runner.robomimic_lowdim_runner import RobomimicLowdimRunner

class DemoCollector:
    def __init__(self, checkpoint_path: str, config_path: str = None):
        """
        Initialize demo collector with trained policy weights.
        
        Args:
            checkpoint_path: Path to trained policy checkpoint
            config_path: Optional path to config file. If None, tries to find it in checkpoint dir
        """
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        
        # Load config
        if config_path is None:
            # Try to find config in same directory as checkpoint
            config_dir = self.checkpoint_path.parent.parent
            config_files = list(config_dir.glob(".hydra"))
            if len(config_files) > 0:
                hydra_dir = config_files[0] / "config.yaml"
                self.cfg = OmegaConf.load(hydra_dir)
            else:
                raise ValueError(f"Could not find config file. Please specify config_path")
        else:
            self.cfg = OmegaConf.load(config_path)
        
        # Initialize workspace and load checkpoint
        self.workspace = TrainDiffusionUnetLowdimWorkspace(self.cfg)
        self.workspace.load_checkpoint(path=self.checkpoint_path)
        
        # Get policy
        self.policy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
        self.policy.eval()
        
        # Initialize environment runner
        self.env_runner: RobomimicLowdimRunner = hydra.utils.instantiate(
            self.cfg.task.env_runner,
            output_dir=self.checkpoint_path.parent
        )
        
        # Set device
        self.device = torch.device(self.cfg.training.device)
        self.policy.to(self.device)
    
    def collect_episodes(self, num_episodes: int, start_seed: int = 50000, 
                        output_dir: str = None, save_videos: bool = True,
                        batch_size: int = None):
        """
        Collect demonstration episodes using the trained policy via the parallel runner.
        
        Args:
            num_episodes: Number of episodes to collect
            start_seed: Starting seed for episode generation
            output_dir: Directory to save videos. If None, uses checkpoint parent dir
            save_videos: Whether to save videos for each episode
            batch_size: Number of episodes to collect in parallel. If None, uses all available envs
            
        Returns:
            List of episode dictionaries containing obs and actions
        """
        if output_dir is None:
            output_dir = self.checkpoint_path.parent / "demo_collection"
        else:
            output_dir = pathlib.Path(output_dir)
        
        # Create directories
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate seed list
        seeds = [start_seed + i for i in range(num_episodes)]
        
        # If batch_size is specified, process in smaller batches
        if batch_size is not None and batch_size < len(seeds):
            all_episodes = []
            for i in range(0, len(seeds), batch_size):
                batch_seeds = seeds[i:i + batch_size]
                print(f"Collecting batch {i//batch_size + 1}/{math.ceil(len(seeds)/batch_size)}: {len(batch_seeds)} episodes")
                
                batch_episodes = self.env_runner.collect_demo_episodes(
                    policy=self.policy,
                    seeds=batch_seeds,
                    enable_render=save_videos,
                    output_dir=output_dir
                )
                all_episodes.extend(batch_episodes)
            episodes = all_episodes
        else:
            # Use the runner's parallel demo collection method
            episodes = self.env_runner.collect_demo_episodes(
                policy=self.policy,
                seeds=seeds,
                enable_render=save_videos,
                output_dir=output_dir
            )
        
        # Print collection summary
        successful_episodes = sum(1 for ep in episodes if ep.get('success', False))
        total_steps = sum(ep.get('episode_length', 0) for ep in episodes)
        avg_reward = np.mean([ep.get('final_reward', 0) for ep in episodes])
        
        print(f"\nDemo Collection Summary:")
        print(f"  Episodes collected: {len(episodes)}")
        print(f"  Successful episodes: {successful_episodes}/{len(episodes)} ({successful_episodes/len(episodes)*100:.1f}%)")
        print(f"  Total steps: {total_steps}")
        print(f"  Average episode length: {total_steps/len(episodes):.1f}")
        print(f"  Average final reward: {avg_reward:.3f}")
        
        return episodes
    
    def save_to_hdf5(self, episodes, output_path: str, obs_keys: list = None, 
                     save_all_episodes: bool = False):
        """
        Save collected episodes to HDF5 format compatible with robomimic dataset.
        
        Args:
            episodes: List of episode dictionaries
            output_path: Path to save HDF5 file
            obs_keys: List of observation keys to save. If None, uses all available keys
            save_all_episodes: If True, saves all episodes. If False, saves only successful ones
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter episodes based on save_all_episodes flag
        if save_all_episodes:
            episodes_to_save = episodes
            print(f"Saving all {len(episodes_to_save)} episodes")
        else:
            episodes_to_save = [ep for ep in episodes if ep.get('success', False)]
            print(f"Saving {len(episodes_to_save)}/{len(episodes)} successful episodes")
        
        if len(episodes_to_save) == 0:
            print("Warning: No episodes to save!")
            return
        
        # Determine observation keys from first episode
        if obs_keys is None:
            first_obs = episodes_to_save[0]['observations'][0]
            if isinstance(first_obs, dict):
                obs_keys = list(first_obs.keys())
            else:
                obs_keys = ['obs']  # Default key for single observation
        
        # Create summary file with episode info
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Demo Collection Summary\n")
            f.write(f"======================\n")
            f.write(f"Total episodes collected: {len(episodes)}\n")
            f.write(f"Episodes saved to HDF5: {len(episodes_to_save)}\n")
            f.write(f"Success rate: {len([ep for ep in episodes if ep.get('success', False)])/len(episodes)*100:.1f}%\n\n")
            
            f.write(f"Episode Details:\n")
            f.write(f"Idx\tSeed\tSteps\tReward\tSuccess\tVideo Path\n")
            for i, ep in enumerate(episodes):
                f.write(f"{ep.get('episode_idx', i)}\t{ep.get('seed', 'N/A')}\t"
                       f"{ep.get('episode_length', len(ep['actions']))}\t"
                       f"{ep.get('final_reward', 0):.3f}\t"
                       f"{ep.get('success', False)}\t"
                       f"{ep.get('video_path', 'N/A')}\n")
        
        # Save to HDF5
        with h5py.File(output_path, 'w') as f:
            # Create data group
            data_grp = f.create_group('data')
            
            # Save metadata
            f.attrs['total_episodes_collected'] = len(episodes)
            f.attrs['episodes_saved'] = len(episodes_to_save)
            f.attrs['success_rate'] = len([ep for ep in episodes if ep.get('success', False)])/len(episodes)
            
            # Save each episode
            for ep_idx, episode in enumerate(tqdm(episodes_to_save, desc="Saving episodes")):
                demo_grp = data_grp.create_group(f'demo_{ep_idx}')
                
                # Save actions
                actions = episode['actions']
                demo_grp.create_dataset('actions', data=actions)
                
                # Save observations
                obs_grp = demo_grp.create_group('obs')
                observations = episode['observations']
                
                # Handle different observation formats
                if isinstance(observations[0], dict):
                    # Multi-key observations
                    for key in obs_keys:
                        if key in observations[0]:
                            obs_data = np.array([obs[key] for obs in observations])
                            obs_grp.create_dataset(key, data=obs_data)
                else:
                    # Single observation array
                    obs_data = np.array(observations)
                    obs_grp.create_dataset('obs', data=obs_data)
                
                # Save rewards, dones, and states (if available)
                demo_grp.create_dataset('rewards', data=episode['rewards'])
                demo_grp.create_dataset('dones', data=episode['dones'])
                
                # Save episode metadata
                demo_grp.attrs['success'] = episode.get('success', False)
                demo_grp.attrs['num_samples'] = len(actions)
                demo_grp.attrs['final_reward'] = episode.get('final_reward', 0)
                demo_grp.attrs['episode_reward'] = episode.get('episode_reward', 0)
                demo_grp.attrs['seed'] = episode.get('seed', -1)
                demo_grp.attrs['episode_idx'] = episode.get('episode_idx', ep_idx)
                demo_grp.attrs['video_path'] = episode.get('video_path', '')
                
                # Add states if available in infos
                if 'infos' in episode and len(episode['infos']) > 0:
                    # Try to extract states from infos (robomimic format)
                    states = []
                    for info in episode['infos']:
                        if isinstance(info, dict) and 'env_state' in info:
                            states.append(info['env_state'])
                    
                    if len(states) > 0:
                        demo_grp.create_dataset('states', data=np.array(states))
        
        print(f"Saved {len(episodes_to_save)} episodes to {output_path}")
        print(f"Summary saved to {summary_path}")
        
        # Print dataset statistics
        total_steps = sum(len(ep['actions']) for ep in episodes_to_save)
        successful_episodes = sum(1 for ep in episodes_to_save if ep.get('success', False))
        avg_reward = np.mean([ep['final_reward'] for ep in episodes_to_save])
        avg_length = total_steps / len(episodes_to_save)
        
        print(f"Dataset Statistics:")
        print(f"  Total steps: {total_steps}")
        print(f"  Average episode length: {avg_length:.1f}")
        print(f"  Average final reward: {avg_reward:.3f}")
        print(f"  Success rate in saved episodes: {successful_episodes/len(episodes_to_save)*100:.1f}%")

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect demonstration data from trained policy")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--config", help="Path to config file (optional)")
    parser.add_argument("--output", required=True, help="Output HDF5 file path")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes to collect")
    parser.add_argument("--start_seed", type=int, default=50000, help="Starting seed for episodes")
    parser.add_argument("--obs_keys", nargs='+', help="Observation keys to save")
    parser.add_argument("--output_dir", help="Directory to save videos and data")
    parser.add_argument("--save_videos", action="store_true", default=True, help="Save videos for each episode")
    parser.add_argument("--save_all", action="store_true", help="Save all episodes (not just successful ones)")
    parser.add_argument("--batch_size", type=int, help="Number of episodes to collect in parallel (default: use all envs)")
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = DemoCollector(args.checkpoint, args.config)
    
    # Collect episodes
    episodes = collector.collect_episodes(
        num_episodes=args.num_episodes, 
        start_seed=args.start_seed,
        output_dir=args.output_dir,
        save_videos=args.save_videos,
        batch_size=args.batch_size
    )
    
    # Save to file
    collector.save_to_hdf5(
        episodes, 
        args.output, 
        args.obs_keys, 
        save_all_episodes=args.save_all
    )

if __name__ == "__main__":
    main()