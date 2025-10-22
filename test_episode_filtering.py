#!/usr/bin/env python3

"""
Test script to verify episode filtering functionality in dataset classes.
"""

import os
import sys
import tempfile
import numpy as np
import h5py
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset


def create_test_hdf5_data(filepath, num_episodes=5, episode_length=10):
    """Create a test HDF5 file with dummy robomimic data."""
    with h5py.File(filepath, 'w') as f:
        data_group = f.create_group('data')
        
        for ep_idx in range(num_episodes):
            demo_group = data_group.create_group(f'demo_{ep_idx}')
            
            # Create obs group with dummy data
            obs_group = demo_group.create_group('obs')
            obs_group.create_dataset('object', data=np.random.rand(episode_length, 3))
            obs_group.create_dataset('robot0_eef_pos', data=np.random.rand(episode_length, 3))
            obs_group.create_dataset('robot0_eef_quat', data=np.random.rand(episode_length, 4))
            obs_group.create_dataset('robot0_gripper_qpos', data=np.random.rand(episode_length, 2))
            
            # Create actions
            actions = np.random.rand(episode_length, 7)  # 7D actions
            demo_group.create_dataset('actions', data=actions)


def test_robomimic_episode_filtering():
    """Test episode filtering for RobomimicReplayLowdimDataset."""
    print("Testing RobomimicReplayLowdimDataset episode filtering...")
    
    # Create temporary test data
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_file:
        test_file = tmp_file.name
    
    try:
        # Create test data with 5 episodes
        # create_test_hdf5_data(test_file, num_episodes=5, episode_length=10)
        test_file = "data/new_demos/square_mh_policy_demos_test.hdf5"
        finetune = True
        # "data/robomimic/datasets/square/mh/low_dim_abs.hdf5"
        # "data/new_demos/square_mh_policy_demos_test.hdf5"
        # Test 1: Load all episodes (default behavior)
        dataset_all = RobomimicReplayLowdimDataset(
            dataset_path=test_file,
            val_ratio=0.0,
            finetune=finetune
        )
        print(f"All episodes: {dataset_all.replay_buffer.n_episodes} episodes")
        # assert dataset_all.replay_buffer.n_episodes == 5, f"Expected 5 episodes, got {dataset_all.replay_buffer.n_episodes}"
        
        # Test 2: Load specific episodes
        selected_episodes = {0}
        dataset_filtered = RobomimicReplayLowdimDataset(
            dataset_path=test_file,
            val_ratio=0.0,
            episode_indices=selected_episodes,
            finetune=finetune
        )
        print(f"Filtered episodes {selected_episodes}: {dataset_filtered.replay_buffer.n_episodes} episodes")
        assert dataset_filtered.replay_buffer.n_episodes == 1, f"Expected 3 episodes, got {dataset_filtered.replay_buffer.n_episodes}"
        
        # Test 3: Test invalid episode indices
        try:
            invalid_dataset = RobomimicReplayLowdimDataset(
                dataset_path=test_file,
                val_ratio=0.0,
                episode_indices={0, 2, 310},  # 10 is invalid,
                finetune=finetune
            )
            assert False, "Should have raised ValueError for invalid indices"
        except ValueError as e:
            print(f"Correctly caught error for invalid indices: {e}")
        
        # Test 4: Verify data integrity
        # Get a sample from full dataset and filtered dataset
        sample_all = dataset_all[0]
        sample_filtered = dataset_filtered[0]
        
        # Both should have the same structure
        assert 'obs' in sample_all and 'action' in sample_all
        assert 'obs' in sample_filtered and 'action' in sample_filtered
        assert sample_all['obs'].shape[1] == sample_filtered['obs'].shape[1]  # Same obs dimension
        assert sample_all['action'].shape[1] == sample_filtered['action'].shape[1]  # Same action dimension
        
        print("✓ RobomimicReplayLowdimDataset episode filtering tests passed!")
        
    finally:
        pass
    #     # Clean up
    #     if os.path.exists(test_file):
    #         os.unlink(test_file)


if __name__ == "__main__":
    print("Testing episode filtering functionality...\n")
    
    test_robomimic_episode_filtering()
    
    print("\n✓ All tests completed!")