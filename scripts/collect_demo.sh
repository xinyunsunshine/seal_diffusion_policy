#!/bin/bash
# chmod +x scripts/collect_demo.sh

# Set output directory for videos and data
OUTPUT_DIR="data/new_demos/square_mh_collection_test"

# Use epoch 500 which had the highest score (0.32)
python collect_demo_data.py \
    --checkpoint data/outputs/square/pre_train/checkpoints/epoch=0750-test_mean_score=0.440.ckpt \
    --output data/new_demos/square_mh_policy_demos_test.hdf5 \
    --config config/dp_square_mh.yaml \
    --num_episodes 2 \
    --start_seed 100000 \
    --output_dir "$OUTPUT_DIR" \
    --save_videos \
    --save_all