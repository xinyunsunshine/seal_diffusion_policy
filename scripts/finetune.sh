#!/bin/bash
# chmod +x train_eval.sh
# ./train_eval.sh
# Ensure the script stops on error
set -e

# Define the command with parameters
CONFIG_DIR="config"
CONFIG_NAME="dp_square_mh.yaml"
SEED=42
DEVICE="cuda:0"
EPOCH_NUM=5
CKPT_EVERY=1
VAL_EVERY=10
MAX_TRAIN_EPISODES=300

# HYDRA_RUN_DIR="data/outputs/$(date +%Y.%m.%d)/$(date +%H.%M.%S)_lr${LR}_sigma${X_SIGMA}_k${k}" #the old dir name
HYDRA_RUN_DIR="data/outputs/square/sft2"
LOG_NAME="square_training_sft2"
PRETRAINED_CKPT_PATH="'data/outputs/square/pre_train/checkpoints/epoch=0750-test_mean_score=0.440.ckpt'"
DATAPATH="data/new_demos/square_mh_policy_demos_test.hdf5"

# Overwrite flag (set to true to re-run evaluation)
OVERWRITE=false

# Run the training script
python train.py \
    --config-dir=$CONFIG_DIR \
    --config-name=$CONFIG_NAME \
    training.finetune=true \
    training.pretrained_ckpt_path=$PRETRAINED_CKPT_PATH \
    training.seed=$SEED \
    training.device=$DEVICE \
    logging.name="$LOG_NAME" \
    hydra.run.dir="$HYDRA_RUN_DIR"\
    training.num_epochs=$EPOCH_NUM \
    task.dataset.max_train_episodes=$MAX_TRAIN_EPISODES \
    training.val_every=$VAL_EVERY\
    training.checkpoint_every=$CKPT_EVERY\
    training.rollout_every=$CKPT_EVERY\
    # task.dataset.dataset_path=$DATAPATH\
    # task.dataset.episode_indices="[0]"

# Save the directory path
echo "$HYDRA_RUN_DIR" > last_run_dir.txt


# Run evaluation script
# # ./eval_nokp.sh "$HYDRA_RUN_DIR"
# # Run evaluation script
# if [ "$OVERWRITE" = true ]; then
#     python eval_process_nokp.py "$HYDRA_RUN_DIR" --overwrite
# else
#     python eval_process_nokp.py "$HYDRA_RUN_DIR"
# fi