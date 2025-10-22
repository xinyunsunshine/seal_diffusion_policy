# chmod +x single_ckpt_eval.sh
CKPT="data/outputs/square/sft0/checkpoints/epoch=0750-test_mean_score=0.200.ckpt"
# "/home/sunsh16e/seal_diffusion_policy/data/outputs/square/pre_train/checkpoints/epoch=0750-test_mean_score=0.440.ckpt"
OUTPUT_DIR="data/outputs/square/sft0/eval_test"
python eval.py --checkpoint $CKPT --output_dir $OUTPUT_DIR --device cuda:0