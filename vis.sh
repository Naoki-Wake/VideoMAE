# Set the path to save video
OUTPUT_DIR='./demo/vis_k400_1_0.9'
# path to video for visualization
VIDEO_PATH='./demo/segment_result.mp4'
# path to pretrain model
MODEL_PATH='/home/nawake/code/VideoMAE/checkpoint.pth'

python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}