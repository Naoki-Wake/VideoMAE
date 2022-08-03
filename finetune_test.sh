OUTPUT_DIR='./demo/debug'
DATA_PATH='/home/nawake/sthv2/'
MODEL_PATH='/home/nawake/code/VideoMAE/checkpoint.pth'

OMP_NUM_THREADS=1 python run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set HOUSEHOLD \
    --nb_classes 11 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --enable_deepspeed 