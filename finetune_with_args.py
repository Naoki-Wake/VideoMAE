import os.path as osp
import os
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)
    parser.add_argument('--lr', default=0.0075, type=float)
    parser.add_argument('--household_fp_train', default='annotations/wo_pseudo/breakfast_train_list_videos.txt',
                        help='file path to the train file')
    parser.add_argument('--household_fp_test', default='annotations/wo_pseudo/breakfast_test_list_videos.txt',
                        help='file path to the test file')
    parser.add_argument('--household_fp_val', default='annotations/wo_pseudo/breakfast_val_list_videos.txt',
                        help='file path to the val file')
    parser.add_argument('--out_dir', default='/lfovision_log/videomae/finetune/experiment_paramsearch/LR_',
                        help='outdir')
    #annotations/with_pseudo_largedatanum
    args = parser.parse_args()
    LR=args.lr
    OUTPUT_DIR=args.out_dir+str(LR)
    DATA_PATH='/lfovision_sthv2_breakfast/'
    MODEL_PATH='/lfovision_pretrained_models/videomae/pretraining/sthv2/checkpoint.pth'
    train_command = "OMP_NUM_THREADS=1 python run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set HOUSEHOLD \
        --nb_classes 11 \
        --data_path " + DATA_PATH + " " + \
        "--finetune " + MODEL_PATH + " " + \
        "--log_dir " + OUTPUT_DIR + " " + \
        "--output_dir " + OUTPUT_DIR + " " + \
        "--batch_size 1 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --opt adamw \
        --lr " +str(LR) + " " + \
        "--opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 50 \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --enable_deepspeed \
        --household_fp_train " + args.household_fp_train + " " + \
        "--household_fp_test " + args.household_fp_test + " " + \
        "--household_fp_val " + args.household_fp_val
    import subprocess
    print(train_command)
    os.system(train_command)