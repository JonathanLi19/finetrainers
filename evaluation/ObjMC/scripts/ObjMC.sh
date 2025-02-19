#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# 将工作目录添加到 PYTHONPATH 环境变量
export PYTHONPATH=evaluation/ObjMC/Grounded_SAM2

# 运行 main.py
torchrun --master_addr=localhost --master_port=29500 evaluation/ObjMC/ObjMC.py \
        --video_path /datadrive2/lqh/finetrainers/samples/box_condition/checkpoint-12900/4259000.mp4 \
        --masks_path /datadrive2/lqh/testset_data/filtered_masks/4259000 \
        --main_objects man.