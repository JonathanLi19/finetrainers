export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=3
MODEL_PATH="THUDM/CogVideoX-5b-I2V"

python training/cogvideox/inverse.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --validation_prompt "A boat sailing in the river." \
    --validation_images "/home/qid/quanhao/workspace/Open-Sora/assets/images/condition/boat.png" \
    --validation_trajectory_maps "data/DAVIS/JPEGImages/480p/boat/boat_49.mp4" \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --seed 42 \
    --height 480 \
    --width 720