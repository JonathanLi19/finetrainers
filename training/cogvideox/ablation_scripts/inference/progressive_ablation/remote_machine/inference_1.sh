export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="THUDM/CogVideoX-5b-I2V"
perception_head_path="checkpoints/box_ablation_progressive/checkpoint-1500/perception_head-checkpoint-1500.pt"
controlnet_path="checkpoints/box_ablation_progressive/checkpoint-1500/controlnet-checkpoint-1500.pt"

python training/cogvideox/I2V_inference_controlnet.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --pretrained_controlnet_path $controlnet_path \
    --pretrained_perception_head_path $perception_head_path \
    --validation_args_csv  "training/cogvideox/validation_args/DAVIS/box_progressive_ablation/remote_machine/step1500.csv" \
    --num_validation_videos 1 \
    --seed 42 \
    --height 480 \
    --width 720 \
    --use_perception_head
