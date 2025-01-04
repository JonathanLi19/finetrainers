export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=3
MODEL_PATH="THUDM/CogVideoX-5b-I2V"

python training/cogvideox/I2V_inference.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --validation_prompt "A rocket landing." \
    --validation_images "/home/qid/quanhao/workspace/Open-Sora/assets/images/condition/rocket/0.jpg" \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --seed 42 \
    --height 480 \
    --width 720