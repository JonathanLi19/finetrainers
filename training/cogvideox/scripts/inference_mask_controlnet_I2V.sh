export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=3
MODEL_PATH="THUDM/CogVideoX-5b-I2V"

python training/cogvideox/I2V_inference_controlnet.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --validation_args_csv  "training/cogvideox/validation_args/testset.csv" \
    --num_validation_videos 1 \
    --seed 42 \
    --height 480 \
    --width 720 \
    --pretrained_controlnet_path "checkpoints/mask/checkpoint-17000.pt"
