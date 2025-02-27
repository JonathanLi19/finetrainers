export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=2
MODEL_PATH="THUDM/CogVideoX-5b-I2V"
controlnet_path="checkpoints/box-ablation-segmentloss/checkpoint-3300.pt"

python training/cogvideox/I2V_inference_controlnet.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --pretrained_controlnet_path $controlnet_path \
    --validation_args_csv  "training/cogvideox/validation_args/testset/box_segment_loss_ablation/local_part/final_testset_part_3.csv" \
    --num_validation_videos 1 \
    --seed 42 \
    --height 480 \
    --width 720 \