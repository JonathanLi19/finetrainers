export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=3
MODEL_PATH="output_trajectory/cogvideox-sft__optimizer_adamw__steps_1000__lr-schedule_cosine_with_restarts__learning-rate_1e-3/checkpoint-800"

python training/cogvideox/I2V_inference_trajectory.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --validation_prompt "A rocket landing." \
    --validation_images "/home/qid/quanhao/workspace/Open-Sora/assets/images/condition/rocket/0.jpg" \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --validation_trajectory_maps "/home/qid/quanhao/workspace/Open-Sora/assets/boxs_trajectory/rocket_land" \
    --seed 42 \
    --height 480 \
    --width 720 \
    --output_path "samples/rocket.mp4"