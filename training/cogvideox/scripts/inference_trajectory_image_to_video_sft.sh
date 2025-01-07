export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=3
MODEL_PATH="/datadrive2/cogvideox/cpu_offload_optimizer/mask/cogvideox-sft__optimizer_adamw__steps_1000__lr-schedule_cosine_with_restarts__learning-rate_1e-4__trajectory_rotary_embeddings/checkpoint-700"

python training/cogvideox/I2V_inference_trajectory.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --validation_prompt "A boat sailing in the river." \
    --validation_images "/home/qid/quanhao/workspace/Open-Sora/assets/images/condition/boat.png" \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --validation_trajectory_maps "/home/qid/quanhao/workspace/Open-Sora/assets/mask_trajectory/boat/moved_mask_right" \
    --trajectory_guidance_scale 8 \
    --seed 42 \
    --height 480 \
    --width 720 \
    --output_path "samples/mask_condition/boat_right_mask_step700_trajectoryscale_8.mp4"