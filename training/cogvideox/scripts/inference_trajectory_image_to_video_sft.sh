export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=3
MODEL_PATH="/datadrive2/cogvideox/cpu_offload_optimizer/mask/Joint_Attention/cogvideox-sft__optimizer_adamw__steps_10000__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-400"

python training/cogvideox/I2V_inference_trajectory.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --validation_prompt "A hot air balloon moving in the sky." \
    --validation_images "/home/qid/quanhao/workspace/Open-Sora/assets/images/condition/balloon.png" \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --validation_trajectory_maps "/home/qid/quanhao/workspace/Open-Sora/assets/mask_trajectory/balloon/moved_mask_right/mask.mp4" \
    --trajectory_guidance_scale 8 \
    --seed 42 \
    --height 480 \
    --width 720 \
    --output_path "samples/mask_condition/balloon_mask_step400_trajectoryscale_8.mp4"