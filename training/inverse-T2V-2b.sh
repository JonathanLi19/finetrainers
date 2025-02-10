export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=2
MODEL_PATH="THUDM/CogVideoX-2b"

python training/cogvideox/inverse_T2V_2b.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --validation_prompt "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting." \
    --validation_images "" \
    --validation_trajectory_maps "samples/cogvideox/boat_t2v_2b_cogvideox_ddim.mp4" \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --seed 42 \
    --height 480 \
    --width 720