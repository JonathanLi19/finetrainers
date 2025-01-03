export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=3

python training/cogvideox/I2V_lora_inference.py \
    --pretrained_model_name_or_path "output/cogvideox-lora__optimizer_adamw__steps_1000__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-1000" \
    --validation_prompt "BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions" \
    --validation_images "/home/qid/quanhao/workspace/Open-Sora/visualization/debug/frame_0000.jpg" \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --validation_epochs 1 \
    --seed 42 \
    --output_dir "output/inference"