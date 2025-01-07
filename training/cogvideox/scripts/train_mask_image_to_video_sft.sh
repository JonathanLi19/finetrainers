export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="online"
# export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16
GPU_IDS="0,1,2"

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-4")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("1000")

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/my_config.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir /path/to/my/datasets/disney-dataset
DATA_ROOT="/home/qid/quanhao/workspace/Open-Sora/data/DAVIS/DAVIS_data.csv"
MODEL_PATH="THUDM/CogVideoX-5b-I2V"
TRAJECTORY_MAPS_TYPE="mask"
frame_interval=1

# Set ` --load_tensors ` to load tensors from disk instead of recomputing the encoder process.
# Launch experiments with different hyperparameters

for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="/datadrive2/cogvideox/cpu_offload_optimizer/mask/cogvideox-sft__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}__trajectory_rotary_embeddings/"

        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE\
          --gpu_ids $GPU_IDS \
          training/cogvideox/cogvideox_trajectory_image_to_video_sft.py \
          --pretrained_model_name_or_path  $MODEL_PATH \
          --dataset_file $DATA_ROOT \
          --trajectory_maps_type $TRAJECTORY_MAPS_TYPE \
          --frame_interval $frame_interval \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 8 \
          --pin_memory \
          --validation_prompt \"A boat sailing in the river.\" \
          --validation_images \"/home/qid/quanhao/workspace/Open-Sora/assets/images/condition/boat.png\" \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_steps 50 \
          --validation_trajectory_maps \"/home/qid/quanhao/workspace/Open-Sora/assets/mask_trajectory/boat/moved_mask_right\" \
          --trajectory_guidance_scale 2 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size 1 \
          --max_train_steps $steps \
          --checkpointing_steps 100 \
          --gradient_accumulation_steps 1 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 200 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --noised_image_dropout 0.05 \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --nccl_timeout 1800 \
          --use_cpu_offload_optimizer "
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done