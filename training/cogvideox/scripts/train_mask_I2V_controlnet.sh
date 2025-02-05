export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16
GPU_IDS="0,1,2"

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-5")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
EPOCHS=("1")

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir /path/to/my/datasets/disney-dataset
DATA_ROOT="data/Pexels/Pexels_MeViS_MOSE.csv"
MODEL_PATH="THUDM/CogVideoX-5b-I2V"
TRAJECTORY_MAPS_TYPE="mask"
frame_interval=1
output_dir="/datadrive2/cogvideox/mask/Pexels_MeViS_MOSE_DAVIS/Controlnet"

get_latest_checkpoint() {
  # 如果 output_dir 不存在，则创建它
  if [[ ! -d "$output_dir" ]]; then
    echo "Output directory does not exist. Creating: $output_dir"
    mkdir -p "$output_dir"
  fi

  # 检查 output_dir 是否为空
  if [[ -z "$(ls -A "$output_dir")" ]]; then
    echo "Output directory is empty. Setting initial_step to 0."
    echo "0"
    return
  fi

  # 获取最新的 checkpoint
  latest_checkpoint=$(ls -t ${output_dir}/checkpoint-*.pt 2>/dev/null | head -n 1)
  
  if [[ -z "$latest_checkpoint" ]]; then
    echo "No checkpoint found. Setting initial_step to 0."
    echo "0"
    return
  fi

  echo "$latest_checkpoint"
}

while true; do
  # 获取最新 checkpoint 和其对应的 step
  controlnet_path=$(get_latest_checkpoint)
  
  # 如果 latest_step 是 "0"，意味着没有 checkpoint
  if [[ "$controlnet_path" == "0" ]]; then
    latest_step=0
    use_pretrained_controlnet=false
  else
    latest_step=$(basename "$controlnet_path" | grep -oE '[0-9]+')
    use_pretrained_controlnet=true
  fi

  echo "Using checkpoint: $controlnet_path with step: $latest_step"
  
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for epoch in "${EPOCHS[@]}"; do
          # 基础命令
          cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
            --gpu_ids $GPU_IDS \
            training/cogvideox/cogvideox_controlnet_I2V_sft.py \
            --pretrained_model_name_or_path $MODEL_PATH \
            --dataset_file $DATA_ROOT \
            --trajectory_maps_type $TRAJECTORY_MAPS_TYPE \
            --frame_interval $frame_interval \
            --height_buckets 480 \
            --width_buckets 720 \
            --frame_buckets 49 \
            --dataloader_num_workers 8 \
            --pin_memory \
            --trajectory_guidance_scale 2 \
            --seed 42 \
            --mixed_precision bf16 \
            --output_dir $output_dir \
            --max_num_frames 49 \
            --train_batch_size 1 \
            --num_train_epochs $epoch \
            --checkpointing_steps 500 \
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
            --controlnet_weights 1.0 \
            --initial_global_step $latest_step \
            --global_step $latest_step"
          
          # 根据 initial_step 选择是否使用 `--pretrained_controlnet_path` 或 `--init_from_transformer`
          if [[ "$use_pretrained_controlnet" == true ]]; then
            cmd+=" --pretrained_controlnet_path $controlnet_path"
          else
            cmd+=" --init_from_transformer"
          fi

          echo "Running command: $cmd"
          eval $cmd
          echo -ne "-------------------- Finished executing script --------------------\n\n"
        done
      done
    done
  done
done