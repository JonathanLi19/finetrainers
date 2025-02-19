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
GPU_IDS="2"

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
TRAJECTORY_MAPS_TYPE="box"
frame_interval=1
output_dir="/datadrive2/lqh/cogvideox/box-ablation-progressive/Pexels_MeViS_MOSE_DAVIS/Controlnet"

# 获取最新的 checkpoint 目录
while true; do
  latest_checkpoint=$(ls -d $output_dir/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
  if [[ -z "$latest_checkpoint" ]]; then
    latest_step=0
    pretrained_controlnet_path=""
    pretrained_perception_head_path=""
  else
    latest_step=$(basename "$latest_checkpoint" | awk -F'-' '{print $2}')
    pretrained_controlnet_path="$latest_checkpoint/controlnet-checkpoint-$latest_step.pt"
    pretrained_perception_head_path="$latest_checkpoint/perception_head-checkpoint-$latest_step.pt"
  fi
  echo "Using Controlnet checkpoint: $pretrained_controlnet_path & Perception Head checkpoint:$pretrained_perception_head_path with step: $latest_step"

  for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for epoch in "${EPOCHS[@]}"; do

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
            --seed 42 \
            --mixed_precision bf16 \
            --output_dir $output_dir \
            --max_num_frames 49 \
            --train_batch_size 1 \
            --num_train_epochs $epoch \
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
            --controlnet_weights 1.0 \
            --use_perception_head \
            --lambda_latent_segmentation 0.5 \
            --initial_global_step $latest_step \
            --global_step $latest_step"

          # 根据 initial_step 选择是否使用 `--pretrained_controlnet_path` 或 `--init_from_transformer`
          if [[ -n "$pretrained_controlnet_path" ]]; then
            cmd+=" --pretrained_controlnet_path $pretrained_controlnet_path"
          else
            cmd+=" --init_from_transformer"
          fi
          if [[ -n "$pretrained_perception_head_path" ]]; then
            cmd+=" --pretrained_perception_head_path $pretrained_perception_head_path"
          fi

          echo "Running command: $cmd"
          eval $cmd
          echo -ne "-------------------- Finished executing script --------------------\n\n"
        done
      done
    done
  done
done