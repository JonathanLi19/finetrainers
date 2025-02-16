# Ablation Studies 运行脚本
## Segment Loss Ablation (不采用segment loss训练二阶段box condition)
1. 执行 `bash training/cogvideox/ablation_scripts/train_box_ablation_segment_loss.sh`
2. GPU_IDS换成自己的，比如"0,1,2,3"
3. accelerate_configs/deepspeed.yaml这里num_processes换成gpu个数比如4
4. mask_controlnet_path换成自己的checkpoint路径
5. output_dir换成自己的checkpoint保存路径
6. DATA_ROOT换成自己本地的训练集的csv文件
## Progressive Training Ablation1 （直接训box condition， 不用mask pretrain）
1. 执行 `bash training/cogvideox/ablation_scripts/train_box_ablation_progressive.sh`
2. GPU_IDS换成自己的，比如"0,1,2,3"
3. accelerate_configs/deepspeed.yaml这里num_processes换成gpu个数比如4
4. output_dir换成自己的checkpoint保存路径
5. DATA_ROOT换成自己本地的训练集的csv文件
## Dataset Ablation （不用我自己构建的data，只用MeViS和MOSE去train二阶段box condition）
1. 执行 `bash training/cogvideox/ablation_scripts/train_dataset_ablation.sh`
2. GPU_IDS换成自己的，比如"0,1,2,3"
3. accelerate_configs/deepspeed.yaml这里num_processes换成gpu个数比如4
4. output_dir换成自己的checkpoint保存路径
5. DATA_ROOT换成自己本地的消融训练集的csv文件
6. mask_controlnet_path换成自己的checkpoint路径