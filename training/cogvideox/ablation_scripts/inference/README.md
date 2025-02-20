# Ablation Inference Scripts
## Dataset Ablation
1. 执行training/cogvideox/ablation_scripts/inference/inference_box_dataset_ablation_controlnet_I2V.sh
2. perception_head_path, controlnet_path替换为本地路径
3. --validation_args_csv 分别替换为四个part的路径(例如training/cogvideox/validation_args/testset/box_dataset_ablation/part/final_testset_part_1.csv), 然后依次执行脚本