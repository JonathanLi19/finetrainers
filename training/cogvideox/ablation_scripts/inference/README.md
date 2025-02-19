# Ablation Inference Scripts
## Dataset Ablation
1. 执行training/cogvideox/ablation_scripts/inference/inference_box_dataset_ablation_controlnet_I2V.sh
2. perception_head_path, controlnet_path替换为本地路径
3. --validation_args_csv 替换为本地的csv文件，注意如果想在多GPU上执行这个脚本，可以把一个csv文件等分成4份，然后分别对每一个csv文件执行这个脚本即可