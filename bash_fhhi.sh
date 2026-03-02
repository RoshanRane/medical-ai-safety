#!/bin/bash
# mkdir -p _nohup

## 1) Generate data
## a) microscope artifact in y=MEL
# for i in 5 10 20 30 40 50 60 70 80 90; do  
for i in 5; do
    p_value=$(echo "scale=2; $i / 100" | bc)
    cmd="python generate_attacked_dataset.py 
        --dataset_name isic \
        --dataset_path /ritter/roshan/workspace/ICON/experiments/melanoma/dataset/ \
        --attacked_classes 0 \
        --artifact_type microscope \
        --p_artifact $p_value &> _nohup/01_m${i}.out &"
    echo "Running: $cmd"
    eval $cmd
    sleep 0.5 # slight delay to avoid overwhelming the scheduler with simultaneous job launches
done

# ## b) timestamp artifact in microscope artifact
# for i in 10 20 30 40 50 60 70 80 90; do    
#     p_value=$(echo "scale=2; $i / 100" | bc)
#     python generate_attacked_dataset.py \
#         --dataset_name isic \
#         --dataset_path /ritter/roshan/workspace/ICON/experiments/melanoma/dataset/microscope-40/ \
#         --attacked_artifact microscope \
#         --artifact_type timestamp \
#         --p_artifact_to_artifact "$p_value" &> "_nohup/timestamp_${i}.out" &


## 2) Start training
# for i in 20 70 80 90; do  
#     python -m model_training.start_training --config_file config_files/training/isic/vgg16_microscope-${i}.yaml &> _nohup/train_m${i}.out &
# done

## 3) Copy trained models to ICON directory
# for dir in /ritter/roshan/workspace/medical-ai-safety/model_checkpoints/vgg16_*/; do
#   ckpt_name=$(basename "$dir" | sed 's/^vgg16_//')
#   ckpt_name=${ckpt_name//_/-}
#   cp "$dir/checkpoint_vgg16_last.pth" \
#      /ritter/roshan/workspace/ICON/experiments/melanoma/output/isic_y-lbl-NV__vgg16/trial_0/checkpoints/checkpoint_vgg16_isic-attacked_${ckpt_name}.pth
# done

echo "All jobs launched. PIDs:"
jobs -p
disown -a
