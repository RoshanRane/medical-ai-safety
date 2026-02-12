#!/bin/bash

mkdir -p _nohup

for i in 10 40 50 60 90; do  
    # gpu_index=$(( (i / 10) % 8 ))
    python -m model_training.start_training --config_file config_files/training/isic/vgg16_microscope_${i}.yaml &> _nohup/train_m${i}.out &
done

echo "All jobs launched. PIDs:"
jobs -p
disown -a