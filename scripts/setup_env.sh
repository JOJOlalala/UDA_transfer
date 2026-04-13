#!/bin/bash
# Setup conda environment for UDA CycleGAN project
module load anaconda
eval "$(conda shell.bash hook)"

conda create -n uda_cyclegan python=3.10 -y
conda activate uda_cyclegan

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install pyyaml tensorboard matplotlib pillow scipy tqdm
pip install pytorch-fid

echo "Environment 'uda_cyclegan' created successfully."
