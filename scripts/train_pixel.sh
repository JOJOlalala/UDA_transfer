#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=cgan_pixel
#SBATCH --output=logs/pixel_%j.out
#SBATCH --error=logs/pixel_%j.err

cd /scratch-share/QIAO0042/models/acv/UDA_trans

PYTHON=/home/msai/qiao0042/QIAO0042/.conda/envs/uda_cyclegan/bin/python

echo "=== Pixel CycleGAN ==="
echo "Date: $(date) | Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DATASET=${1:-mnist_usps}

$PYTHON train.py \
    --config configs/${DATASET}.yaml \
    --mode pixel

echo "=== Done: $(date) ==="
