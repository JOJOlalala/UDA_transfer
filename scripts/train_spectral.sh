#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=cgan_spec
#SBATCH --output=logs/spectral_%j.out
#SBATCH --error=logs/spectral_%j.err

cd /scratch-share/QIAO0042/models/acv/UDA_trans

PYTHON=/home/msai/qiao0042/QIAO0042/.conda/envs/uda_cyclegan/bin/python

echo "=== Spectral CycleGAN ==="
echo "Date: $(date) | Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DATASET=${1:-mnist_usps}
BETA=${2:-0.05}

$PYTHON train.py \
    --config configs/${DATASET}.yaml \
    --mode spectral \
    --beta ${BETA}

echo "=== Done: $(date) ==="
