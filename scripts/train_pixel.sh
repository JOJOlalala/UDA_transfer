#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=cgan_pixel
#SBATCH --output=logs/pixel_%j.out
#SBATCH --error=logs/pixel_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate uda_cyclegan

cd /home/msai/qiao0042/QIAO0042/models/acv/UDA_trans

echo "=== Pixel CycleGAN ==="
echo "Date: $(date) | Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DATASET=${1:-mnist_usps}

python train.py \
    --config configs/${DATASET}.yaml \
    --mode pixel

echo "=== Done: $(date) ==="
