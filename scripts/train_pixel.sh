#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -P personal-qiao0042
#PBS -l select=1:ngpus=1:ncpus=16:mem=110gb
#PBS -l walltime=24:00:00
#PBS -N cgan_pixel

cd /scratch/users/ntu/qiao0042/models/acv/UDA_transfer
PYTHON=/home/users/ntu/qiao0042/scratch/conda/envs/uda_cyclegan/bin/python

# SSH tunnel for wandb
PROXY_PORT=$((20000 + RANDOM % 10000))
ssh -f -N -D $PROXY_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=5 asp2a-login-ntu01 2>/dev/null
export HTTPS_PROXY=socks5h://localhost:$PROXY_PORT
export HTTP_PROXY=socks5h://localhost:$PROXY_PORT

echo "=== Pixel CycleGAN ==="
echo "Date: $(date) | Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Pass dataset via: qsub -v DATASET=mnist_usps scripts/train_pixel.sh
DATASET=${DATASET:-mnist_usps}

$PYTHON -u train.py \
    --config configs/${DATASET}.yaml \
    --mode pixel

echo "=== Done: $(date) ==="
