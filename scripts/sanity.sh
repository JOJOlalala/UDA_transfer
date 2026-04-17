#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -P personal-qiao0042
#PBS -l select=1:ngpus=1:ncpus=16:mem=110gb
#PBS -l walltime=02:00:00
#PBS -N sanity

cd /scratch/users/ntu/qiao0042/models/acv/UDA_transfer
PYTHON=/home/users/ntu/qiao0042/scratch/conda/envs/uda_cyclegan/bin/python

echo "=== Sanity: 5-epoch Pixel CycleGAN on MNIST->USPS ==="
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

$PYTHON -u train.py --config configs/mnist_usps.yaml --mode pixel --epochs 5

echo "=== Done: $(date) ==="
