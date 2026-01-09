#!/bin/bash
#SBATCH --job-name=llava1.6_vqa
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=25:00:00
#SBATCH --output=slurm_logs/llava1.6_vqa_%j.out
#SBATCH --error=slurm_logs/llava1.6_vqa_%j.err

# Create log directory
LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"

module load 2024
module load CUDA/12.6.0

nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh

conda activate vicuna_vlm

python VLMs_inference.py  --model_name LLaVA-v1.6-Vicuna --conda_env vicuna_vlm --input_folder /home/tchen1/workshop/datasets/meme/benchmark_extracted --output_folder LLaVA-v1.6-Vicuna
