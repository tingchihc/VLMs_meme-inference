#!/bin/bash
#SBATCH --job-name=pixtral_vqa
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=25:00:00
#SBATCH --output=slurm_logs/pixtral_vqa_%j.out
#SBATCH --error=slurm_logs/pixtral_vqa_%j.err

# Create log directory
LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"

module load 2024
module load CUDA/12.6.0

nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh

conda activate pixtral

python VLMs_inference.py  --model_name Pixtral-12B --conda_env pixtral --input_folder datasets/meme/benchmark_extracted --output_folder Pixtral-12B
