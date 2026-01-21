#!/bin/bash
#SBATCH --job-name=blip2_vqa
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=25:00:00
#SBATCH --output=slurm_logs/blip2_vqa_%j.out
#SBATCH --error=slurm_logs/blip2_vqa_%j.err

# Create log directory
LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"

module load 2024
module load CUDA/12.6.0

nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh

conda activate blip2_vqa

python VLMs_inference.py  --model_name BLIP2-Flan-T5-xl --conda_env blip2_vqa --input_folder datasets/meme/benchmark_extracted --output_folder BLIP2-Flan-T5-xl
