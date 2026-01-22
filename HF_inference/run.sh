#!/bin/bash
#SBATCH --job-name=HF_test
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/HF_test_%j.out
#SBATCH --error=slurm_logs/HF_test_%j.err

LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"

module load 2024
module load CUDA/12.6.0

nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh

conda activate allenvs

# ==== BLIP2-Flan-T5-xl ====
python -u VLMs_inferences.py \
  --model_name BLIP2-Flan-T5-xl \
  --output_folder results

echo "-----------BLIP2-Flan-T5-xl------------------"

# ==== InstructBLIP-Vicunna-7B ====
python -u VLMs_inferences.py \
 --model_name InstructBLIP-Vicunna-7B \
 --output_folder results

echo "-----------InstructBLIP-Vicunna-7B------------------"

# ==== LLaVA-v1.5 ===
python -u VLMs_inferences.py \
  --model_name LLaVA-v1.5 \
  --output_folder results

echo "-----------LLaVA-v1.5------------------"

# ==== LLaVA-v1.6-Vicuna ===
python -u VLMs_inferences.py \
  --model_name LLaVA-v1.6-Vicuna \
  --output_folder results

echo "-----------LLaVA-v1.6-Vicuna------------------"

# ==== Pixtral-12B ===
python -u VLMs_inferences.py \
  --model_name Pixtral-12B \
  --output_folder results

echo "-----------Pixtral-12B------------------"

# ==== Qwen2-VL-7B ===
python -u VLMs_inferences.py \
  --model_name Qwen2-VL-7B-Instruct \
  --output_folder results

echo "-----------Qwen2-VL-7B------------------"

# ==== Qwen2.5 ===
python -u VLMs_inferences.py \
  --model_name Qwen2.5-VL-7B-Instruct \
  --output_folder results

echo "-----------Qwen2.5-VL-7B------------------"

# ==== Qwen3 ===
python -u VLMs_inferences.py \
  --model_name Qwen3-VL-8B-Instruct \
  --output_folder results

echo "-----------Qwen3-VL-8B------------------"  