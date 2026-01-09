# VLMs Meme Inference — Usage Guide

## 1. Clone the Repository

```bash
git clone https://github.com/tingchihc/VLMs_meme-inference.git
cd VLMs_meme-inference
```

## 2. Create the Conda Environment 
We have multiple Conda environments. Here is the example for BLIP2-Flan-T5-xl.

```
conda env create -f blip2_environment.yml
conda activate blip2_vqa
```

| VLMs                     | Conda Environment YAML        |
|---------------------------|------------------------------|
| BLIP2-Flan-T5-xl          | blip2_environment.yml        |
| Qwen2-VL-7B-Instruct      | qwen2vl_blip2_environment.yml|
| Qwen2.5-VL-7B-Instruct    | qwen25vl_environment.yml     |
| InstructBLIP-Vicunna-7B.  | instructblip_environment.yml |
| LLaVA-v1.5                | llava1.5_environment.yml     |
| LLaVA-v1.6-Vicuna         | llava1.6_environment.yml     |
| Pixtral-12B               | pixtral_environment.yml      |

## 3. Submit the Job

```
sbatch blip2_work.sh
```
