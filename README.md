# VLMs Meme Inference

## Introduction
This repository provides code and tools for performing **Visual Language Model (VLM) inference on memes**. The goal is to leverage state-of-the-art VLMs to analyze and answer questions about meme images, enabling research in **multimodal reasoning and meme understanding**.

---

## Supported VLMs
Currently, the repository supports:
- BLIP2-Flan-T5-xl
- Qwen2-VL-7B-Instruct
- Qwen2.5-VL-7B-Instruct
- Qwen3-VL-8B-Instruct
- InstructBLIP-Vicunna-7B
- LLaVA-v1.5
- LLaVA-v1.6-Vicuna
- Pixtral-12B
---

## How to Use the Code

This [folder](tutorial/README.md) explains how to run inference on meme datasets using the provided VLMs.

---

## Results
You can find the results from [Google Drive](https://drive.google.com/drive/u/0/folders/1i_JanAQyi_Ou8X7v12JxcCivEoAaD5cz).

| VLMs                      | Overall Accuracy | ToxicityAssessment(307) | Scene(6)  | VisualMaterial(13) | TextualMaterial(10) | OverallIntent(5) | BackgroundKnowledge(9) | MarcoAverage |
|---------------------------|------------------|--------------------|--------|----------------|-----------------|---------------|---------------------|--------------|
| BLIP2-Flan-T5-xl          | 14.57%           | 14.65%             | 33.33% | 7.69% | 20% | 0% | 11.11% | 14.46% |
| InstructBLIP-Vicunna-7B.  | 22.00%           | 22.80%             | 16.66% | 7.69% | 20% | 20% | 22.22% | 18.23% |
| LLaVA-v1.6-Vicuna         | 48.86%           | 50.16%             | 50.00% | 46.15% | 20% | 20% |55.55% | 40.31% |
| LLaVA-v1.5                | 56.57%           | 54.72%             | 100%   | 76.92% | 50% | 80% | 55.55% | 69.53% |
| Pixtral-12B               | 70.00%           | 71.98%             | 66.66% | 53.84% | 60% | 20% | 66.66% | 56.52% |
| Qwen3-VL-8B-Instruct.     | 82.00%           | 80.13%             | 100%   | 100% | 80% | 100% | 100% | 93.35% | 
| Qwen2-VL-7B-Instruct      | 85.71%           | 84.36%             | 100%   | 92.30% | 90% | 100% | 100% | 94.44% |
| Qwen2.5-VL-7B-Instruct    | 87.71%           | 86.64%             | 100%   | 100% | 100% | 90% | 80% | 100% | 92.77% |
