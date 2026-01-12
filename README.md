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

| VLMs                      | Overall Accuracy | ToxicityAssessment |
|---------------------------|------------------|--------------------|
| BLIP2-Flan-T5-xl          | 14.57%           | 14.65%             |
| InstructBLIP-Vicunna-7B.  | 22.00%           | 22.80%             |
| LLaVA-v1.6-Vicuna         | 48.86%           |                    |
| LLaVA-v1.5                | 56.57%           |                    |
| Pixtral-12B               | 70.00%           |                    |
| Qwen3-VL-8B-Instruct.     | 82.00%           |                    |
| Qwen2-VL-7B-Instruct      | 85.71%           |                    |
| Qwen2.5-VL-7B-Instruct    | 87.71%           |                    |
