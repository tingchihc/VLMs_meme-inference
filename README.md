# M-QUEST: VLMs Meme Inference

[![arXiv](https://img.shields.io/badge/arXiv-2603.03315-b31b1b.svg)](https://arxiv.org/abs/2603.03315) [![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-blue)](https://aclanthology.org/) [![Dataset](https://img.shields.io/badge/🤗%20Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/vulr/M-QUEST)

<p align="center">
<img src="images/framework-no-ke.png" alt="intro" width="600" height="600"/>
</p>

## Introduction

This repository contains the VLMs inference pipeline developed as part of the [**Semantic Memes**](https://github.com/StenDoipanni/semantic-memes) project. It provides a unified framework for running **Visual Language Models (VLMs)** on meme images and collecting their responses to questions about meme content, semantics, and toxicity.

The inference pipeline supports a range of state-of-the-art VLMs, enabling systematic evaluation of their ability to **understand and reason about multimodal meme content**. The generated model responses can be used for downstream evaluation with **M-QUEST (Meme Question-Understanding Evaluation on Semantics and Toxicity)**, a benchmark designed to assess VLM performance on different aspects of meme understanding.

The repository currently provides inference implementations for several popular VLMs, together with instructions for running them on meme datasets. This makes it possible to reproduce and extend the VLM experiments conducted in our work.


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

## Citation

This repository is part of the [**Semantic Memes**](https://github.com/StenDoipanni/semantic-memes) project and contains the VLM inference code used in our experiments.

If you find this work useful, please cite:

```bibtex
@misc{degiorgis2026mquestmemequestionunderstanding,
      title={M-QUEST -- Meme Question-Understanding Evaluation on Semantics and Toxicity}, 
      author={Stefano De Giorgis and Ting-Chih Chen and Filip Ilievski},
      year={2026},
      eprint={2603.03315},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.03315}, 
}
```
