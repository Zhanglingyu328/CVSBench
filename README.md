<div align="center">

# 🌍 CVSBench

### Cross-View Spatial Reasoning and Dreaming Benchmark

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/zlyzlyzly/CVSBench)
[![License](https://img.shields.io/badge/License-CC--BY--4.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()

<h4>

<a href="PAPER_LINK">📄 Paper</a> | <a href="PROJECT_PAGE">🌐 Project Page</a> | <a href="https://huggingface.co/datasets/zlyzlyzly/CVSBench">🤗 Dataset</a>

</h4>

</div>

---

## 📖 Overview

CVSBench is a benchmark for evaluating multimodal foundation models on:

* 🧭 Cross-view spatial reasoning
* 🎯 Cross-view grounding
* 🛰️ Satellite ↔ Street-view understanding
* 🖼️ Visual imagination from partial observations

This repository contains the official evaluation toolkit used for CVSBench experiments.

---

## 📦 Dataset Download

The dataset is hosted on Hugging Face:

👉 https://huggingface.co/datasets/zlyzlyzly/CVSBench

Download and extract the dataset.

After extraction, place:

```text
fov/
cvusa/
```

directly inside:

```text
evaluate/
```

Required structure:

```text
evaluate/
├── eval.py
├── eval_double_category.py
├── summarize_results.py
├── eval_config.example.json
├── requirements.txt
├── fov/
└── cvusa/
```

> [!IMPORTANT]
>
> Evaluation scripts assume that both `fov/` and `cvusa/`
> are located directly under `evaluate/`.

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### OpenAI-Compatible APIs

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=http://localhost:8000/v1
export EVAL_MODEL=your_model

python eval.py --config eval_config.json
```

Supported:

* OpenAI API
* vLLM
* SGLang
* LMDeploy
* Other OpenAI-compatible servers

---

### Local Models

Currently supported:

```text
qwen3vl
gemma3
```

Example:

```bash
export LOCAL_TRANSFORMERS=1
export LOCAL_MODEL_FAMILY=qwen3vl
export LOCAL_MODEL_PATH=/path/to/model

python eval.py --config eval_config.json
```

---

## 📂 Repository Structure

| File                       | Description                                |
| -------------------------- | ------------------------------------------ |
| `eval.py`                  | Main evaluation entry point                |
| `eval_double_category.py`  | Two-image evaluation with auxiliary inputs |
| `summarize_results.py`     | Result aggregation and summarization       |
| `eval_config.example.json` | Example configuration                      |
| `requirements.txt`         | Dependencies                               |

---

## 🎯 Supported Tasks

| Task           | Description                   |
| -------------- | ----------------------------- |
| `g2s`          | Ground-to-Satellite reasoning |
| `s2g`          | Satellite-to-Ground reasoning |
| `ge_view`      | Cross-view matching           |
| `gs_grounding` | Cross-view grounding          |
| `mcq_vqa`      | Generic MCQ VQA               |
| `bbox_5level`  | Legacy grounding              |
| `arrow_5level` | Legacy localization           |
| `arrow_mcq`    | Legacy arrow tasks            |

---

## 🖼️ Two-Image Evaluation

Supported auxiliary inputs:

* `depth`
* `zimage`
* `nanobanana`

Example:

```bash
python eval_double_category.py \
    --base-dir . \
    --extra-kind nanobanana \
    --local-model-path /path/to/Qwen3-VL
```

---

## 📊 Summarizing Results

```bash
python summarize_results.py --root outputs
```

---

## 📁 Output Structure

```text
outputs/
└── model_name/
    ├── dataset_name/
    │   ├── predictions.jsonl
    │   └── metrics.json
    └── summary.json
```

---

## 🙏 Citation

```bibtex
@article{cvsbench2026,
  title={CVSBench: A Comprehensive Benchmark for Cross-View Spatial Reasoning and Dreaming},
  author={...},
  journal={ECCV},
  year={2026}
}
```

---

## ⚖️ License

CC-BY-4.0

---

## ⭐ Star History

If CVSBench is useful for your research, please consider giving this repository a star.
