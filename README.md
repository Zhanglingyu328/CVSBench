<div align="center">

# 🌍 CVSBench Evaluation Toolkit

### A Benchmark for Cross-View Spatial Reasoning and Dreaming

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/zlyzlyzly/CVSBench)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

</div>

---

## 📖 Overview

**CVSBench** is a benchmark designed to evaluate the **cross-view spatial reasoning**, **cross-view grounding**, and **visual imagination** capabilities of multimodal foundation models.

This repository provides the official evaluation toolkit used in CVSBench experiments, including:

* 📝 Cross-view VQA evaluation
* 🎯 Cross-view grounding evaluation
* 📊 Category-aware performance analysis
* 🖼️ Multi-image evaluation with auxiliary visual inputs
* 🤖 Evaluation of OpenAI-compatible APIs and local vision-language models

> [!IMPORTANT]
>
> This repository contains **evaluation code only**.
>
> Dataset files are **NOT included** and must be downloaded separately from Hugging Face before running any evaluation.

---

# 📦 Dataset Download

Download the official CVSBench dataset from:

### 🤗 Hugging Face Dataset

https://huggingface.co/datasets/zlyzlyzly/CVSBench

After downloading and extracting the dataset, place the **`fov`** and **`cvusa`** directories directly inside the **`evaluate/`** folder.

Required directory structure:

```text
evaluate/
├── eval.py
├── eval_double_category.py
├── summarize_results.py
├── eval_config.example.json
├── requirements.txt
├── fov/
│   ├── g2s/
│   ├── s2g/
│   ├── ge_view/
│   └── ...
└── cvusa/
    ├── images/
    ├── annotations/
    └── ...
```

> [!WARNING]
>
> Evaluation scripts assume that both **`fov/`** and **`cvusa/`** are located directly inside the **`evaluate/`** directory.
>
> Moving these folders elsewhere may cause image loading and dataset path resolution failures.

---

# 🚀 Quick Start

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-repo/CVSBench.git
cd CVSBench/evaluate
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Download Dataset

Download:

```text
https://huggingface.co/datasets/zlyzlyzly/CVSBench
```

Then place:

```text
fov/
cvusa/
```

inside:

```text
evaluate/
```

Final structure:

```text
evaluate/
├── eval.py
├── eval_double_category.py
├── summarize_results.py
├── fov/
└── cvusa/
```

---

## 4️⃣ Run Evaluation

### OpenAI-Compatible APIs

Configure your API settings:

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=http://localhost:8000/v1
export EVAL_MODEL=your-model
```

Run:

```bash
python eval.py --config eval_config.json
```

Compatible with:

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

> [!TIP]
>
> Additional model families can be easily added by implementing new adapters in `eval.py`.

---

# 📂 Repository Structure

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

| File                       | Description                             |
| -------------------------- | --------------------------------------- |
| `eval.py`                  | Main evaluation entry point             |
| `eval_double_category.py`  | Evaluation with auxiliary visual inputs |
| `summarize_results.py`     | Aggregate and summarize results         |
| `eval_config.example.json` | Example configuration                   |
| `requirements.txt`         | Python dependencies                     |

---

# 🎯 Supported Tasks

The evaluator dispatches according to:

```python
dataset["type"]
```

Supported task types:

| Type           | Description               |
| -------------- | ------------------------- |
| `g2s`          | Ground-to-Satellite VQA   |
| `s2g`          | Satellite-to-Ground VQA   |
| `s2s`          | Satellite-view VQA        |
| `ge_view`      | Cross-view matching       |
| `gs_grounding` | Cross-view grounding      |
| `mcq_vqa`      | Generic MCQ VQA           |
| `bbox_5level`  | Legacy bbox localization  |
| `arrow_5level` | Legacy arrow localization |
| `arrow_mcq`    | Legacy arrow MCQ          |

---

# 📝 VQA Evaluation

Supported formats:

```text
g2s
s2g
s2s
ge_view
```

Each JSONL sample should contain:

* Question
* Answer
* Options
* Required image fields

Category information is automatically stored in prediction files:

```json
{
  "category": "...",
  "category_l1": "...",
  "category_l2": "..."
}
```

This enables reliable downstream category-level analysis.

---

# 🎯 Grounding Evaluation

Recommended public format:

```text
Image 1 : Source Image
Image 2 : Target Image
Output  : Bounding Box in Target Image
```

Supported fields:

```text
source_image
target_image
source_bbox
target_bbox
answer
answer_bbox
gt_bbox
region_hint
```

> [!NOTE]
>
> Separate source and target images are recommended.
>
> Combined-image formats remain supported for backward compatibility.

---

# 🖼️ Two-Image Evaluation

For experiments involving auxiliary visual information such as depth maps or generated views:

Supported auxiliary inputs:

| Type         |
| ------------ |
| `depth`      |
| `zimage`     |
| `nanobanana` |

Example:

```bash
python eval_double_category.py \
  --base-dir . \
  --extra-kind nanobanana \
  --local-model-path /path/to/Qwen3-VL \
  --g2s-path fov/g2s/Ground2Sat_VQA_test.jsonl \
  --s2g-path fov/s2g/Sat2Ground_VQA_test.jsonl
```

Generated outputs include:

* ✅ Prediction files
* ✅ Missing-pair diagnostics
* ✅ Category tables
* ✅ Summary reports

---

# 📊 Result Summarization

After evaluation:

```bash
python summarize_results.py --root outputs
```

Category lookup priority:

```text
1. category_l1 / category_l2
2. category
3. dataset qid mapping
4. option-signature fallback
```

This design keeps public prediction files fully self-contained.

---

# 📁 Output Structure

```text
outputs/
└── model_name/
    ├── dataset_name/
    │   ├── predictions.jsonl
    │   └── metrics.json
    └── summary.json
```

Additional files generated by `eval_double_category.py`:

* category tables
* split files
* summary reports

---

# 🔧 Before Running

Please verify:

* [ ] Dataset downloaded from Hugging Face
* [ ] `fov/` exists under `evaluate/`
* [ ] `cvusa/` exists under `evaluate/`
* [ ] API credentials configured correctly
* [ ] Local model path configured correctly (if applicable)

---

# 📌 Notes

* This repository contains evaluation scripts only.
* Dataset files must be downloaded separately.
* Private APIs and internal infrastructure have been removed.
* Both local and API-based evaluation are supported.
* The toolkit is designed to be easily extended to additional multimodal models.

---

# 🙏 Citation

If you find CVSBench useful in your research, please cite our paper:

```bibtex
@article{cvsbench2026,
  title={CVSBench: A Comprehensive Benchmark for Cross-View Spatial Reasoning and Dreaming},
  author={...},
  journal={ECCV},
  year={2026}
}
```

---

# ⭐ Acknowledgement

We thank the open-source vision-language community for making large-scale multimodal evaluation possible.

If this repository helps your research, please consider giving it a ⭐ Star.
