import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ============================================================
# Default paths
# ============================================================

DEFAULT_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "")

DEFAULT_BASE_DIR = "."

DEFAULT_G2S_PATH = "fov/g2s/Ground2Sat_VQA_test.jsonl"
DEFAULT_S2G_PATH = "fov/s2g/Sat2Ground_VQA_test.jsonl"

DEFAULT_STREET_ROOT = "fov/data/street"
DEFAULT_SATELLITE_ARROW_ROOT = "fov/data/satellite_arrow"

# extra-kind=depth
DEFAULT_DEPTH_STREET_ROOT = ""
DEFAULT_DEPTH_SATELLITE_ROOT = ""

# extra-kind=zimage
DEFAULT_ZIMAGE_ROOT = ""

# extra-kind=nanobanana
DEFAULT_NANOBANANA_ROOT = "fov/nanobanana"

DEFAULT_OUT_ROOT = "outputs"


G2S_CATS = [
    "facade",
    "roof",
    "symmetry",
]

S2G_CATS = [
    "facade",
    "ground_material",
    "height_comparison",
    "occlusion_binary",
    "relative_position",
    "vegetation_sector",
]


# ============================================================
# Local model cache
# ============================================================

_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None


# ============================================================
# Basic utils
# ============================================================

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def numeric_sort_key(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", name)
    if nums:
        return [int(x) for x in nums]
    return [name]


def list_images_in_dir(
    dir_path: str,
    max_images: Optional[int] = None,
    skip_zero: bool = False,
) -> List[str]:
    if not dir_path or not os.path.exists(dir_path):
        return []

    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(dir_path, ext)))

    imgs = sorted(imgs, key=numeric_sort_key)

    if skip_zero:
        keep = []
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in {"0", "view_0", "view-0"}:
                continue
            keep.append(p)
        imgs = keep

    if max_images is not None:
        imgs = imgs[:max_images]

    return imgs


def first_existing(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def load_completed_ids(preds_path: str):
    completed = set()
    existing = []

    if not os.path.exists(preds_path):
        return completed, existing

    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("qid") or obj.get("id")
            if qid:
                completed.add(qid)
            existing.append(obj)

    return completed, existing


# ============================================================
# Answer / category
# ============================================================

def extract_choice(text: Any) -> Optional[str]:
    if text is None:
        return None

    t = str(text).strip().upper()
    if not t:
        return None

    m = re.search(r"(?:ANSWER|ANS|OPTION|CHOICE|OUTPUT)\s*[:：]?\s*([ABCD])\b", t)
    if m:
        return m.group(1)

    m = re.fullmatch(r"\s*\(?([ABCD])\)?[\.\)]?\s*", t)
    if m:
        return m.group(1)

    m = re.search(r"(?<![A-Z0-9])([ABCD])(?![A-Z0-9])", t)
    if m:
        return m.group(1)

    return None


def clean_choice(x: Any, raw: Any = None) -> str:
    c = extract_choice(x)
    if c:
        return c
    c = extract_choice(raw)
    return c or ""


def normalize_category(cat: Any) -> str:
    if cat is None:
        return "UNKNOWN"

    c = str(cat).strip()

    aliases = {
        "ground_mat": "ground_material",
        "ground-material": "ground_material",
        "height_comp": "height_comparison",
        "height-comparison": "height_comparison",
        "occulusion_binary": "occlusion_binary",
        "occlusion": "occlusion_binary",
        "occlusion-binary": "occlusion_binary",
        "relative-pos": "relative_position",
        "relative_positioning": "relative_position",
        "vegetation": "vegetation_sector",
        "vegetation-sector": "vegetation_sector",
    }

    return aliases.get(c, c)


def get_category(item: dict) -> str:
    return normalize_category(
        item.get("category")
        or item.get("category_l2")
        or item.get("category_l1")
        or "UNKNOWN"
    )


def question_to_prompt(question: str, options: Any, task_hint: str) -> str:
    lines = []

    if task_hint:
        lines.append(task_hint)

    lines.append(question or "")

    if options:
        lines.append("Options:")

        if isinstance(options, list):
            for opt in options:
                if isinstance(opt, dict) and len(opt) == 1:
                    k = list(opt.keys())[0]
                    lines.append(f"{k}. {opt[k]}")
                else:
                    lines.append(str(opt))

        elif isinstance(options, dict):
            for k in sorted(options.keys()):
                lines.append(f"{k}. {options[k]}")

        else:
            lines.append(str(options))

    lines.append("Answer with a single letter A/B/C/D.")
    return "\n".join(lines).strip()


# ============================================================
# Image helpers
# ============================================================

def concat_images_horizontally(image_paths: List[str], out_path: str) -> str:
    if not image_paths:
        raise RuntimeError("No images to concatenate.")

    ensure_dir(os.path.dirname(out_path))

    if os.path.exists(out_path):
        return out_path

    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    min_h = min(im.height for im in imgs)

    resized = []
    for im in imgs:
        if im.height != min_h:
            new_w = int(im.width * min_h / im.height)
            im = im.resize((new_w, min_h))
        resized.append(im)

    total_w = sum(im.width for im in resized)
    canvas = Image.new("RGB", (total_w, min_h))

    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width

    canvas.save(out_path)
    return out_path


def get_sample_and_view(item: dict, row_idx: int):
    sample_id = str(
        item.get("sample_id")
        or item.get("img_id")
        or item.get("sample")
        or item.get("id")
        or row_idx
    ).strip()

    view_id = item.get("view_id", item.get("view", None))
    return sample_id, view_id


def original_images_for_task(
    task_type: str,
    sample_id: str,
    view_id,
    street_root: str,
    satellite_arrow_root: str,
    cache_root: str,
    max_sym_views: int,
) -> List[str]:
    """
    Original images:
    g2s normal: fov/data/street/<sample_id>/<view_id>.jpg
    g2s view_id=0: concatenate street views 1/2/3/4 and skip 0.jpg
    s2g: fov/data/satellite_arrow/<sample_id>/view_<view_id>_annotated.jpg
    """
    exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    if task_type == "g2s":
        if view_id is not None and str(view_id).isdigit() and int(view_id) == 0:
            rgb_dir = os.path.join(street_root, str(sample_id))
            imgs = list_images_in_dir(
                rgb_dir,
                max_images=max_sym_views,
                skip_zero=True,
            )
            if not imgs:
                return []

            out_path = os.path.join(
                cache_root,
                "original_g2s_view0",
                f"{sample_id}_rgb_stitched_max{max_sym_views}.jpg",
            )
            return [concat_images_horizontally(imgs, out_path)]

        if view_id is None:
            return []

        view_str = str(view_id)
        candidates = []
        for e in exts:
            candidates.extend([
                os.path.join(street_root, str(sample_id), f"{view_str}{e}"),
                os.path.join(street_root, str(sample_id), f"view_{view_str}{e}"),
            ])
        p = first_existing(candidates)
        return [p] if p else []

    # s2g
    if view_id is None:
        return []

    view_str = str(view_id)
    candidates = []
    for e in exts:
        candidates.extend([
            os.path.join(satellite_arrow_root, str(sample_id), f"view_{view_str}_annotated{e}"),
            os.path.join(satellite_arrow_root, str(sample_id), f"view_{view_str}{e}"),
            os.path.join(satellite_arrow_root, str(sample_id), f"{view_str}{e}"),
        ])

    p = first_existing(candidates)
    return [p] if p else []


# ============================================================
# Extra image path rules
# ============================================================

def depth_street_for_view(depth_street_root: str, sample_id: str, view_id) -> Optional[str]:
    if view_id is None:
        return None

    view_str = str(int(view_id)) if str(view_id).isdigit() else str(view_id)
    exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

    candidates = []
    for e in exts:
        candidates.extend([
            os.path.join(depth_street_root, str(sample_id), f"gen_{view_str}{e}"),
            os.path.join(depth_street_root, str(sample_id), f"{view_str}{e}"),
        ])

    return first_existing(candidates)


def depth_satellite_for_sample(depth_sat_root: str, sample_id: str) -> Optional[str]:
    exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

    candidates = []
    for e in exts:
        candidates.extend([
            os.path.join(depth_sat_root, str(sample_id), f"gen_{sample_id}{e}"),
            os.path.join(depth_sat_root, str(sample_id), f"{sample_id}{e}"),
            os.path.join(depth_sat_root, f"gen_{sample_id}{e}"),
            os.path.join(depth_sat_root, f"{sample_id}{e}"),
        ])

    return first_existing(candidates)


def depth_extra_for_task(
    task_type: str,
    sample_id: str,
    view_id,
    depth_street_root: str,
    depth_sat_root: str,
    street_root: str,
    cache_root: str,
    max_sym_views: int,
) -> Optional[str]:
    """
    Depth extra:
    g2s normal: depth_street/<sample_id>/gen_<view_id>.png
    g2s view_id=0: concatenate depth_street/<sample_id>/gen_1/2/3/4.png
    s2g: depth_satellite/<sample_id>/gen_<sample_id>.png
    """
    if task_type == "g2s":
        if view_id is not None and str(view_id).isdigit() and int(view_id) == 0:
            rgb_dir = os.path.join(street_root, str(sample_id))
            rgb_imgs = list_images_in_dir(
                rgb_dir,
                max_images=max_sym_views,
                skip_zero=True,
            )
            if not rgb_imgs:
                return None

            depth_imgs = []
            for rgb in rgb_imgs:
                stem = os.path.splitext(os.path.basename(rgb))[0]
                dep = depth_street_for_view(depth_street_root, sample_id, stem)
                if not dep:
                    return None
                depth_imgs.append(dep)

            out_path = os.path.join(
                cache_root,
                "extra_depth_g2s_view0",
                f"{sample_id}_depth_stitched_max{max_sym_views}.jpg",
            )
            return concat_images_horizontally(depth_imgs, out_path)

        return depth_street_for_view(depth_street_root, sample_id, view_id)

    return depth_satellite_for_sample(depth_sat_root, sample_id)


def zimage_extra_for_task(
    task_type: str,
    sample_id: str,
    view_id,
    zimage_root: str,
) -> Optional[str]:
    """
    Z-Image:
    g2s normal: single_gen/street/<sample_id>/gen_<view_id>.jpg
    g2s view_id=0: multiview_gen/<sample_id>.jpg
    s2g: single_gen/satellite/<sample_id>/gen_<sample_id>.jpg
    """
    exts = [".jpg", ".png", ".jpeg", ".webp"]

    if task_type == "g2s":
        if view_id is not None and str(view_id).isdigit() and int(view_id) == 0:
            candidates = []
            for e in exts:
                candidates.append(os.path.join(zimage_root, "multiview_gen", f"{sample_id}{e}"))
            return first_existing(candidates)

        if view_id is None:
            return None

        view_str = str(int(view_id)) if str(view_id).isdigit() else str(view_id)
        candidates = []
        for e in exts:
            candidates.append(
                os.path.join(zimage_root, "single_gen", "street", str(sample_id), f"gen_{view_str}{e}")
            )
        return first_existing(candidates)

    candidates = []
    for e in exts:
        candidates.append(
            os.path.join(zimage_root, "single_gen", "satellite", str(sample_id), f"gen_{sample_id}{e}")
        )
    return first_existing(candidates)


def nanobanana_extra_for_task(
    task_type: str,
    sample_id: str,
    view_id,
    nanobanana_root: str,
) -> Optional[str]:
    """
    nanobanana:
    g2s normal: train_street/<sample_id>/gen_<view_id>.jpg
    g2s view_id=0: results_multiview_final/<sample_id>.jpg
    s2g: gallery_satellite/<sample_id>/gen_<sample_id>.jpg
    """
    exts = [".jpg", ".png", ".jpeg", ".webp"]

    if task_type == "g2s":
        if view_id is not None and str(view_id).isdigit() and int(view_id) == 0:
            candidates = []
            for e in exts:
                candidates.append(os.path.join(nanobanana_root, "results_multiview_final", f"{sample_id}{e}"))
            return first_existing(candidates)

        if view_id is None:
            return None

        view_str = str(int(view_id)) if str(view_id).isdigit() else str(view_id)
        candidates = []
        for e in exts:
            candidates.append(
                os.path.join(nanobanana_root, "train_street", str(sample_id), f"gen_{view_str}{e}")
            )
        return first_existing(candidates)

    candidates = []
    for e in exts:
        candidates.append(
            os.path.join(nanobanana_root, "gallery_satellite", str(sample_id), f"gen_{sample_id}{e}")
        )

    p = first_existing(candidates)
    if p:
        return p

    gallery = os.path.join(nanobanana_root, "gallery_satellite", str(sample_id))
    imgs = list_images_in_dir(gallery, max_images=1, skip_zero=False)
    return imgs[0] if imgs else None


def extra_image_for_task(
    extra_kind: str,
    task_type: str,
    sample_id: str,
    view_id,
    args,
) -> Optional[str]:
    if extra_kind == "depth":
        return depth_extra_for_task(
            task_type=task_type,
            sample_id=sample_id,
            view_id=view_id,
            depth_street_root=args.depth_street_root,
            depth_sat_root=args.depth_satellite_root,
            street_root=args.street_root,
            cache_root=args.cache_root_abs,
            max_sym_views=args.max_sym_views,
        )

    if extra_kind == "zimage":
        return zimage_extra_for_task(
            task_type=task_type,
            sample_id=sample_id,
            view_id=view_id,
            zimage_root=args.zimage_root,
        )

    if extra_kind == "nanobanana":
        return nanobanana_extra_for_task(
            task_type=task_type,
            sample_id=sample_id,
            view_id=view_id,
            nanobanana_root=args.nanobanana_root,
        )

    raise RuntimeError(f"Unknown extra_kind: {extra_kind}")


# ============================================================
# Local Qwen3-VL inference
# ============================================================

def load_local_qwen(model_path: str):
    global _LOCAL_MODEL, _LOCAL_PROCESSOR

    if _LOCAL_MODEL is not None and _LOCAL_PROCESSOR is not None:
        return _LOCAL_MODEL, _LOCAL_PROCESSOR

    print(f"Loading local Qwen3-VL model from: {model_path}")

    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    try:
        _LOCAL_MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            **kwargs,
        )
        print("Using flash_attention_2")
    except Exception as e:
        print(f"flash_attention_2 failed, fallback to default attention: {e}")
        _LOCAL_MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            **kwargs,
        )

    _LOCAL_PROCESSOR = AutoProcessor.from_pretrained(model_path)
    _LOCAL_MODEL.eval()

    return _LOCAL_MODEL, _LOCAL_PROCESSOR


@torch.inference_mode()
def run_mcq_local_qwen(
    model_path: str,
    prompt: str,
    image_paths: List[str],
    max_new_tokens: int = 32,
):
    model, processor = load_local_qwen(model_path)

    for p in image_paths:
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"Image not found for Qwen input: {p}")

    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": p} for p in image_paths]
            + [{"type": "text", "text": prompt}]
        ),
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    pred = clean_choice(raw)
    return raw, pred


# ============================================================
# Metrics
# ============================================================

def empty_stat():
    return {"correct": 0, "total": 0, "acc": 0.0}


def update_stat(stats: Dict[str, Dict[str, int]], key: str, ok: bool):
    if key not in stats:
        stats[key] = {"correct": 0, "total": 0}
    stats[key]["total"] += 1
    stats[key]["correct"] += 1 if ok else 0


def finalize_stat(stats: Dict[str, Dict[str, int]]) -> Dict[str, dict]:
    out = {}
    for k, v in stats.items():
        total = v["total"]
        correct = v["correct"]
        out[k] = {
            "correct": correct,
            "total": total,
            "acc": correct / total if total else 0.0,
        }
    return out


def table_row_from_metrics(name: str, metrics: dict) -> dict:
    row = {"method": name}

    g2s_cat = metrics.get("category_accuracy", {}).get("g2s", {})
    s2g_cat = metrics.get("category_accuracy", {}).get("s2g", {})

    for c in G2S_CATS:
        v = g2s_cat.get(c)
        row[f"g2s/{c}"] = f'{v["acc"]:.3f}' if v else ""

    for c in S2G_CATS:
        v = s2g_cat.get(c)
        row[f"s2g/{c}"] = f'{v["acc"]:.3f}' if v else ""

    return row


def save_category_table(out_dir: str, model_name: str, metrics: dict):
    row = table_row_from_metrics(model_name, metrics)

    headers = (
        ["method"]
        + [f"g2s/{c}" for c in G2S_CATS]
        + [f"s2g/{c}" for c in S2G_CATS]
    )

    md_path = os.path.join(out_dir, "category_table.md")
    csv_path = os.path.join(out_dir, "category_table.csv")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow({h: row.get(h, "") for h in headers})

    print("\nCategory table:")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    print("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    print("\nSaved category table:")
    print(" ", md_path)
    print(" ", csv_path)


# ============================================================
# Evaluation
# ============================================================

def eval_split(
    args,
    task_type: str,
    data_path: str,
    out_dir: str,
):
    items = load_jsonl(data_path)
    if args.max_samples is not None:
        items = items[:args.max_samples]

    ensure_dir(out_dir)

    preds_path = os.path.join(out_dir, "predictions_double.jsonl")
    split_path = os.path.join(out_dir, f"fov_{args.extra_kind}_{task_type}-double.jsonl")
    metrics_path = os.path.join(out_dir, "metrics.json")

    completed, existing = load_completed_ids(preds_path)

    total = 0
    correct = 0
    category_stats = {}

    seen = set()
    for row in existing:
        qid = row.get("qid") or row.get("id")
        if not qid or qid in seen:
            continue
        seen.add(qid)

        pred = clean_choice(row.get("pred"), row.get("raw"))
        ans = clean_choice(row.get("answer"))
        ok = bool(pred and ans and pred == ans)

        cat = normalize_category(row.get("category"))
        total += 1
        correct += 1 if ok else 0
        update_stat(category_stats, cat, ok)

    split_records = []
    pending = []
    missing = []

    for idx, item in enumerate(items):
        sample_id, view_id = get_sample_and_view(item, idx)
        cat = get_category(item)

        original_imgs = original_images_for_task(
            task_type=task_type,
            sample_id=sample_id,
            view_id=view_id,
            street_root=args.street_root,
            satellite_arrow_root=args.satellite_arrow_root,
            cache_root=args.cache_root_abs,
            max_sym_views=args.max_sym_views,
        )

        extra_img = extra_image_for_task(
            extra_kind=args.extra_kind,
            task_type=task_type,
            sample_id=sample_id,
            view_id=view_id,
            args=args,
        )

        if not original_imgs or not extra_img or not os.path.exists(extra_img):
            missing.append({
                "idx": idx,
                "sample_id": sample_id,
                "view_id": view_id,
                "task_type": task_type,
                "category": cat,
                "original_imgs": original_imgs,
                "extra_img": extra_img,
            })
            continue

        qid = f"{sample_id}.{view_id}.q{idx}.{args.extra_kind}_double"
        image_paths = original_imgs + [extra_img]

        rec = {
            "qid": qid,
            "id": qid,
            "task_type": task_type,
            "sample_id": sample_id,
            "view_id": view_id,
            "category": cat,
            "question": item.get("question", ""),
            "options": item.get("options"),
            "answer": item.get("answer"),
            "image_paths": image_paths,
            "mode": "double",
            "extra_kind": args.extra_kind,
        }
        split_records.append(rec)

        if qid not in completed:
            pending.append((idx, item, rec))

    with open(split_path, "w", encoding="utf-8") as f:
        for rec in split_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if missing:
        missing_path = os.path.join(out_dir, "missing_pairs.json")
        dump_json(missing_path, missing)
        print(f"  ⚠ Missing pairs: {len(missing)}. Saved to {missing_path}")
        print("  First missing:", missing[0])

    pbar = tqdm(total=len(pending), desc=f"Processing {task_type}")

    def process_one(idx, item, rec):
        question = item.get("question", "")
        options = item.get("options")
        answer = clean_choice(item.get("answer"))
        category = normalize_category(rec.get("category"))

        if args.extra_kind == "depth":
            task_hint = (
                "Two images are provided: the original visual input and an aligned depth map. "
                "Use both images jointly to answer."
            )
        elif args.extra_kind == "zimage":
            task_hint = (
                "Two images are provided: the original visual input and an additional generated isometric image. "
                "Use both images jointly to answer."
            )
        else:
            task_hint = (
                "Two images are provided: the original visual input and an additional image generated by nanobanana. "
                "Use both images jointly to answer."
            )

        prompt = question_to_prompt(question, options, task_hint)

        raw, pred = run_mcq_local_qwen(
            model_path=args.local_model_path,
            prompt=prompt,
            image_paths=rec["image_paths"],
            max_new_tokens=args.max_new_tokens,
        )

        pred = clean_choice(pred, raw)
        ok = bool(pred and answer and pred == answer)

        out = dict(rec)
        out.update({
            "answer": answer,
            "pred": pred,
            "correct": ok,
            "raw": raw,
            "category": category,
        })

        return out, ok, category

    with open(preds_path, "a", encoding="utf-8") as f:
        for task in pending:
            qid = task[2]["qid"]
            try:
                out, ok, category = process_one(*task)
            except Exception as e:
                print(f"\n  ⚠ sample {qid} failed: {e}")
                pbar.update(1)
                continue

            total += 1
            correct += 1 if ok else 0
            update_stat(category_stats, category, ok)

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

            pbar.update(1)

    pbar.close()

    metrics = {
        "task_type": task_type,
        "extra_kind": args.extra_kind,
        "accuracy_double": correct / total if total else 0.0,
        "total_double": total,
        "correct_double": correct,
        "missing_pairs": len(missing),
        "predictions_file": preds_path,
        "split_file": split_path,
        "category_accuracy": finalize_stat(category_stats),
    }

    dump_json(metrics_path, metrics)

    print(f"\n[{task_type}]")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {metrics['accuracy_double']:.4f}")
    print("Category accuracy:")
    for cat, s in metrics["category_accuracy"].items():
        print(f"  {cat}: {s['correct']}/{s['total']} = {s['acc']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--g2s-path", default=DEFAULT_G2S_PATH)
    parser.add_argument("--s2g-path", default=DEFAULT_S2G_PATH)

    parser.add_argument("--street-root", default=DEFAULT_STREET_ROOT)
    parser.add_argument("--satellite-arrow-root", default=DEFAULT_SATELLITE_ARROW_ROOT)

    parser.add_argument("--extra-kind", choices=["depth", "zimage", "nanobanana"], required=True)

    parser.add_argument("--depth-street-root", default=DEFAULT_DEPTH_STREET_ROOT)
    parser.add_argument("--depth-satellite-root", default=DEFAULT_DEPTH_SATELLITE_ROOT)

    parser.add_argument("--zimage-root", default=DEFAULT_ZIMAGE_ROOT)
    parser.add_argument("--nanobanana-root", default=DEFAULT_NANOBANANA_ROOT)

    parser.add_argument("--model-name", default=None)
    parser.add_argument("--local-model-path", default=DEFAULT_MODEL_PATH)

    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cache-root", default=None)

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-sym-views", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--only", choices=["both", "g2s", "s2g"], default="both")

    args = parser.parse_args()

    args.base_dir = os.path.abspath(args.base_dir)

    def abs_or_empty(path: str) -> str:
        if not path:
            return ""
        return path if os.path.isabs(path) else os.path.join(args.base_dir, path)

    args.street_root = abs_or_empty(args.street_root)
    args.satellite_arrow_root = abs_or_empty(args.satellite_arrow_root)
    args.depth_street_root = abs_or_empty(args.depth_street_root)
    args.depth_satellite_root = abs_or_empty(args.depth_satellite_root)
    args.zimage_root = abs_or_empty(args.zimage_root)
    args.nanobanana_root = abs_or_empty(args.nanobanana_root)
    args.out_root = abs_or_empty(args.out_root)

    if args.max_sym_views < 1:
        raise ValueError("--max-sym-views must be >= 1")
    if args.max_sym_views > 4:
        print(f"⚠ --max-sym-views={args.max_sym_views} > 4, forcing to 4.")
        args.max_sym_views = 4

    if args.model_name is None:
        if args.extra_kind == "depth":
            args.model_name = "depth_CategoryEval"
        elif args.extra_kind == "zimage":
            args.model_name = "zimage_CategoryEval"
        else:
            args.model_name = "nanobanana_CategoryEval"

    if args.extra_kind == "depth":
        if not args.depth_street_root or not args.depth_satellite_root:
            raise ValueError("For --extra-kind depth, please provide --depth-street-root and --depth-satellite-root.")
    if args.extra_kind == "zimage":
        if not args.zimage_root:
            raise ValueError("For --extra-kind zimage, please provide --zimage-root.")
    if args.extra_kind == "nanobanana":
        if not args.nanobanana_root:
            raise ValueError("For --extra-kind nanobanana, please provide --nanobanana-root.")

    if args.cache_root is None:
        args.cache_root = f"outputs/_{args.model_name}_cache"

    args.cache_root_abs = (
        args.cache_root
        if os.path.isabs(args.cache_root)
        else os.path.join(args.base_dir, args.cache_root)
    )

    model_out = os.path.join(args.out_root, args.model_name)
    ensure_dir(model_out)

    print("=" * 80)
    print("Model:", args.model_name)
    print("Local model:", args.local_model_path)
    print("Extra kind:", args.extra_kind)
    print("Output:", model_out)
    print("=" * 80)

    summary = {
        "model": args.model_name,
        "local_model_path": args.local_model_path,
        "extra_kind": args.extra_kind,
        "datasets": {},
        "category_accuracy": {
            "g2s": {},
            "s2g": {},
        },
    }

    if args.only in ("both", "g2s"):
        g2s_metrics = eval_split(
            args=args,
            task_type="g2s",
            data_path=os.path.join(args.base_dir, args.g2s_path),
            out_dir=os.path.join(model_out, f"fov_{args.extra_kind}_g2s"),
        )
        summary["datasets"][f"fov_{args.extra_kind}_g2s"] = g2s_metrics
        summary["category_accuracy"]["g2s"] = g2s_metrics["category_accuracy"]

    if args.only in ("both", "s2g"):
        s2g_metrics = eval_split(
            args=args,
            task_type="s2g",
            data_path=os.path.join(args.base_dir, args.s2g_path),
            out_dir=os.path.join(model_out, f"fov_{args.extra_kind}_s2g"),
        )
        summary["datasets"][f"fov_{args.extra_kind}_s2g"] = s2g_metrics
        summary["category_accuracy"]["s2g"] = s2g_metrics["category_accuracy"]

    summary_path = os.path.join(model_out, "summary.json")
    dump_json(summary_path, summary)

    save_category_table(model_out, args.model_name, summary)

    print("\nDone.")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
