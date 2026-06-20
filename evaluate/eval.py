"""
Generic evaluation entrypoint for CVSBench-style datasets.

Supports:
- MCQ / VQA evaluation
- bbox / grounding evaluation
- OpenAI-compatible APIs
- local Hugging Face transformers models

This file is adapted from internal research scripts and cleaned for public use.
"""
import argparse
import base64
import glob
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv
try:
    from PIL import Image
except Exception:
    Image = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# load .env for API keys
load_dotenv()


# =========================
# Default model config
# =========================
MODEL_CONF = {
    "name": "model",
    "backend": os.environ.get("EVAL_BACKEND", "openai_compat"),
    "model": os.environ.get("EVAL_MODEL", "YOUR_MODEL_NAME"),
    "base_url": os.environ.get("OPENAI_BASE_URL", os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")),
    "api_key": os.environ.get("OPENAI_API_KEY", os.environ.get("LLM_API_KEY", os.environ.get("API_KEY"))),
    "concurrency": 8,
    "timeout": 60,
    "max_retries": 3,
    "bbox_format": os.environ.get("BBOX_FORMAT", "pixel"),
}

GLOBAL_BASE_DIR = None
LOCAL_MODE = os.environ.get("LOCAL_TRANSFORMERS", "").lower() in ("1", "true", "yes")
_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None
_LOCAL_LOCK = None


# =========================
# Utils
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def dump_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_category(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def get_concurrency(model_conf, default=8):
    env_val = os.environ.get("EVAL_CONCURRENCY")
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass
    return max(1, int(model_conf.get("concurrency", default)))


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def resolve_image_path(base_dir: str, path: str) -> Optional[str]:
    """
    Compatibility-oriented path resolution:
      - first try the original path
      - then try the path relative to base_dir
      - keep lightweight fov/cvusa path normalization
    """
    if not path:
        return None
    # fov train/test path normalization
    p = path
    p = p.replace("fov/test_satellite", "fov/data/satellite")
    p = p.replace("fov/train_satellite", "fov/data/satellite")
    p = p.replace("fov/test_street", "fov/data/street")
    p = p.replace("fov/train_street", "fov/data/street")
    p = p.replace("fov\\test_satellite", "fov\\data\\satellite")
    p = p.replace("fov\\train_satellite", "fov\\data\\satellite")
    p = p.replace("fov\\test_street", "fov\\data\\street")
    p = p.replace("fov\\train_street", "fov\\data\\street")
    if os.path.exists(p):
        return p
    alt = os.path.normpath(os.path.join(base_dir, p))
    if os.path.exists(alt):
        return alt
    return p


def list_images_in_dir(dir_path: str) -> List[str]:
    if not dir_path or not os.path.exists(dir_path):
        return []
    imgs = []
    imgs += sorted(glob.glob(os.path.join(dir_path, "*.jpg")))
    imgs += sorted(glob.glob(os.path.join(dir_path, "*.jpeg")))
    imgs += sorted(glob.glob(os.path.join(dir_path, "*.png")))
    return imgs


def read_image_as_data_url(path: str, base_dir: str) -> str:
    """
    Encode an image as a data URL, with GLOBAL_BASE_DIR as a fallback root.
    """
    p = resolve_image_path(base_dir, path)
    if not p or not os.path.exists(p):
        if GLOBAL_BASE_DIR:
            alt = os.path.normpath(os.path.join(GLOBAL_BASE_DIR, path))
            if os.path.exists(alt):
                p = alt
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"Image not found: {path} (resolved={p})")

    with open(p, "rb") as f:
        data = f.read()

    ext = os.path.splitext(p)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _init_local_model():
    global _LOCAL_MODEL, _LOCAL_PROCESSOR, _LOCAL_LOCK
    if _LOCAL_LOCK is None:
        import threading
        _LOCAL_LOCK = threading.Lock()
    with _LOCAL_LOCK:
        if _LOCAL_MODEL is not None and _LOCAL_PROCESSOR is not None:
            return _LOCAL_MODEL, _LOCAL_PROCESSOR

        model_path = os.environ.get("LOCAL_MODEL_PATH", "")
        if not model_path:
            raise RuntimeError("LOCAL_MODEL_PATH is not set for local transformers mode.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LOCAL_MODEL_PATH not found: {model_path}")

        # Lazy imports avoid unnecessary dependencies when local mode is disabled.
        import torch
        family = os.environ.get("LOCAL_MODEL_FAMILY", "").strip().lower()

        if family in ("gemma", "gemma3", "gemma-3"):
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration

            _LOCAL_PROCESSOR = AutoProcessor.from_pretrained(model_path)
            _LOCAL_MODEL = Gemma3ForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
            ).eval()
            _LOCAL_MODEL._local_family = "gemma3"

        elif family in ("qwen3vl", "qwen3-vl", "qwen3_vl", "qwen3vl4b", "qwen3-vl-4b"):
            # Qwen3-VL should use AutoModelForImageTextToText, not AutoModelForCausalLM.
            from transformers import AutoProcessor, AutoModelForImageTextToText

            print(f"[LOCAL] Loading Qwen3-VL from: {model_path}")
            _LOCAL_PROCESSOR = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            _LOCAL_MODEL = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            _LOCAL_MODEL._local_family = "qwen3_vl"

        else:
            raise RuntimeError(
                "Unsupported LOCAL_MODEL_FAMILY for public local mode. "
                "Currently supported values are: qwen3vl, gemma3. "
                "For other models, either add a local adapter in eval.py or use an OpenAI-compatible API endpoint."
            )
        return _LOCAL_MODEL, _LOCAL_PROCESSOR

def _local_generate(prompt: str, image_paths: List[str]) -> str:
    model, processor = _init_local_model()
    family = getattr(model, "_local_family", "")

    if family == "gemma3":
        from PIL import Image
        import torch

        images = []
        for p in image_paths:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": im} for im in images]
                + [{"type": "text", "text": prompt}],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = model.device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        inputs = inputs.to(device, dtype=dtype)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generation = generation[0][input_len:]
        return processor.decode(generation, skip_special_tokens=True)

    if family == "qwen3_vl":
        from PIL import Image
        import torch

        real_paths = []
        for p in image_paths:
            if not p:
                continue
            rp = resolve_image_path(GLOBAL_BASE_DIR or ".", p)
            if rp and os.path.exists(rp):
                real_paths.append(rp)
            elif os.path.exists(p):
                real_paths.append(p)

        if not real_paths:
            raise RuntimeError(f"No valid images for local Qwen3-VL. image_paths={image_paths}")

        images = []
        for p in real_paths:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": im} for im in images]
                + [{"type": "text", "text": prompt}],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        out_trim = out[:, inputs.input_ids.shape[1]:]
        return processor.batch_decode(
            out_trim,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    raise RuntimeError(
        "Unsupported local model family. Use LOCAL_MODEL_FAMILY=qwen3vl or gemma3, "
        "or route the model through an OpenAI-compatible API."
    )

def read_image_as_data_url_with_scale(path: str, base_dir: str):
    """
    Return (data_url, scale, resized_size, original_size).
    scale = resized_w / original_w (same for h).
    """
    p = resolve_image_path(base_dir, path)
    if not p or not os.path.exists(p):
        if GLOBAL_BASE_DIR:
            alt = os.path.normpath(os.path.join(GLOBAL_BASE_DIR, path))
            if os.path.exists(alt):
                p = alt
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"Image not found: {path} (resolved={p})")

    max_side = int(os.environ.get("IMG_MAX_SIDE", "512"))
    if max_side > 0 and Image is not None:
        try:
            from io import BytesIO
            with Image.open(p) as im:
                im = im.convert("RGB")
                ow, oh = im.size
                scale = max_side / float(max(ow, oh))
                if scale < 1.0:
                    new_size = (max(1, int(ow * scale)), max(1, int(oh * scale)))
                    im = im.resize(new_size, Image.BILINEAR)
                else:
                    new_size = (ow, oh)
                    scale = 1.0
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=80)
                data = buf.getvalue()
                return (
                    f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}",
                    scale,
                    new_size,
                    (ow, oh),
                )
        except Exception:
            pass

    with open(p, "rb") as f:
        data = f.read()
    ext = os.path.splitext(p)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", 1.0, (None, None), (None, None)


def build_message_with_scales(text: str, image_paths: List[str], base_dir: str):
    content = []
    scales = []
    sizes = []
    added = 0
    for p in image_paths:
        if not p:
            continue
        url, scale, resized_size, _ = read_image_as_data_url_with_scale(p, base_dir=base_dir)
        content.append({"type": "image_url", "image_url": {"url": url}})
        scales.append(scale)
        sizes.append(resized_size)
        added += 1
    if added == 0:
        raise RuntimeError(f"No valid images attached. image_paths={image_paths}")
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}], scales, sizes


def build_message(text: str, image_paths: List[str], base_dir: str):
    # Require at least one valid image attachment.
    content = []
    added = 0
    for p in image_paths:
        if not p:
            continue
        # Keep failures explicit when an image cannot be resolved.
        url = read_image_as_data_url(p, base_dir=base_dir)
        content.append({"type": "image_url", "image_url": {"url": url}})
        added += 1

    if added == 0:
        raise RuntimeError(f"No valid images attached. image_paths={image_paths}")

    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


# =========================
# Robust parsing
# =========================
def extract_choice(text: str) -> Optional[str]:
    """
    Avoid incorrectly taking the "A" in "Answer" as the final choice.
    """
    if not text:
        return None
    t = text.strip().upper()

    m = re.search(r"(?:ANSWER|ANS|OPTION|CHOICE|OUTPUT)\s*[:：]?\s*([ABCD])\b", t)
    if m:
        return m.group(1)

    m = re.search(r"(?<![A-Z0-9])([ABCD])(?![A-Z0-9])", t)
    if m:
        return m.group(1)

    return None


def extract_float_list(text: str, expected_len: Optional[int] = None) -> Optional[List[float]]:
    if not text:
        return None
    candidates = re.findall(r"\[[^\]]+\]", text)
    for c in candidates:
        try:
            arr = json.loads(c)
            if isinstance(arr, list) and (expected_len is None or len(arr) == expected_len):
                return [float(x) for x in arr]
        except Exception:
            pass
    nums = re.findall(r"-?\d+\.\d+|-?\d+", text)
    if expected_len and len(nums) >= expected_len:
        return [float(x) for x in nums[:expected_len]]
    return None


def extract_angle(text: str) -> Optional[float]:
    m = re.search(r"angle\s*[:=]\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    nums = re.findall(r"-?\d+\.\d+|-?\d+", text)
    if nums:
        return float(nums[-1])
    return None


def extract_xy_angle(text: str):
    candidates = re.findall(r"\{[^\}]+\}", text)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                if "xy" in obj and isinstance(obj["xy"], list) and len(obj["xy"]) == 2:
                    angle = obj.get("angle")
                    return [float(obj["xy"][0]), float(obj["xy"][1])], float(angle) if angle is not None else None
        except Exception:
            pass
    xy = extract_float_list(text, expected_len=2)
    angle = extract_angle(text)
    return xy, angle


def question_to_prompt(question: str, options: Any, task_hint: str, force_mcq: bool = False) -> str:
    lines = []
    if task_hint:
        lines.append(task_hint)
    lines.append(question)

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

    if force_mcq:
        lines.append("Answer with a single letter A/B/C/D.")
    return "\n".join(lines).strip()


# =========================
# OpenAI SDK client
# =========================
_CLIENTS: Dict[Tuple[str, str], OpenAI] = {}


def get_client(model_conf) -> OpenAI:
    api_key = (
        model_conf.get("api_key")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or MODEL_CONF.get("api_key")
    )
    base_url_raw = (model_conf.get("base_url") or MODEL_CONF.get("base_url") or "").strip()
    base_url = base_url_raw.rstrip("/") + "/"
    if not api_key:
        raise RuntimeError("Missing api_key (set OPENAI_API_KEY / API_KEY)")
    k = (api_key, base_url)
    if k not in _CLIENTS:
        _CLIENTS[k] = OpenAI(api_key=api_key, base_url=base_url)
    return _CLIENTS[k]


def call_chat(model_conf, messages, max_tokens=64, temperature=0.01) -> str:
    client = get_client(model_conf)
    max_retries = int(model_conf.get("max_retries", 3))
    timeout = int(model_conf.get("timeout", 60))
    last_err = None

    for attempt in range(max_retries):
        try:
            rsp = client.chat.completions.create(
                model=model_conf["model"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return rsp.choices[0].message.content
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # optional: fallback model if wrong
            if ("model_not_found" in msg or "does not exist" in msg) and model_conf.get("model") != MODEL_CONF.get("model"):
                if not model_conf.get("_used_fallback"):
                    fallback = MODEL_CONF.get("model")
                    print(f"  Warning: model not found. Falling back to {fallback} and retrying.")
                    model_conf["model"] = fallback
                    model_conf["_used_fallback"] = True

            safe_err = str(last_err).encode("utf-8", errors="replace").decode("utf-8")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Warning: API call failed ({attempt+1}/{max_retries}): {safe_err} -> retrying in {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError(f"API call failed after {max_retries} retries: {safe_err}")


def run_mcq(model_conf, prompt: str, image_paths: List[str], base_dir: str):
    system_text = (
        "You are a visual question answering assistant. "
        "Look at the images carefully and then choose the single correct option letter. "
        "Answer with ONLY one letter: A, B, C, or D. Do not add any other text."
    )
    if LOCAL_MODE:
        raw = _local_generate(system_text + "\n" + prompt, image_paths)
    else:
        user_msgs = build_message(prompt, image_paths, base_dir=base_dir)
        messages = [{"role": "system", "content": system_text}] + user_msgs
        raw = call_chat(model_conf, messages, max_tokens=32, temperature=0.01)
    pred = extract_choice(raw)
    return raw, pred


def run_bbox(model_conf, prompt: str, image_paths: List[str], base_dir: str):
    # Build message with optional downsampling; use resized size as model-visible size.
    second_scale = 1.0
    second_size = None
    user_msgs, scales, sizes = build_message_with_scales(prompt, image_paths, base_dir=base_dir)
    if scales:
        second_scale = scales[-1]
    if sizes:
        second_size = sizes[-1]
    if second_size and all(second_size):
        w, h = second_size
        fmt = f"Return ONLY [x_min,y_min,x_max,y_max] in PIXEL coords of SECOND image (w={w}, h={h})."
    else:
        fmt = "Return ONLY [x_min,y_min,x_max,y_max] in normalized [0,1] of SECOND image."
    system_text = "Return only one bbox array, no extra text."
    user_msgs, scales, sizes = build_message_with_scales(prompt + "\n" + fmt, image_paths, base_dir=base_dir)
    if scales:
        second_scale = scales[-1]
    if LOCAL_MODE:
        # In local mode, always instruct pixel coords using actual second image size if possible.
        if Image is not None and len(image_paths) >= 2:
            try:
                p2 = resolve_image_path(base_dir, image_paths[1])
                with Image.open(p2) as im:
                    w, h = im.size
                fmt_local = f"Return ONLY [x_min,y_min,x_max,y_max] in PIXEL coords of SECOND image (w={w}, h={h})."
            except Exception:
                fmt_local = fmt
        else:
            fmt_local = fmt
        raw = _local_generate(system_text + "\n" + prompt + "\n" + fmt_local, image_paths)
    else:
        messages = [{"role": "system", "content": system_text}] + user_msgs
        raw = call_chat(model_conf, messages, max_tokens=512, temperature=0.0)
    pred = extract_float_list(raw, expected_len=4)
    if pred is None:
        # retry once with a stricter instruction to avoid partial outputs
        strict_fmt = (
            "Return EXACTLY four numbers as a JSON array like [x_min, y_min, x_max, y_max]. "
            "No markdown, no code fences, no extra text."
        )
        user_msgs, scales, sizes = build_message_with_scales(prompt + "\n" + fmt + "\n" + strict_fmt, image_paths, base_dir=base_dir)
        if scales:
            second_scale = scales[-1]
        if LOCAL_MODE:
            raw = _local_generate(system_text + "\n" + prompt + "\n" + fmt + "\n" + strict_fmt, image_paths)
        else:
            messages = [{"role": "system", "content": system_text}] + user_msgs
            raw = call_chat(model_conf, messages, max_tokens=512, temperature=0.0)
    pred = extract_float_list(raw, expected_len=4)
    # If model still returns normalized coords, scale to pixels.
    if pred:
        # Heuristic: all values <= 1.5 => treat as normalized.
        if max(pred) <= 1.5:
            if Image is not None and len(image_paths) >= 2:
                try:
                    p2 = resolve_image_path(base_dir, image_paths[1])
                    with Image.open(p2) as im:
                        w, h = im.size
                    pred = [pred[0]*w, pred[1]*h, pred[2]*w, pred[3]*h]
                except Exception:
                    pass
        elif second_scale and second_scale != 1.0:
            pred = [v / second_scale for v in pred]
    return raw, pred


def run_point(model_conf, prompt: str, image_paths: List[str], base_dir: str):
    fmt = "Return ONLY a JSON array [x, y] with normalized values in [0,1]."
    system_text = (
        "You are a visual question answering assistant. "
        "Return ONLY the requested JSON array. Do not add any other text."
    )
    if LOCAL_MODE:
        raw = _local_generate(system_text + "\n" + prompt + "\n" + fmt, image_paths)
    else:
        user_msgs = build_message(prompt + "\n" + fmt, image_paths, base_dir=base_dir)
        messages = [{"role": "system", "content": system_text}] + user_msgs
        raw = call_chat(model_conf, messages, max_tokens=96, temperature=0.01)
    pred = extract_float_list(raw, expected_len=2)
    return raw, pred


def run_point_with_angle(model_conf, prompt: str, image_paths: List[str], base_dir: str):
    fmt = "Return ONLY JSON like {\"xy\":[x,y],\"angle\":deg} with normalized xy in [0,1]."
    system_text = (
        "You are a visual question answering assistant. "
        "Return ONLY the requested JSON object {\"xy\":[x,y],\"angle\":deg}. Do not add other text."
    )
    if LOCAL_MODE:
        raw = _local_generate(system_text + "\n" + prompt + "\n" + fmt, image_paths)
    else:
        user_msgs = build_message(prompt + "\n" + fmt, image_paths, base_dir=base_dir)
        messages = [{"role": "system", "content": system_text}] + user_msgs
        raw = call_chat(model_conf, messages, max_tokens=128, temperature=0.01)
    xy, angle = extract_xy_angle(raw)
    return raw, xy, angle


# =========================
# Metrics helpers (old code)
# =========================
def normalize_bbox_order(b):
    x1, y1, x2, y2 = b
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def bbox_iou(a, b):
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = normalize_bbox_order(a)
    bx1, by1, bx2, by2 = normalize_bbox_order(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def clamp_bbox(b):
    if not b or len(b) != 4:
        return b
    return [float(x) for x in normalize_bbox_order(b)]


def center_in_gt(pred, gt):
    if not pred or not gt:
        return False
    px1, py1, px2, py2 = normalize_bbox_order(pred)
    gx1, gy1, gx2, gy2 = normalize_bbox_order(gt)
    cx = (px1 + px2) / 2.0
    cy = (py1 + py2) / 2.0
    return gx1 <= cx <= gx2 and gy1 <= cy <= gy2


def l2_dist(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def angle_diff_deg(a, b):
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def quadrant_from_xy(xy):
    x, y = xy
    if x < 0.5 and y < 0.5:
        return "A"
    if x >= 0.5 and y < 0.5:
        return "B"
    if x < 0.5 and y >= 0.5:
        return "C"
    return "D"


def get_arrow_thresholds(thresholds, level):
    point_thr = thresholds.get("point_l2", 0.05)
    angle_thr = thresholds.get("angle_deg", 15.0)
    if level == 4:
        point_thr = thresholds.get("point_l2_level4", point_thr)
        angle_thr = thresholds.get("angle_deg_level4", angle_thr)
    return point_thr, angle_thr


def get_bbox_iou_threshold(thresholds, level):
    if level == 4:
        return thresholds.get("bbox_iou_level4", thresholds.get("bbox_iou", 0.5))
    if level == 5:
        return thresholds.get("bbox_iou_level5", thresholds.get("bbox_iou", 0.5))
    return thresholds.get("bbox_iou", 0.5)


def get_adaptive_config(config):
    adaptive = config.get("adaptive", {}) if isinstance(config, dict) else {}
    return {
        "enabled": adaptive.get("enabled", False),
        "start_level": adaptive.get("start_level", 3),
        "steps": adaptive.get("steps", 3),
    }


# =========================
# Resume support
# =========================
def load_completed_tasks(preds_path: str):
    completed = set()
    existing = []
    if os.path.exists(preds_path):
        with open(preds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "id" in obj and "level" in obj:
                    completed.add((obj["id"], obj["level"]))
                elif "qid" in obj:
                    completed.add(obj["qid"])
                existing.append(obj)
    return completed, existing


def load_completed_ids(preds_path: str):
    completed = set()
    existing = []
    if os.path.exists(preds_path):
        with open(preds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = obj.get("id") or obj.get("qid")
                if pid is not None:
                    completed.add(pid)
                existing.append(obj)
    return completed, existing


# =========================
# Images for bbox_5level (old logic)
# =========================
def get_bbox_images(question, base_dir):
    img = question.get("image", {}) if isinstance(question, dict) else {}
    if isinstance(img, dict) and "combined" in img:
        return [resolve_image_path(base_dir, img["combined"])]
    left = img.get("left_source") if isinstance(img, dict) else None
    right = img.get("right_target") if isinstance(img, dict) else None
    if left and right:
        return [resolve_image_path(base_dir, left), resolve_image_path(base_dir, right)]
    return []


def get_bbox_images_for_level(question_map, level, base_dir):
    if level == 1:
        q1 = question_map.get(1)
        return get_bbox_images(q1, base_dir) if q1 else []
    elif level in (2, 3):
        q2 = question_map.get(2)
        if q2:
            images = get_bbox_images(q2, base_dir)
            if images:
                return images
        q = question_map.get(level)
        return get_bbox_images(q, base_dir) if q else []
    elif level in (4, 5):
        q4 = question_map.get(4)
        if q4:
            images = get_bbox_images(q4, base_dir)
            if images:
                return images
        q = question_map.get(level)
        return get_bbox_images(q, base_dir) if q else []
    else:
        q = question_map.get(level)
        return get_bbox_images(q, base_dir) if q else []


# =========================
# FOV (new) image picking (g2s/s2s/gs_grounding/ge_view)
# =========================
def pick_images_fov(item: dict, dataset: dict, base_dir: str):
    ds_type = dataset.get("type")
    img_id = item.get("img_id", item.get("id"))
    view_id = item.get("view_id", item.get("view", None))
    pick_meta = {"ds_type": ds_type, "rule": None}

    # helper: try alternate well-known folders for missing images
    def try_fallback(filename: str, candidates: List[str]):
        if not filename:
            return None
        name = os.path.basename(filename)
        for d in candidates:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return None

    # g2s
    if ds_type == "g2s":
        pick_meta["rule"] = "g2s"
        src = item.get("source_image") or item.get("street_path") or item.get("street") or item.get("image_path")
        sat = item.get("sat_path") or item.get("satellite_path") or item.get("target_image") or item.get("satellite")
        src_p = resolve_image_path(base_dir, src) if src else None
        sat_p = resolve_image_path(base_dir, sat) if sat else None

        # cvusa-specific fallback: if street image not found, try cvusa/data/streetview/<basename>
        if (not src_p or not os.path.exists(src_p)):
            ds_path = dataset.get("path", "")
            if src and ("cvusa" in ds_path or "cvusa" in dataset.get("name", "")):
                name = os.path.basename(src)
                cand = os.path.join(base_dir, "cvusa", "data", "streetview", name)
                if os.path.exists(cand):
                    src_p = cand

        # special symmetric view (many street images for same sat view)
        # Only return here if we can resolve images from the source path; otherwise
        # continue to fov-specific directory-based fallback below.
        if view_id is not None and str(view_id).isdigit() and int(view_id) == 0:
            pick_meta["rule"] = "g2s_sym_view0_multi"
            imgs = []
            if src_p and os.path.exists(src_p):
                imgs = list_images_in_dir(os.path.dirname(src_p)) or [src_p]
            if imgs:
                hint = "These are multiple street-view perspectives (view_id==0). Please use ALL provided street-view images."
                return imgs, hint, pick_meta

        # For g2s we send the street-view image only (satellite may be present but not used here)
        if src_p and os.path.exists(src_p):
            pick_meta["rule"] = "g2s_street_only"
            return [src_p], "This is a street-view image (g2s).", pick_meta

        # fallback: no street image found
        # fov-specific fallback: for symmetric view (view_id==0) return all street images in sample dir
        ds_path = dataset.get("path", "")
        if ("fov" in ds_path or "fov" in dataset.get("name", "")):
            sample_id = item.get("sample_id") or item.get("sample") or item.get("img_id")
            view_id = item.get("view_id", item.get("view"))
            # symmetric case: return all images in sample directory when view_id==0
            if view_id is not None and str(view_id).isdigit() and int(view_id) == 0:
                if sample_id:
                    dir_candidate = os.path.join(base_dir, "fov", "data", "street", str(sample_id))
                    if os.path.exists(dir_candidate):
                            imgs = list_images_in_dir(dir_candidate)
                            if imgs:
                                pick_meta["rule"] = "g2s_fov_sym_view0_dir"
                                return imgs, "These are multiple street-view perspectives (view_id==0), from fov/data/street/<sample_id>. Please use ALL provided street-view images.", pick_meta
                # fallback: try to use basename to find any matching directory under fov/data/street
                name = os.path.basename(src) if src else None
                if name:
                    cand2 = os.path.join(base_dir, "fov", "data", "street", name)
                    if os.path.exists(cand2):
                        imgs = list_images_in_dir(os.path.dirname(cand2)) or [cand2]
                        pick_meta["rule"] = "g2s_fov_sym_view0_basename"
                        return imgs, "These are multiple street-view perspectives (view_id==0), matched by filename. Please use ALL provided street-view images.", pick_meta

            # non-symmetric fallbacks: try specific file locations
            if src and sample_id and view_id is not None:
                cand = os.path.join(base_dir, "fov", "data", "street", str(sample_id), f"{view_id}.jpg")
                if os.path.exists(cand):
                    pick_meta["rule"] = "g2s_fov_layout"
                    return [cand], "This is a street-view image (g2s fov layout).", pick_meta
            # try basename under fov/data/street
            if src:
                name = os.path.basename(src)
                cand2 = os.path.join(base_dir, "fov", "data", "street", name)
                if os.path.exists(cand2):
                    pick_meta["rule"] = "g2s_fov_basename"
                    return [cand2], "This is a street-view image (g2s fov basename).", pick_meta

        return [], "This is a street-view image (g2s).", pick_meta

    # s2s / s2g (satellite -> ground variants)
    if ds_type in ("s2s", "s2g"):
        pick_meta["rule"] = "s2s"
        tgt = item.get("target_image") or item.get("sat_path") or item.get("satellite_path") or item.get("image_path")
        tgt_p = resolve_image_path(base_dir, tgt) if tgt else None
        if tgt_p and os.path.exists(tgt_p):
            pick_meta["rule"] = "s2s_direct_path"
            return [tgt_p], "This is a satellite image (s2s).", pick_meta

        # fallback for satellite images (cvusa bingmap, fov satellite_arrow)
        if (not tgt_p or not os.path.exists(tgt_p)):
            candidates = []
            if GLOBAL_BASE_DIR:
                candidates.append(os.path.join(GLOBAL_BASE_DIR, "cvusa", "data", "bingmap"))
                candidates.append(os.path.join(GLOBAL_BASE_DIR, "fov", "data", "satellite_arrow"))
            candidates.append(os.path.join(base_dir, "cvusa", "data", "bingmap"))
            candidates.append(os.path.join(base_dir, "fov", "data", "satellite_arrow"))
            fb = try_fallback(tgt, candidates)
            if fb:
                tgt_p = fb

        # cvusa-specific fallback: try cvusa/data/bingmap/<basename>
        if (not tgt_p or not os.path.exists(tgt_p)):
            ds_path = dataset.get("path", "")
            if tgt and ("cvusa" in ds_path or "cvusa" in dataset.get("name", "")):
                name = os.path.basename(tgt)
                cand = os.path.join(base_dir, "cvusa", "data", "bingmap", name)
                if os.path.exists(cand):
                    tgt_p = cand

            if tgt_p and os.path.exists(tgt_p):
                return [tgt_p], "This is a satellite image (s2s).", pick_meta

        # ===== FOV satellite fallback =====
        ds_path = dataset.get("path", "")

        if "fov" in ds_path.lower() or "fov" in dataset.get("name", "").lower():

            sample_id = item.get("sample_id") or item.get("img_id")
            view_id = item.get("view_id") or item.get("view")

            if sample_id is not None and view_id is not None:

                folder = f"{int(sample_id):04d}"

                cand = os.path.join(
                    base_dir,
                    "fov",
                    "data",
                    "satellite_arrow",
                    folder,
                    f"view_{view_id}_annotated.jpg",
                )

                if os.path.exists(cand):
                    pick_meta["rule"] = "s2g_fov_layout"
                    return [cand], "This is a satellite image (s2g fov layout).", pick_meta

        return [], "This is a satellite image (s2s).", pick_meta

    # gs_grounding (your new bbox question=34 case)
    if ds_type == "gs_grounding":
        pick_meta["rule"] = "gs_grounding"
        level = item.get("level")
        pick_meta["level"] = level
        img_field = item.get("image")
        if isinstance(img_field, dict) and img_field.get("combined"):
            combined = img_field.get("combined")
            p = resolve_image_path(base_dir, combined)
            if p and os.path.exists(p):
                pick_meta["rule"] = "gs_grounding_combined"
                return [p], "This is a grounding task combined image.", pick_meta
            # cvusa fallback for combined images (often under common_question/bingmap -> cvusa/data/bingmap)
            ds_path = dataset.get("path", "")
            if combined and ("cvusa" in ds_path or "cvusa" in dataset.get("name", "")):
                name = os.path.basename(combined)
                cand = os.path.join(base_dir, "cvusa", "data", "bingmap", name)
                if os.path.exists(cand):
                    pick_meta["rule"] = "gs_grounding_combined_cvusa_bingmap"
                    return [cand], "This is a grounding task combined image (cvusa fallback).", pick_meta
            # fov fallback: sometimes combined images live under fov/data/*
            if combined and ("fov" in ds_path or "fov" in dataset.get("name", "")):
                name = os.path.basename(combined)
                cand = os.path.join(base_dir, "fov", "data", name)
                if os.path.exists(cand):
                    pick_meta["rule"] = "gs_grounding_combined_fov_data"
                    return [cand], "This is a grounding task combined image (fov fallback).", pick_meta
        return [], "This is a grounding task image.", pick_meta

    # ge_view
    if ds_type == "ge_view":
        pick_meta["rule"] = "ge_view"
        # If the JSON already contains an explicit image field, use it first.
        direct_image_fields = [
            ("image_path", item.get("image_path")),
            ("image", item.get("image")),
            ("street_image", item.get("street_image")),
            ("street_path", item.get("street_path")),
            ("source_image", item.get("source_image")),
            ("target_image", item.get("target_image")),
            ("level1_candidate_image", item.get("level1_candidate_image")),
            ("level1_image", item.get("level1_image")),
            ("candidate_image", item.get("candidate_image")),
        ]
        for field_name, field_value in direct_image_fields:
            if isinstance(field_value, str) and field_value:
                p = resolve_image_path(base_dir, field_value)
                if p and os.path.exists(p):
                    pick_meta["rule"] = f"ge_view_{field_name}"
                    pick_meta["found"] = field_value
                    return [p], f"This is a ge_view task image ({field_name}).", pick_meta
        # Prefer level1 candidate image (e.g. "0839_view1_L1.jpg") then satellite image
        ds_path = dataset.get("path", "")
        ds_json_name = os.path.basename(ds_path)
        ds_dir = os.path.dirname(resolve_path(base_dir, ds_path)) if ds_path else resolve_path(base_dir, "")

        # try to infer subfolder name from dataset json filename, e.g. arrow_select_test.jsonl -> arrow_select
        subfolder = None
        if "arrow_select" in ds_json_name:
            subfolder = "arrow_select"
        elif "arrow_street" in ds_json_name:
            subfolder = "arrow_street"

        cand_dirs = [ds_dir]
        if subfolder:
            cand_dirs.insert(0, os.path.join(ds_dir, subfolder))

        candidates = []
        level1 = item.get("level1_candidate_image") or item.get("level1_image")
        sat = item.get("satellite_image") or item.get("sat_image") or item.get("sat_path")

        def find_in_dirs(name: str):
            if not name:
                return None
            for d in cand_dirs:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    return p
            # fallback: try resolving relative to base_dir
            p = resolve_image_path(base_dir, name)
            if p and os.path.exists(p):
                return p
            # try global fallback
            if GLOBAL_BASE_DIR:
                p = os.path.join(GLOBAL_BASE_DIR, name)
                if os.path.exists(p):
                    return p
            return None

        def find_named_variants(subfolder_name: str, filename_candidates: List[str]):
            exts = ["", ".jpg", ".jpeg", ".png", ".webp", ".bmp"]
            roots = [
                os.path.join(base_dir, "fov", "gs_view", subfolder_name),
                os.path.join(ds_dir, subfolder_name),
                ds_dir,
            ]
            for root in roots:
                for name in filename_candidates:
                    for ext in exts:
                        cand = os.path.join(root, name if os.path.splitext(name)[1] else name + ext)
                        if os.path.exists(cand):
                            return cand
            return None

        if isinstance(level1, str) and level1:
            p = find_in_dirs(level1)
            if p:
                pick_meta["rule"] = "ge_view_level1"
                pick_meta["found"] = level1
                return [p], "This is a ge_view task image (level1).", pick_meta

        if isinstance(sat, str) and sat:
            # satellite image may be like '0839.jpg'
            p = find_in_dirs(sat)
            if p:
                pick_meta["rule"] = "ge_view_satellite"
                pick_meta["found"] = sat
                return [p], "This is a ge_view task image (satellite).", pick_meta

        sample_id = item.get("sample_id") or item.get("sample") or item.get("img_id")
        if sample_id:
            prefix = str(sample_id)
            for d in cand_dirs:
                matches = sorted(glob.glob(os.path.join(d, f"{prefix}*")))
                if matches:
                    pick_meta["rule"] = "ge_view_prefix_match"
                    pick_meta["found_dir"] = d
                    return [matches[0]], "This is a ge_view task image (prefix match).", pick_meta

        # gs_view specific naming conventions (fov/gs_view)
        if sample_id:
            # arrow_select -> <sample>_view<view>_L1.jpg or <sample>_<view>_L1.jpg
            if subfolder == "arrow_select":
                view_id = item.get("view_id") or item.get("view")
                if view_id is not None:
                    candidates = [
                        f"{sample_id}_view{view_id}_L1",
                        f"{sample_id}_{view_id}_L1",
                    ]
                    cand = find_named_variants("arrow_select", candidates)
                    if cand:
                        pick_meta["rule"] = "ge_view_gs_view_arrow_select"
                        pick_meta["found"] = cand
                        return [cand], "This is a ge_view task image (gs_view arrow_select).", pick_meta

            # arrow_street -> <sample>_view<view>_arrow_sv or <sample>_<view>_arrow_sv
            if subfolder == "arrow_street":
                view_id = item.get("view_id") or item.get("view")
                if view_id is not None:
                    candidates = [
                        f"{sample_id}_view{view_id}_arrow_sv",
                        f"{sample_id}_{view_id}_arrow_sv",
                    ]
                    cand = find_named_variants("arrow_street", candidates)
                    if cand:
                        pick_meta["rule"] = "ge_view_gs_view_arrow_street"
                        pick_meta["found"] = cand
                        return [cand], "This is a ge_view task image (gs_view arrow_street).", pick_meta

        # fallback: try common fov/cvusa level1 dirs
        if isinstance(level1, str) and level1:
            candidates = [os.path.join(base_dir, "fov"), os.path.join(base_dir, "cvusa")]
            if GLOBAL_BASE_DIR:
                candidates.insert(0, os.path.join(GLOBAL_BASE_DIR, "fov"))
                candidates.insert(0, os.path.join(GLOBAL_BASE_DIR, "cvusa"))
            # prefer subfolder/data when subfolder is present
            fb_candidates = [os.path.join(c, "data") for c in candidates] + [os.path.join(base_dir, "fov", "data")]
            if subfolder:
                fb_candidates.insert(0, os.path.join(ds_dir, subfolder, "data"))
                fb_candidates.insert(0, os.path.join(ds_dir, subfolder))
            fb = try_fallback(level1, fb_candidates)
            if fb:
                pick_meta["rule"] = "ge_view_fallback"
                pick_meta["found"] = fb
                return [fb], "This is a ge_view task image (fallback).", pick_meta
        if isinstance(level1, str) and level1:
            for root, _, files in os.walk(ds_dir):
                if level1 in files:
                    p = os.path.join(root, level1)
                    pick_meta["rule"] = "ge_view_walk_level1"
                    pick_meta["found"] = p
                    return [p], "This is a ge_view task image (walk).", pick_meta

        return [], "This is a ge_view task image.", pick_meta

    return [], "", pick_meta


# =========================
# Eval: mcq_vqa (old)
# =========================
def eval_mcq_dataset(model_conf, dataset, base_dir, out_dir, max_samples=None):
    items = load_jsonl(resolve_path(base_dir, dataset["path"]))
    if max_samples:
        items = items[:max_samples]

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    completed_ids, existing_preds = load_completed_ids(preds_path)

    correct = 0
    total = 0
    per_l1 = {}
    per_l2 = {}

    # resume stats
    item_map = {item.get("img_id", str(i)): item for i, item in enumerate(items)}
    seen = set()
    for pred in existing_preds:
        pid = pred.get("id")
        if pid in seen:
            continue
        seen.add(pid)
        item = item_map.get(pid)
        if not item:
            continue
        is_correct = bool(pred.get("correct"))
        total += 1
        correct += 1 if is_correct else 0

        l1 = item.get("category_l1")
        l2 = item.get("category_l2")
        if l1:
            per_l1.setdefault(l1, {"correct": 0, "total": 0})
            per_l1[l1]["total"] += 1
            per_l1[l1]["correct"] += 1 if is_correct else 0
        if l2:
            per_l2.setdefault(l2, {"correct": 0, "total": 0})
            per_l2[l2]["total"] += 1
            per_l2[l2]["correct"] += 1 if is_correct else 0

    pending = [(i, it) for i, it in enumerate(items) if it.get("img_id", str(i)) not in completed_ids]
    concurrency = get_concurrency(model_conf)
    total_pending = len(pending)

    def process_one(idx, item):
        item_id = item.get("img_id", str(idx))
        question = item.get("question", "")
        options = item.get("options", [])
        ans = item.get("answer")
        category = normalize_category(item.get("category"))
        category_l1 = normalize_category(item.get("category_l1")) or category
        category_l2 = normalize_category(item.get("category_l2"))
        task_type = item.get("task") or item.get("type") or dataset.get("type")
        dataset_name = item.get("dataset") or dataset.get("name")

        # choose image
        if item.get("view_from") == "street":
            image_paths = [resolve_image_path(base_dir, item.get("street_path"))]
            hint = "This is a street-view image."
        else:
            image_paths = [resolve_image_path(base_dir, item.get("sat_path"))]
            hint = "This is a satellite image."

        prompt = question_to_prompt(question, options, hint, force_mcq=True)
        raw, pred = run_mcq(model_conf, prompt, image_paths, base_dir=base_dir)
        is_correct = (pred == ans)

        return {
            "id": item_id,
            "question": question,
            "options": options,
            "pred": pred,
            "answer": ans,
            "correct": is_correct,
            "raw": raw,
            "task": task_type,
            "dataset": dataset_name,
            "category": category,
            "category_l1": category_l1,
            "category_l2": category_l2,
        }, is_correct, item.get("category_l1"), item.get("category_l2")

    with open(preds_path, "a", encoding="utf-8") as f:
        if total_pending > 0:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(process_one, idx, item): item.get("img_id", str(idx)) for idx, item in pending}
                done = 0
                for fut in as_completed(futs):
                    try:
                        rec, is_correct, l1, l2 = fut.result()
                    except Exception as e:
                        print(f"\n  Warning: sample {futs[fut]} failed: {e}")
                        continue

                    done += 1
                    if tqdm is None:
                        if done == 1 or done % max(1, total_pending // 10) == 0:
                            print(f"  Progress: {done}/{total_pending}", end="\r")

                    total += 1
                    correct += 1 if is_correct else 0
                    if l1:
                        per_l1.setdefault(l1, {"correct": 0, "total": 0})
                        per_l1[l1]["total"] += 1
                        per_l1[l1]["correct"] += 1 if is_correct else 0
                    if l2:
                        per_l2.setdefault(l2, {"correct": 0, "total": 0})
                        per_l2[l2]["total"] += 1
                        per_l2[l2]["correct"] += 1 if is_correct else 0

                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
        else:
            print("  Progress: 0/0", end="\r")
    print()

    metrics = {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "category_l1": {k: {"correct": v["correct"], "total": v["total"], "acc": v["correct"] / v["total"] if v["total"] else 0.0} for k, v in per_l1.items()},
        "category_l2": {k: {"correct": v["correct"], "total": v["total"], "acc": v["correct"] / v["total"] if v["total"] else 0.0} for k, v in per_l2.items()},
    }
    return metrics


# =========================
# Eval: bbox_5level (old) + FIX resume + robust MCQ parsing
# =========================
def _run_bbox_task(model_conf, base_dir, thresholds, task):
    item_id = task["item_id"]
    level = task["level"]
    q = task["question"]
    question_map = task["question_map"]

    question = q.get("question", "")
    options = q.get("options")
    image_paths = get_bbox_images_for_level(question_map, level, base_dir)

    if level in (1, 2):
        prompt = question_to_prompt(question, options, "", force_mcq=True)
        raw, pred = run_mcq(model_conf, prompt, image_paths, base_dir=base_dir)
        ans = q.get("answer")
        is_correct = (pred == ans)
        return level, {
            "id": item_id,
            "level": level,
            "pred": pred,
            "answer": ans,
            "correct": is_correct,
            "raw": raw,
        }, is_correct, None

    prompt = question_to_prompt(question, None, "", force_mcq=False)
    raw, pred_bbox = run_bbox(model_conf, prompt, image_paths, base_dir=base_dir)
    ans = q.get("answer")
    iou = bbox_iou(pred_bbox, ans) if pred_bbox else 0.0
    iou_thr = get_bbox_iou_threshold(thresholds, level)
    is_correct = iou >= iou_thr
    return level, {
        "id": item_id,
        "level": level,
        "pred": pred_bbox,
        "answer": ans,
        "iou": iou,
        "iou_thr": iou_thr,
        "correct": is_correct,
        "raw": raw,
    }, is_correct, iou


def eval_bbox_5level_dataset(model_conf, dataset, base_dir, out_dir, thresholds, adaptive, max_samples=None):
    items = load_jsonl(resolve_path(base_dir, dataset["path"]))
    if max_samples:
        items = items[:max_samples]

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    completed, existing_preds = load_completed_tasks(preds_path)

    # stats (resume)
    stats = {"levels": {}}
    for pred in existing_preds:
        level = pred.get("level")
        if level is None:
            continue
        lk = str(level)
        stats["levels"].setdefault(lk, {"correct": 0, "total": 0, "ious": []})
        stats["levels"][lk]["total"] += 1
        if pred.get("correct"):
            stats["levels"][lk]["correct"] += 1
        if pred.get("iou") is not None:
            stats["levels"][lk]["ious"].append(float(pred["iou"]))

    concurrency = get_concurrency(model_conf)

    # Tasks for the non-adaptive path. Adaptive logic is kept for compatibility.
    if adaptive["enabled"]:
        raise RuntimeError("adaptive enabled in config, but merged runner currently expects adaptive=false for bbox_5level.")
    else:
        tasks = []
        for idx, item in enumerate(items):
            item_id = item.get("img_id", str(idx))
            question_map = {q.get("level"): q for q in item.get("questions", [])}
            for q in item.get("questions", []):
                level = q.get("level")
                if (item_id, level) in completed:
                    continue
                lk = str(level)
                stats["levels"].setdefault(lk, {"correct": 0, "total": 0, "ious": []})
                # Do not increment total here before completion, or resume accounting will double count.
                tasks.append({"item_id": item_id, "level": level, "question": q, "question_map": question_map})

        total_tasks = len(tasks)

        # progress bar
        pbar = tqdm(total=total_tasks, desc=f"Processing {dataset.get('name')}") if tqdm else None

        with open(preds_path, "a", encoding="utf-8") as f:
            if total_tasks > 0:
                with ThreadPoolExecutor(max_workers=concurrency) as ex:
                    futs = {ex.submit(_run_bbox_task, model_conf, base_dir, thresholds, t): t for t in tasks}
                    done = 0
                    for fut in as_completed(futs):
                        task = futs[fut]
                        try:
                            level, pred_record, is_correct, iou = fut.result()
                        except Exception as e:
                            print(f"\n  Warning: sample {task['item_id']} level {task['level']} failed: {e}")
                            if pbar:
                                pbar.update(1)
                            continue

                        done += 1
                        if pbar:
                            pbar.update(1)
                        else:
                            if done == 1 or done % max(1, total_tasks // 10) == 0:
                                print(f"  Progress: {done}/{total_tasks}", end="\r")

                        lk = str(level)
                        stats["levels"].setdefault(lk, {"correct": 0, "total": 0, "ious": []})
                        stats["levels"][lk]["total"] += 1
                        if is_correct:
                            stats["levels"][lk]["correct"] += 1
                        if iou is not None:
                            stats["levels"][lk]["ious"].append(float(iou))

                        f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                        f.flush()
            else:
                print("  Progress: 0/0", end="\r")

        if pbar:
            pbar.close()

    metrics = {"levels": {}, "adaptive": None}
    for lk, v in stats["levels"].items():
        acc = v["correct"] / v["total"] if v["total"] else 0.0
        mean_iou = sum(v["ious"]) / len(v["ious"]) if v["ious"] else None
        metrics["levels"][lk] = {
            "accuracy": acc,
            "correct": v["correct"],
            "total": v["total"],
            "mean_iou": mean_iou,
        }
    return metrics


# =========================
# Eval: arrow_5level + arrow_mcq (old) using OpenAI SDK
# =========================
def get_combined_dirs(base_dir, dataset):
    dirs = dataset.get("combined_dirs")
    if not dirs:
        single = dataset.get("combined_dir")
        if single:
            dirs = [single]
        else:
            dataset_path = dataset.get("path", "")
            if "test" in dataset_path.lower():
                dirs = ["fov/5level_array/test_arrary"]
            elif "train" in dataset_path.lower():
                dirs = ["fov/5level_array/train_array"]
            else:
                dirs = ["fov/5level_array/test_arrary", "fov/5level_array/train_array"]
    return [resolve_path(base_dir, d) for d in dirs if d]


def find_combined_path(combined_dirs, sample_id, view_id):
    name = f"arrow_{sample_id}_{view_id}.jpg"
    for d in combined_dirs:
        candidate = os.path.join(d, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(combined_dirs[0], name) if combined_dirs else None


def _run_arrow_task(model_conf, base_dir, thresholds, task):
    item_id = task["item_id"]
    level = task["level"]
    q = task["question"]
    image_paths = task["image_paths"]

    question = q.get("question", "")
    options = q.get("options")

    if level in (1, 2):
        if level == 1:
            hint = "Image1=left street-view, right satellite. Select the direction where street-view is located relative to satellite image."
        else:
            hint = "Image1=left satellite, right street-view. Select the direction of street-view relative to satellite image."
        prompt = question_to_prompt(question, options, hint, force_mcq=True)
        raw, pred = run_mcq(model_conf, prompt, image_paths, base_dir=base_dir)
        ans = q.get("answer")
        is_correct = (pred == ans)
        return level, {
            "id": item_id,
            "level": level,
            "pred": pred,
            "answer": ans,
            "correct": is_correct,
            "raw": raw,
        }, {
            "is_correct": is_correct,
            "l2": None,
            "angle_present": False,
            "angle_ok": None,
            "angle_err": None,
            "quad_ok": None,
        }

    hint = "Background: The arrow indicates the viewing direction of street-view relative to satellite image location.\n"
    if level == 3:
        hint += "Image1=left satellite, right street-view. Identify the quadrant the arrow points to."
        prompt = question_to_prompt(question, None, hint, force_mcq=False)
        raw, pred = run_point(model_conf, prompt, image_paths, base_dir=base_dir)
        pred_angle = None
    else:
        hint += "Image1=left satellite, right street-view. Draw an arrow on the left satellite image pointing to where street-view is located."
        prompt = question_to_prompt(question, None, hint, force_mcq=False)
        raw, pred, pred_angle = run_point_with_angle(model_conf, prompt, image_paths, base_dir=base_dir)

    ans_xy = q.get("answer_xy")
    ans_dir = q.get("answer_dir")
    ans_angle = q.get("answer_angle")

    point_thr, angle_thr = get_arrow_thresholds(thresholds, level)
    l2 = l2_dist(pred, ans_xy) if pred and ans_xy else None

    angle_present = ans_angle is not None
    angle_err = None
    angle_ok = True
    if angle_present:
        if pred_angle is None:
            angle_ok = False
        else:
            angle_err = angle_diff_deg(pred_angle, ans_angle)
            angle_ok = angle_err <= angle_thr

    is_correct = (l2 is not None and l2 <= point_thr and angle_ok)
    quad_ok = None
    if pred and ans_dir:
        quad_ok = quadrant_from_xy(pred) == ans_dir

    return level, {
        "id": item_id,
        "level": level,
        "pred": pred,
        "answer_xy": ans_xy,
        "answer_dir": ans_dir,
        "l2": l2,
        "angle_err": angle_err,
        "correct": is_correct,
        "raw": raw,
    }, {
        "is_correct": is_correct,
        "l2": l2,
        "angle_present": angle_present,
        "angle_ok": angle_ok,
        "angle_err": angle_err,
        "quad_ok": quad_ok,
    }


def eval_arrow_5level_dataset(model_conf, dataset, base_dir, out_dir, thresholds, adaptive, max_samples=None):
    items = load_jsonl(resolve_path(base_dir, dataset["path"]))
    if max_samples:
        items = items[:max_samples]

    level1_dir = resolve_path(base_dir, dataset["level1_image_dir"])
    combined_dirs = get_combined_dirs(base_dir, dataset)

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    completed, existing_preds = load_completed_tasks(preds_path)

    stats = {"levels": {}}
    for pred in existing_preds:
        level = pred.get("level")
        if level is None:
            continue
        lk = str(level)
        stats["levels"].setdefault(lk, {"correct": 0, "total": 0, "l2": [], "angle": [], "angle_correct": 0, "angle_total": 0, "quad_correct": 0})
        stats["levels"][lk]["total"] += 1
        if pred.get("correct"):
            stats["levels"][lk]["correct"] += 1
        if pred.get("l2") is not None:
            stats["levels"][lk]["l2"].append(float(pred["l2"]))
        if pred.get("angle_err") is not None:
            stats["levels"][lk]["angle"].append(float(pred["angle_err"]))
            stats["levels"][lk]["angle_total"] += 1

    if adaptive["enabled"]:
        raise RuntimeError("adaptive enabled in config, but merged runner currently expects adaptive=false for arrow_5level.")
    else:
        tasks = []
        for idx, item in enumerate(items):
            sample_id = item.get("sample_id")
            view_id = item.get("view_id")
            item_id = f"{sample_id}_{view_id}"
            combined_path = find_combined_path(combined_dirs, sample_id, view_id)
            if not combined_path or not os.path.exists(combined_path):
                raise RuntimeError(f"Missing combined image for {sample_id}/{view_id}")

            for q in item.get("questions", []):
                level = q.get("level")
                if (item_id, level) in completed:
                    continue

                if level == 1:
                    level1_filename = f"{sample_id}_view{view_id}_L1.jpg"
                    image_paths = [os.path.join(level1_dir, level1_filename)]
                else:
                    image_paths = [combined_path]

                tasks.append({"item_id": item_id, "level": level, "question": q, "image_paths": image_paths})

        concurrency = get_concurrency(model_conf)
        total_tasks = len(tasks)
        pbar = tqdm(total=total_tasks, desc=f"Processing {dataset.get('name')}") if tqdm else None

        with open(preds_path, "a", encoding="utf-8") as f:
            if total_tasks > 0:
                with ThreadPoolExecutor(max_workers=concurrency) as ex:
                    futs = {ex.submit(_run_arrow_task, model_conf, base_dir, thresholds, t): t for t in tasks}
                    done = 0
                    for fut in as_completed(futs):
                        task = futs[fut]
                        try:
                            level, pred_record, aux = fut.result()
                        except Exception as e:
                            print(f"\n  Warning: sample {task['item_id']} level {task['level']} failed: {e}")
                            if pbar:
                                pbar.update(1)
                            continue

                        done += 1
                        if pbar:
                            pbar.update(1)
                        else:
                            if done == 1 or done % max(1, total_tasks // 10) == 0:
                                print(f"  Progress: {done}/{total_tasks}", end="\r")

                        lk = str(level)
                        stats["levels"].setdefault(lk, {"correct": 0, "total": 0, "l2": [], "angle": [], "angle_correct": 0, "angle_total": 0, "quad_correct": 0})
                        stats["levels"][lk]["total"] += 1
                        if aux["is_correct"]:
                            stats["levels"][lk]["correct"] += 1
                        if aux.get("l2") is not None:
                            stats["levels"][lk]["l2"].append(float(aux["l2"]))
                        if aux.get("angle_present"):
                            stats["levels"][lk]["angle_total"] += 1
                            if aux.get("angle_err") is not None:
                                stats["levels"][lk]["angle"].append(float(aux["angle_err"]))
                            if aux.get("angle_ok"):
                                stats["levels"][lk]["angle_correct"] += 1
                        if aux.get("quad_ok"):
                            stats["levels"][lk]["quad_correct"] += 1

                        f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                        f.flush()
            else:
                print("  Progress: 0/0", end="\r")

        if pbar:
            pbar.close()

    metrics = {"levels": {}, "adaptive": None}
    for lk, v in stats["levels"].items():
        acc = v["correct"] / v["total"] if v["total"] else 0.0
        mean_l2 = sum(v["l2"]) / len(v["l2"]) if v["l2"] else None
        quad_acc = v["quad_correct"] / v["total"] if v["total"] else 0.0
        mean_angle = sum(v["angle"]) / len(v["angle"]) if v["angle"] else None
        angle_acc = (v["angle_correct"] / v["angle_total"]) if v["angle_total"] else None
        metrics["levels"][lk] = {
            "accuracy": acc,
            "quadrant_acc": quad_acc,
            "correct": v["correct"],
            "total": v["total"],
            "mean_l2": mean_l2,
            "mean_angle_error": mean_angle,
            "angle_acc": angle_acc,
        }
    return metrics


def eval_arrow_mcq_dataset(model_conf, dataset, base_dir, out_dir, max_samples=None):
    items = load_jsonl(resolve_path(base_dir, dataset["path"]))
    if max_samples:
        items = items[:max_samples]

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    completed_ids, existing_preds = load_completed_ids(preds_path)

    correct = 0
    total = 0
    for pred in existing_preds:
        if pred.get("id") in completed_ids:
            total += 1
            correct += 1 if pred.get("correct") else 0

    concurrency = get_concurrency(model_conf)
    pending = []
    for idx, item in enumerate(items):
        sample_id = item.get("sample_id", "unknown")
        view_id = item.get("view_id", "unknown")
        item_id = f"{sample_id}_{view_id}"
        if item_id in completed_ids:
            continue
        pending.append((idx, item, item_id))

    def process_one(idx, item, item_id):
        question = item.get("question", "")
        options = item.get("options", [])
        image_paths = [resolve_image_path(base_dir, item.get("image_path"))]
        prompt = question_to_prompt(question, options, "", force_mcq=True)
        raw, pred = run_mcq(model_conf, prompt, image_paths, base_dir=base_dir)
        ans = item.get("answer")
        is_correct = pred == ans
        return {
            "id": item_id,
            "pred": pred,
            "answer": ans,
            "correct": is_correct,
            "raw": raw,
        }, is_correct

    with open(preds_path, "a", encoding="utf-8") as f:
        if pending:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(process_one, idx, item, item_id): item_id for idx, item, item_id in pending}
                done = 0
                for fut in as_completed(futs):
                    item_id = futs[fut]
                    try:
                        rec, ok = fut.result()
                    except Exception as e:
                        print(f"\n  Warning: sample {item_id} failed: {e}")
                        continue
                    done += 1
                    total += 1
                    correct += 1 if ok else 0
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

    metrics = {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}
    return metrics


# =========================
# Eval: FOV gs_grounding in the nested-question bbox format
# =========================
def eval_gs_grounding_bbox(model_conf, dataset, base_dir, out_dir, max_samples=None):
    items = load_jsonl(resolve_path(base_dir, dataset["path"]))
    if max_samples:
        items = items[:max_samples]

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    completed, existing = load_completed_tasks(preds_path)
    iou_thr = float(dataset.get("iou_threshold", 0.5))

    total = 0
    correct = 0
    valid = 0
    center_hit = 0
    recall_03 = 0
    sum_iou = 0.0
    # per-level IOU aggregation (for separate mean_iou per level)
    sum_iou_by_level = {}
    count_by_level = {}

    # resume
    for obj in existing:
        if "qid" not in obj:
            continue
        total += 1
        if obj.get("pred_bbox") is not None:
            valid += 1
        correct += 1 if obj.get("correct") else 0
        if obj.get("center_in_gt"):
            center_hit += 1
        st = obj.get("bbox_stats") or {}
        this_iou = float(st.get("iou", 0.0) or 0.0)
        sum_iou += this_iou
        if this_iou >= 0.3:
            recall_03 += 1
        # try to extract level from qid like '0001.1.L3.q0'
        qid = obj.get("qid", "")
        m = re.search(r"\.L(\d+)\.q", qid)
        if m:
            lvl = int(m.group(1))
            sum_iou_by_level[lvl] = sum_iou_by_level.get(lvl, 0.0) + this_iou
            count_by_level[lvl] = count_by_level.get(lvl, 0) + 1

    tasks = []
    for idx, item in enumerate(items):
        img_id = item.get("img_id", str(idx))
        view_id = item.get("view_id", item.get("view", "NA"))
        qs = item.get("questions") or []
        for qi, q in enumerate(qs):
            if not isinstance(q, dict):
                continue
            level = q.get("level")
            qid = f"{img_id}.{view_id}.L{level}.q{qi}"
            if qid in completed:
                continue
            tasks.append((idx, item, q, level, qi, qid))

    concurrency = get_concurrency(model_conf)
    pbar = tqdm(total=len(tasks), desc=f"Processing {dataset.get('name')}") if tqdm else None

    def process_one(idx, item, q, level, qi, qid):
        local_img_id = item.get("img_id", str(idx))
        local_view_id = item.get("view_id", item.get("view", "NA"))
        question = q.get("question", "")
        category = normalize_category(q.get("category") or item.get("category"))
        category_l1 = normalize_category(q.get("category_l1") or item.get("category_l1") or category)
        category_l2 = normalize_category(q.get("category_l2") or item.get("category_l2"))
        gt_bbox = (
            q.get("answer")
            or q.get("target_bbox")
            or item.get("target_bbox")
            or item.get("gt_bbox")
            or item.get("answer_bbox")
            or item.get("answer")
        )
        if isinstance(gt_bbox, list) and len(gt_bbox) == 4:
            gt_bbox = clamp_bbox([float(x) for x in gt_bbox])
        else:
            gt_bbox = None

        def resolve_fov_path(p):
            if not p:
                return p
            cand = resolve_image_path(base_dir, p)
            if os.path.exists(cand):
                return cand
            cand = cand.replace("fov/train_satellite", "fov/data/satellite")
            cand = cand.replace("fov/test_satellite", "fov/data/satellite")
            cand = cand.replace("fov/train_street", "fov/data/street")
            cand = cand.replace("fov/test_street", "fov/data/street")
            cand = cand.replace("fov\\train_satellite", "fov\\data\\satellite")
            cand = cand.replace("fov\\test_satellite", "fov\\data\\satellite")
            cand = cand.replace("fov\\train_street", "fov\\data\\street")
            cand = cand.replace("fov\\test_street", "fov\\data\\street")
            return cand

        task_type = (item.get("task") or "").lower()
        dataset_name = (item.get("dataset") or "").lower()
        source_img = resolve_fov_path(item.get("source_image"))
        target_img = resolve_fov_path(item.get("target_image"))

        def resolve_bbox_image(dataset_name, task_type, img_id):
            if not img_id:
                return None
            if "fov" in dataset_name:
                base = os.path.join(base_dir, "fov", "gs_grounding", "bbox_images")
            else:
                base = os.path.join(base_dir, "cvusa", "gs_grounding", "bbox_images")
            if "sat2ground" in task_type or "s2g" in task_type:
                folder = os.path.join(base, "satellite")
                suffix = "_G2S_bbox.jpg"
            else:
                folder = os.path.join(base, "street")
                suffix = "_S2G_bbox.jpg"
            cand1 = os.path.join(folder, f"{img_id}{suffix}")
            if os.path.exists(cand1):
                return cand1
            if "_" in img_id:
                base_id = img_id.split("_")[0]
                cand2 = os.path.join(folder, f"{base_id}{suffix}")
                if os.path.exists(cand2):
                    return cand2
            return None

        if level == 3:
            bbox_img = resolve_bbox_image(dataset_name, task_type, local_img_id)
            if not bbox_img:
                raise RuntimeError(f"Missing bbox image for level3: img_id={local_img_id}, task={task_type}, dataset={dataset_name}")
            image_paths = [bbox_img, target_img]
            task_hint = "Two images. FIRST has bbox; SECOND is target. Return bbox in SECOND (pixel)."
        else:
            image_paths = [source_img, target_img]
            task_hint = "Two images. FIRST source; SECOND target. Return bbox in SECOND (pixel)."

        pick_meta = {"ds_type": dataset.get("type"), "rule": "gs_grounding_two_images", "level": level}

        region_hint = q.get("region_hint")
        if region_hint:
            task_hint = (task_hint or "") + f"\nRegion hint: {region_hint}."

        source_bbox = q.get("source_bbox") or q.get("bbox") or q.get("input_bbox") or item.get("source_bbox")
        if source_bbox:
            task_hint = (task_hint or "") + f"\nThe target object in the source image is indicated by bbox: {source_bbox}."

        prompt = question_to_prompt(question, None, task_hint, force_mcq=False)
        raw, pred_bbox = run_bbox(model_conf, prompt, image_paths, base_dir=base_dir)
        pred_bbox = clamp_bbox(pred_bbox) if pred_bbox else pred_bbox

        iou = bbox_iou(pred_bbox, gt_bbox) if pred_bbox and gt_bbox else 0.0
        ok = iou >= iou_thr
        hit = center_in_gt(pred_bbox, gt_bbox)

        return {
            "qid": qid,
            "img_id": local_img_id,
            "view_id": local_view_id,
            "task": item.get("task") or dataset.get("type"),
            "dataset": item.get("dataset") or dataset.get("name"),
            "level": level,
            "q_index": qi,
            "question": question,
            "source_bbox": source_bbox,
            "answer_bbox": gt_bbox,
            "pred_bbox": pred_bbox,
            "bbox_stats": {"iou": iou},
            "iou_threshold": iou_thr,
            "correct": ok,
            "center_in_gt": hit,
            "raw": raw,
            "image_paths": [os.path.relpath(p, start=base_dir) if p else p for p in image_paths],
            "pick_meta": pick_meta,
            "category": category,
            "category_l1": category_l1,
            "category_l2": category_l2,
        }, ok, iou, pred_bbox is not None, hit

    with open(preds_path, "a", encoding="utf-8") as f:
        if tasks:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(process_one, *t): t[-1] for t in tasks}
                for fut in as_completed(futs):
                    qid = futs[fut]
                    try:
                        rec, ok, iou, is_valid, hit = fut.result()
                    except Exception as e:
                        print(f"\n  Warning: sample {qid} failed: {e}")
                        if pbar:
                            pbar.update(1)
                        continue

                    total += 1
                    correct += 1 if ok else 0
                    valid += 1 if is_valid else 0
                    center_hit += 1 if hit else 0
                    this_iou = float(iou)
                    sum_iou += this_iou
                    if this_iou >= 0.3:
                        recall_03 += 1
                    lvl = rec.get("level")
                    if lvl is not None:
                        try:
                            lvl_i = int(lvl)
                        except Exception:
                            lvl_i = None
                        if lvl_i is not None:
                            sum_iou_by_level[lvl_i] = sum_iou_by_level.get(lvl_i, 0.0) + this_iou
                            count_by_level[lvl_i] = count_by_level.get(lvl_i, 0) + 1

                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    if pbar:
                        pbar.update(1)

    if pbar:
        pbar.close()

    # compute per-level mean IOU for levels present (explicitly include level 3 and 4)
    mean_iou_by_level = {}
    for lvl, s in sum_iou_by_level.items():
        cnt = count_by_level.get(lvl, 0)
        mean_iou_by_level[str(lvl)] = (s / cnt) if cnt else None

    metrics = {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "valid_bbox_rate": valid / total if total else 0.0,
        "center_hit_rate": center_hit / total if total else 0.0,
        "recall_iou_0.3": recall_03 / total if total else 0.0,
        "mean_iou": sum_iou / total if total else 0.0,
        "mean_iou_by_level": mean_iou_by_level,
        "mean_iou_level3": mean_iou_by_level.get("3"),
        "mean_iou_level4": mean_iou_by_level.get("4"),
    }
    return metrics


# =========================
# Eval: FOV MCQ (g2s/s2s/ge_view) using pick_images_fov
# =========================
def eval_fov_mcq_dataset(model_conf, dataset, base_dir, out_dir, max_samples=None):
    items = load_jsonl(resolve_path(base_dir, dataset["path"]))
    if max_samples:
        items = items[:max_samples]

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    completed_ids, existing = load_completed_ids(preds_path)

    total = 0
    correct = 0
    for obj in existing:
        if obj.get("qid") and obj.get("qid") in completed_ids:
            total += 1
            correct += 1 if obj.get("correct") else 0

    pending = []
    for idx, item in enumerate(items):
        img_id = item.get("img_id") or item.get("sample_id") or item.get("sample") or item.get("id") or str(idx)
        view_id = item.get("view_id", item.get("view", "NA"))
        qs = item.get("questions") if isinstance(item.get("questions"), list) else None
        if qs:
            for qi, q in enumerate(qs):
                qid = f"{img_id}.{view_id}.q{qi}"
                if qid in completed_ids:
                    continue
                pending.append((idx, item, q, qid))
        else:
            q_index = item.get("q_index", 0)
            qid = f"{img_id}.{view_id}.q{q_index}"
            if qid in completed_ids:
                continue
            pending.append((idx, item, None, qid))

    concurrency = get_concurrency(model_conf)
    pbar = tqdm(total=len(pending), desc=f"Processing {dataset.get('name')}") if tqdm else None

    def process_one(idx, item, q, qid):
        # support nested question objects (item may contain 'questions' list)
        if q is not None:
            question = q.get("question", "")
            options = q.get("options")
            answer = q.get("answer")
        else:
            question = item.get("question", "")
            options = item.get("options")
            answer = item.get("answer")

        # allow question-level overrides (e.g., per-question image fields)
        tmp_item = dict(item)
        if q is not None and isinstance(q, dict):
            tmp_item.update(q)

        category = normalize_category(tmp_item.get("category"))
        category_l1 = normalize_category(tmp_item.get("category_l1")) or category
        category_l2 = normalize_category(tmp_item.get("category_l2"))
        task_type = tmp_item.get("task") or tmp_item.get("type") or dataset.get("type")
        dataset_name = tmp_item.get("dataset") or dataset.get("name")

        image_paths, task_hint, pick_meta = pick_images_fov(tmp_item, dataset, base_dir)

        prompt = question_to_prompt(question, options, task_hint, force_mcq=True)
        raw, pred = run_mcq(model_conf, prompt, image_paths, base_dir=base_dir)
        ok = pred == answer
        # best-effort img_id/view_id
        img_id = tmp_item.get("img_id") or tmp_item.get("sample_id") or tmp_item.get("sample") or tmp_item.get("id")
        view_id = tmp_item.get("view_id", tmp_item.get("view"))
        return {
            "qid": qid,
            "id": qid,  # keep compatibility
            "dataset_type": dataset.get("type"),
            "img_id": img_id,
            "view_id": view_id,
            "question": question,
            "options": options,
            "answer": answer,
            "pred": pred,
            "correct": ok,
            "raw": raw,
            "task": task_type,
            "dataset": dataset_name,
            "category": category,
            "category_l1": category_l1,
            "category_l2": category_l2,
            "image_paths": [os.path.relpath(p, start=base_dir) if p else p for p in image_paths],
            "pick_meta": pick_meta,
        }, ok

    with open(preds_path, "a", encoding="utf-8") as f:
        if pending:
                with ThreadPoolExecutor(max_workers=concurrency) as ex:
                    futs = {ex.submit(process_one, idx, item, q, qid): qid for idx, item, q, qid in pending}
                    for fut in as_completed(futs):
                        qid = futs[fut]
                        try:
                            rec, ok = fut.result()
                        except Exception as e:
                            print(f"\n  Warning: sample {qid} failed: {e}")
                            if pbar:
                                pbar.update(1)
                            continue
                        total += 1
                        correct += 1 if ok else 0
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        if pbar:
                            pbar.update(1)

    if pbar:
        pbar.close()

    metrics = {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}
    return metrics


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval_config.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    base_dir = os.path.dirname(config_path)

    global GLOBAL_BASE_DIR
    GLOBAL_BASE_DIR = base_dir

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    outputs_root = os.path.join(base_dir, "outputs")
    ensure_dir(outputs_root)

    # model selection: if config has models -> loop; else single MODEL_CONF
    models = config.get("models")
    if not models:
        models = [MODEL_CONF]
    else:
        # fill defaults
        for mc in models:
            for k, v in MODEL_CONF.items():
                if mc.get(k) is None:
                    mc[k] = v

    datasets = config.get("datasets", [])
    if not datasets:
        raise RuntimeError("Config is missing datasets=[...]")

    total_datasets = len(datasets)

    for model_conf in models:
        model_name = model_conf.get("name") or model_conf.get("model") or "model"
        model_out = os.path.join(outputs_root, model_name)
        ensure_dir(model_out)

        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}\n")

        summary = {"model": model_name, "datasets": {}}

        for ds_idx, dataset in enumerate(datasets, 1):
            ds_name = dataset["name"]
            ds_type = dataset["type"]
            ds_out = os.path.join(model_out, ds_name)
            ensure_dir(ds_out)

            print(f"\n[{ds_idx}/{total_datasets}] Dataset: {ds_name}")
            print(f"Type: {ds_type}")

            metrics_file = os.path.join(ds_out, "metrics.json")
            dataset_complete = False
            if os.path.exists(metrics_file):
                try:
                    existing_metrics = json.load(open(metrics_file, "r", encoding="utf-8"))
                    if "error" not in existing_metrics:
                        if args.max_samples is None:
                            dataset_complete = True
                        else:
                            # if you run subset, don't skip unless total >= max_samples
                            if existing_metrics.get("total", 0) >= args.max_samples:
                                dataset_complete = True
                    if dataset_complete:
                        print("Completed already, skipping. Delete metrics.json to rerun.")
                        summary["datasets"][ds_name] = existing_metrics
                        continue
                except Exception:
                    pass

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if ds_type == "mcq_vqa":
                        metrics = eval_mcq_dataset(model_conf, dataset, base_dir, ds_out, args.max_samples)
                    elif ds_type == "bbox_5level":
                        metrics = eval_bbox_5level_dataset(
                            model_conf, dataset, base_dir, ds_out,
                            config.get("thresholds", {}),
                            get_adaptive_config(config),
                            args.max_samples
                        )
                    elif ds_type == "arrow_5level":
                        metrics = eval_arrow_5level_dataset(
                            model_conf, dataset, base_dir, ds_out,
                            config.get("thresholds", {}),
                            get_adaptive_config(config),
                            args.max_samples
                        )
                    elif ds_type == "arrow_mcq":
                        metrics = eval_arrow_mcq_dataset(model_conf, dataset, base_dir, ds_out, args.max_samples)

                    # ---- FOV new types ----
                    elif ds_type in ("g2s", "s2s", "s2g", "ge_view"):
                        metrics = eval_fov_mcq_dataset(model_conf, dataset, base_dir, ds_out, args.max_samples)
                    elif ds_type == "gs_grounding":
                        metrics = eval_gs_grounding_bbox(model_conf, dataset, base_dir, ds_out, args.max_samples)
                    else:
                        raise RuntimeError(f"Unknown dataset type: {ds_type}")

                    dump_json(metrics_file, metrics)
                    summary["datasets"][ds_name] = metrics
                    print(f"Done: {ds_name}")

                    if args.sleep > 0:
                        time.sleep(args.sleep)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Warning: attempt {attempt+1} failed: {str(e)}")
                        print(f"  Retrying... ({attempt+2}/{max_retries})")
                        time.sleep(2)
                    else:
                        err = {"error": str(e)}
                        dump_json(metrics_file, err)
                        summary["datasets"][ds_name] = err
                        print(f"Failed: {ds_name} - {str(e)}")

        dump_json(os.path.join(model_out, "summary.json"), summary)

    print("Done.")


if __name__ == "__main__":
    main()
    
