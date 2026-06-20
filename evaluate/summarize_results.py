import argparse
import json
import os

def calculate_vision_score(levels_data, threshold=0.5, require_continuous=True):
    """
    Compute a level-style score using a vision-chart-like progression.

    Args:
        levels_data: per-level evaluation results
        threshold: passing threshold, default 0.5
        require_continuous: whether passing must be continuous across levels
    """
    level_scores = {1: 1.0, 2: 1.2, 3: 1.5, 4: 2.0, 5: 2.5}
    achieved_levels = []
    
    for level in sorted([1, 2, 3, 4, 5]):
        level_key = str(level)
        if level_key in levels_data:
            metrics = levels_data[level_key]
            accuracy = metrics.get("accuracy", 0)
            
            if accuracy >= threshold:
                achieved_levels.append(level)
            elif require_continuous and achieved_levels:
                break

    if achieved_levels:
        highest_level = max(achieved_levels)
        return level_scores[highest_level], highest_level, achieved_levels
    return 0.0, 0, []

def find_eval_config(start_dir, max_up=6):
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        candidate = os.path.join(cur, "eval_config.json")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None

def load_eval_config_map(start_dir):
    config_path = find_eval_config(start_dir)
    if not config_path:
        return None, None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    dataset_map = {d["name"]: d for d in config.get("datasets", []) if "name" in d and "path" in d}
    base_dir = os.path.dirname(config_path)
    return dataset_map, base_dir

def normalize_option_text(s):
    if s is None:
        return ""
    # remove leading option labels like "A) " or "A. "
    s = s.strip()
    if len(s) >= 2 and s[1] in [")", "."] and s[0].isalpha():
        s = s[2:].strip()
    return " ".join(s.lower().split())

def build_option_signature(options):
    if not options or not isinstance(options, list):
        return None
    norm = [normalize_option_text(o) for o in options]
    return "||".join(norm)

def build_option_category_map_hardcoded():
    """
    Hardcoded option-signature -> category mapping.
    This is intentionally file-free so the script can be moved elsewhere.
    """
    mapping = {}

    def add(options, category):
        sig = build_option_signature(options)
        if sig:
            mapping[sig] = category

    # ---- FOV g2s ----
    add(
        ["Visible in Satellite", "Not visible in Satellite"],
        "facade",
    )
    add(
        ["Symmetric", "Not symmetric"],
        "symmetry",
    )
    add(
        ["Flat roof", "Pitched / sloped roof", "Mixed roof (flat + pitched sections)", "Multiple separate roof blocks with different forms"],
        "roof",
    )

    # ---- FOV s2g ----
    add(
        ["Trees", "Shrubs / bushes", "Grass / lawn", "No obvious vegetation"],
        "vegetation_sector",
    )
    add(
        ["Occluded (blocked)", "Not occluded (clear view)"],
        "occlusion_binary",
    )
    add(
        ["Left side", "Right side", "Near the center (mostly straight ahead)", "Not visible in the forward view"],
        "relative_position",
    )
    add(
        ["Asphalt", "Concrete", "Grass", "Bare soil / dirt"],
        "ground_material",
    )
    add(
        ["The first building is taller", "The second building is taller", "They are about the same height", "Only one of them is visible"],
        "height_comparison",
    )
    add(
        ["Light (white/cream/light gray)", "Medium (tan/brown)", "Dark (dark gray/black)", "Red / brick-like"],
        "facade_color",
    )

    # ---- CVUSA g2s ----
    add(
        ["Building A", "Building B", "They are about the same distance", "Both are not visible in Satellite"],
        "distance_to_camera",
    )
    add(
        ["Building A", "Building B", "They are about the same area", "Neither is clearly identifiable in Satellite"],
        "building_footprint_area",
    )
    add(
        ["Flat roof", "Gabled roof", "Arched/curved roof", "Mixed/unclear roof form"],
        "roof_form",
    )
    add(
        ["Connected", "Not connected", "Partially connected", "Cannot determine from satellite"],
        "building_connection",
    )
    add(
        ["Yes, they are connected", "No, they are separate", "Connected by a narrow covered link only", "Not visible in Satellite"],
        "building_connection",
    )
    add(
        ["Yes, they are connected", "No, they are separate", "Connected by a narrow link only", "Not visible in Satellite"],
        "building_connection",
    )

    # ---- CVUSA s2g ----
    add(
        ["Occluded", "Not occluded"],
        "visibility",
    )
    add(
        ["The center region", "The left-quarter region", "The right-quarter region", "The far edge region"],
        "location",
    )
    add(
        ["Trees", "Shrubs", "Grass", "No visible vegetation"],
        "vegetation",
    )
    add(
        ["Building A is taller", "Building B is taller", "They are approximately the same height", "Only one building is clearly visible"],
        "height",
    )

    return mapping

def build_qid_category_map(dataset_path):
    qid_map = {}
    if not os.path.exists(dataset_path):
        return qid_map
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            img_id = item.get("img_id") or item.get("sample_id") or item.get("sample") or item.get("id") or str(idx)
            view_id = item.get("view_id", item.get("view", "NA"))
            qs = item.get("questions") if isinstance(item.get("questions"), list) else None
            if qs:
                for qi, q in enumerate(qs):
                    qid = f"{img_id}.{view_id}.q{qi}"
                    l1 = q.get("category_l1") or item.get("category_l1") or item.get("category")
                    l2 = q.get("category_l2") or item.get("category_l2")
                    qid_map[qid] = (l1, l2)
            else:
                q_index = item.get("q_index", 0)
                qid = f"{img_id}.{view_id}.q{q_index}"
                l1 = item.get("category_l1") or item.get("category")
                l2 = item.get("category_l2")
                qid_map[qid] = (l1, l2)
    return qid_map

def compute_category_stats(preds_path, qid_map, option_map=None):
    per_l1 = {}
    per_l2 = {}
    if not os.path.exists(preds_path):
        return per_l1, per_l2
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qid") or obj.get("id")
            l1 = l2 = None
            pred_l1 = obj.get("category_l1") or obj.get("category")
            pred_l2 = obj.get("category_l2")
            if pred_l1:
                l1 = pred_l1
            if pred_l2:
                l2 = pred_l2
            if qid and qid_map and qid in qid_map:
                mapped_l1, mapped_l2 = qid_map.get(qid, (None, None))
                if not l1:
                    l1 = mapped_l1
                if not l2:
                    l2 = mapped_l2
            if not l1 and option_map:
                sig = build_option_signature(obj.get("options"))
                if sig and sig in option_map:
                    l1 = option_map.get(sig)
            is_corr = bool(obj.get("correct"))
            if l1:
                per_l1.setdefault(l1, {"correct": 0, "total": 0})
                per_l1[l1]["total"] += 1
                per_l1[l1]["correct"] += 1 if is_corr else 0
            if l2:
                per_l2.setdefault(l2, {"correct": 0, "total": 0})
                per_l2[l2]["total"] += 1
                per_l2[l2]["correct"] += 1 if is_corr else 0
    # convert to acc format
    cat_l1 = {k: {"correct": v["correct"], "total": v["total"], "acc": v["correct"] / v["total"] if v["total"] else 0.0} for k, v in per_l1.items()}
    cat_l2 = {k: {"correct": v["correct"], "total": v["total"], "acc": v["correct"] / v["total"] if v["total"] else 0.0} for k, v in per_l2.items()}
    return cat_l1, cat_l2

def compute_category_metrics_from_predictions(ds_dir, option_map):
    preds_path = os.path.join(ds_dir, "predictions.jsonl")
    if not os.path.exists(preds_path):
        return None
    cat_l1, _ = compute_category_stats(preds_path, qid_map={}, option_map=option_map)
    if not cat_l1:
        return None
    total = sum(v["total"] for v in cat_l1.values())
    correct = sum(v["correct"] for v in cat_l1.values())
    return {
        "overall": {
            "accuracy": (correct / total) if total else 0.0,
            "correct": correct,
            "total": total,
        },
        "category_l1": cat_l1,
    }

def compute_metrics_from_predictions(ds_dir, ds_name, option_map):
    preds_path = os.path.join(ds_dir, "predictions.jsonl")
    if not os.path.exists(preds_path):
        return None

    # try MCQ-style categories first
    cat_l1, cat_l2 = compute_category_stats(preds_path, qid_map={}, option_map=option_map)
    if cat_l1:
        total = sum(v["total"] for v in cat_l1.values())
        correct = sum(v["correct"] for v in cat_l1.values())
        return {
            "accuracy": (correct / total) if total else 0.0,
            "correct": correct,
            "total": total,
            "category_l1": cat_l1,
            "category_l2": cat_l2,
        }

    # otherwise, try level-based metrics
    levels = {}
    total = 0
    correct = 0
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            if obj.get("correct"):
                correct += 1
            level = obj.get("level")
            if level is None:
                continue
            lk = str(level)
            levels.setdefault(lk, {"correct": 0, "total": 0, "ious": []})
            levels[lk]["total"] += 1
            if obj.get("correct"):
                levels[lk]["correct"] += 1
            iou_val = obj.get("iou")
            if iou_val is None:
                st = obj.get("bbox_stats") or {}
                iou_val = st.get("iou")
            if iou_val is not None:
                levels[lk]["ious"].append(float(iou_val))

    if levels:
        for lk, v in levels.items():
            v["accuracy"] = v["correct"] / v["total"] if v["total"] else 0.0
            v["mean_iou"] = (sum(v["ious"]) / len(v["ious"])) if v["ious"] else None
        return {"levels": levels}

    # fallback: simple accuracy if present
    if total:
        return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}
    return None

def enrich_categories_if_missing(ds_name, result, metrics_dir, dataset_map, base_dir, option_map=None, only_option_map=False):
    if "category_l1" in result or "category_l2" in result:
        return result
    if not dataset_map or not base_dir:
        return result
    ds_cfg = dataset_map.get(ds_name)
    preds_path = os.path.join(metrics_dir, ds_name, "predictions.jsonl")
    qid_map = {}
    if ds_cfg and not only_option_map:
        dataset_path = os.path.join(base_dir, ds_cfg["path"])
        qid_map = build_qid_category_map(dataset_path)
    cat_l1, cat_l2 = compute_category_stats(preds_path, qid_map, option_map=option_map)
    if cat_l1:
        result["category_l1"] = cat_l1
    if cat_l2:
        result["category_l2"] = cat_l2
    return result

def format_summary_dict(summary, metrics_dir, dataset_map, base_dir, return_text=False):
    """Generate a human-readable evaluation summary.

    Args:
        return_text: when True, return the formatted text instead of printing
    """
    
    # summary is already a dict
    fov_option_map = build_option_category_map_hardcoded()
    cvusa_option_map = build_option_category_map_hardcoded()
    
    output_lines = []
    
    model_name = summary["model"]
    output_lines.append("\n" + "="*80)
    output_lines.append(f"Evaluation Summary - Model: {model_name}")
    output_lines.append("="*80 + "\n")
    
    for ds_name, result in summary["datasets"].items():
        use_option_map_only = ("g2s" in ds_name or "s2g" in ds_name) and ("vqa" in ds_name)
        option_map = fov_option_map if "fov" in ds_name else cvusa_option_map
        result = enrich_categories_if_missing(
            ds_name,
            result,
            metrics_dir,
            dataset_map,
            base_dir,
            option_map=option_map,
            only_option_map=use_option_map_only,
        )
        output_lines.append(f"\n【{ds_name}】")
        output_lines.append("-" * 80)
        
        if "error" in result:
            output_lines.append(f"Error: {result['error']}")
            continue

        if "category_l1" in result:
            output_lines.append(f"Overall Accuracy: {result['accuracy']*100:.1f}% ({result['correct']}/{result['total']})")

            if result.get("category_l1"):
                output_lines.append("\n  [Category L1 Accuracy]")
                for cat, metrics in sorted(result["category_l1"].items()):
                    acc = metrics["acc"] * 100
                    correct = metrics["correct"]
                    total = metrics["total"]
                    output_lines.append(f"    {cat:20s}: {acc:5.1f}% ({correct}/{total})")

            if result.get("category_l2"):
                output_lines.append("\n  [Category L2 Accuracy]")
                for cat, metrics in sorted(result["category_l2"].items()):
                    acc = metrics["acc"] * 100
                    correct = metrics["correct"]
                    total = metrics["total"]
                    output_lines.append(f"    {cat:25s}: {acc:5.1f}% ({correct}/{total})")

        elif "levels" in result:
            l3 = result["levels"].get("3")
            l4 = result["levels"].get("4")
            total = 0
            weighted = 0.0
            for lv in (l3, l4):
                if lv and lv.get("mean_iou") is not None and lv.get("total"):
                    total += lv["total"]
                    weighted += lv["mean_iou"] * lv["total"]
            overall = (weighted / total) if total else None

            if overall is not None:
                output_lines.append(f"Mean IoU (L3/L4 only): {overall:.3f}")
            else:
                output_lines.append("Mean IoU (L3/L4 only): N/A")

            if l3 and l3.get("mean_iou") is not None:
                output_lines.append(f"  L3 Mean IoU: {l3['mean_iou']:.3f} ({l3.get('total', 0)} samples)")
            else:
                output_lines.append("  L3 Mean IoU: N/A")
            if l4 and l4.get("mean_iou") is not None:
                output_lines.append(f"  L4 Mean IoU: {l4['mean_iou']:.3f} ({l4.get('total', 0)} samples)")
            else:
                output_lines.append("  L4 Mean IoU: N/A")

        else:
            if "accuracy" in result:
                output_lines.append(f"Accuracy: {result['accuracy']*100:.1f}% ({result.get('correct', 0)}/{result.get('total', 0)})")

    output_lines.append("\n" + "="*80 + "\n")

    formatted_text = "\n".join(output_lines)

    if return_text:
        return formatted_text
    else:
        print(formatted_text)
        return formatted_text

def format_summary(summary_path, metrics_dir, return_text=False):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    dataset_map, base_dir = load_eval_config_map(os.path.dirname(summary_path))
    return format_summary_dict(summary, metrics_dir, dataset_map, base_dir, return_text=return_text)

def save_results_json(summary_path, output_path):
    """Save evaluation results as a JSON file using the same structure as the console summary."""
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    dataset_map, base_dir = load_eval_config_map(os.path.dirname(summary_path))
    fov_option_map = build_option_category_map_hardcoded()
    cvusa_option_map = build_option_category_map_hardcoded()
    
    results = {
        "model": summary["model"],
        "timestamp": summary.get("timestamp", ""),
        "datasets": {}
    }
    
    for ds_name, result in summary["datasets"].items():
        use_option_map_only = ("g2s" in ds_name or "s2g" in ds_name) and ("vqa" in ds_name)
        option_map = fov_option_map if "fov" in ds_name else cvusa_option_map
        result = enrich_categories_if_missing(
            ds_name,
            result,
            os.path.dirname(summary_path),
            dataset_map,
            base_dir,
            option_map=option_map,
            only_option_map=use_option_map_only,
        )
        dataset_result = {"dataset_name": ds_name}
        
        if "error" in result:
            dataset_result["status"] = "error"
            dataset_result["error_message"] = result["error"]
        else:
            dataset_result["status"] = "success"
            
            if "category_l1" in result:
                dataset_result["type"] = "mcq_vqa"
                dataset_result["overall"] = {
                    "accuracy": f"{result['accuracy']*100:.1f}%",
                    "accuracy_value": result['accuracy'],
                    "correct": result['correct'],
                    "total": result['total']
                }
                
                if result.get("category_l1"):
                    dataset_result["category_l1"] = {}
                    for cat, metrics in sorted(result["category_l1"].items()):
                        dataset_result["category_l1"][cat] = {
                            "accuracy": f"{metrics['acc']*100:.1f}%",
                            "accuracy_value": metrics['acc'],
                            "correct": metrics["correct"],
                            "total": metrics["total"]
                        }
                
                if result.get("category_l2"):
                    dataset_result["category_l2"] = {}
                    for cat, metrics in sorted(result["category_l2"].items()):
                        dataset_result["category_l2"][cat] = {
                            "accuracy": f"{metrics['acc']*100:.1f}%",
                            "accuracy_value": metrics['acc'],
                            "correct": metrics["correct"],
                            "total": metrics["total"]
                        }

            elif "levels" in result:
                dataset_result["type"] = "bbox"
                l3 = result["levels"].get("3")
                l4 = result["levels"].get("4")

                total = 0
                weighted = 0.0
                for lv in (l3, l4):
                    if lv and lv.get("mean_iou") is not None and lv.get("total"):
                        total += lv["total"]
                        weighted += lv["mean_iou"] * lv["total"]
                overall = (weighted / total) if total else None

                dataset_result["mean_iou_l3l4"] = overall
                dataset_result["l3_mean_iou"] = l3.get("mean_iou") if l3 else None
                dataset_result["l4_mean_iou"] = l4.get("mean_iou") if l4 else None
                dataset_result["L3_total"] = l3.get("total") if l3 else 0
                dataset_result["L4_total"] = l4.get("total") if l4 else 0

            else:
                dataset_result["type"] = "other"
                if "accuracy" in result:
                    dataset_result["accuracy"] = f"{result['accuracy']*100:.1f}%"
                    dataset_result["accuracy_value"] = result['accuracy']
                    dataset_result["correct"] = result.get('correct', 0)
                    dataset_result["total"] = result.get('total', 0)
        
        results["datasets"][ds_name] = dataset_result
    
    output_file = os.path.join(os.path.dirname(output_path), "results_summary.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {output_file}")
    return results

def save_results_json_from_dict(summary, output_dir):
    dataset_map, base_dir = load_eval_config_map(output_dir)
    fov_option_map = build_option_category_map_hardcoded()
    cvusa_option_map = build_option_category_map_hardcoded()
    # Reuse save_results_json logic by writing a temporary summary file.
    tmp_path = os.path.join(output_dir, "_tmp_summary.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    save_results_json(tmp_path, tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs", help="Root directory containing model output folders.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    dataset_map, base_dir = load_eval_config_map(root)
    fov_option_map = build_option_category_map_hardcoded()
    cvusa_option_map = build_option_category_map_hardcoded()

    for model_name in os.listdir(root):
        model_dir = os.path.join(root, model_name)
        if not os.path.isdir(model_dir):
            continue

        summary = {"model": model_name, "datasets": {}}
        for ds_name in os.listdir(model_dir):
            ds_dir = os.path.join(model_dir, ds_name)
            if not os.path.isdir(ds_dir):
                continue
            option_map = fov_option_map if "fov" in ds_name else cvusa_option_map
            metrics = compute_metrics_from_predictions(ds_dir, ds_name, option_map)
            if metrics:
                summary["datasets"][ds_name] = metrics

        if not summary["datasets"]:
            continue

        print(format_summary_dict(summary, model_dir, dataset_map, base_dir))
        save_results_json_from_dict(summary, model_dir)
