import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import classification_report, confusion_matrix

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PREDICTIONS = BASE_DIR / "data" / "processed" / "test_predictions.json"
DEFAULT_LABELS_REPS = BASE_DIR / "data" / "processed" / "labels_reps.json"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "processed" / "eval_eda"
DEFAULT_MODEL = BASE_DIR / "data" / "processed" / "model.pt"
DEFAULT_DATASET = BASE_DIR / "data" / "processed" / "reps_dataset.npz"

EX_ALIASES = {
    "lat_pulldown": "latpulldown",
    "lateral_raise": "lateral_raises",
    "lateralraises": "lateral_raises",
    "deadlift": "romanian_deadlift",
    "dead_lift": "romanian_deadlift",
    "romanian deadlift": "romanian_deadlift",
    "rdl": "romanian_deadlift",
}


def canonical_exercise_name(name):
    ex = str(name).strip().lower()
    return EX_ALIASES.get(ex, ex)


def canonical_form_name(name):
    s = str(name).strip().lower()
    if s in ("correct", "1", "true", "good"):
        return "correct"
    if s in ("incorrect", "0", "false", "bad"):
        return "incorrect"
    return ""


def normalize_key(path_like):
    return str(path_like).replace("\\", "/").strip().lower()


def basename_key(path_like):
    return Path(str(path_like)).name.strip().lower()


def load_json(path):
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_predictions(path):
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Predictions must be a JSON list: {path}")
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        row = dict(r)
        row["video_path"] = str(row.get("video_path", ""))
        row["exercise_pred"] = canonical_exercise_name(row.get("exercise_pred", ""))
        row["form_pred"] = canonical_form_name(row.get("form_pred", ""))
        out.append(row)
    return out


def derive_video_form_from_reps(reps):
    if not isinstance(reps, list) or len(reps) == 0:
        return ""
    for rep in reps:
        errs = (rep or {}).get("errors") or {}
        if not isinstance(errs, dict):
            continue
        for v in errs.values():
            try:
                if int(v) == 1:
                    return "incorrect"
            except Exception:
                continue
    return "correct"


def load_labels_reps_maps(path):
    data = load_json(path)
    if not isinstance(data, dict):
        return {}, {}, set()

    exact_map = {}
    basename_items = defaultdict(list)

    for video_path, entry in data.items():
        if not isinstance(entry, dict):
            continue
        ex = canonical_exercise_name(entry.get("exercise", ""))
        if not ex:
            continue
        form = derive_video_form_from_reps(entry.get("reps", []))
        key_full = normalize_key(video_path)
        key_base = basename_key(video_path)

        exact_map[key_full] = {"exercise": ex, "form": form}
        basename_items[key_base].append(
            {"exercise": ex, "form": form, "path": video_path}
        )

    basename_unique = {}
    basename_ambiguous = set()
    for key, items in basename_items.items():
        ex_set = sorted(set(i["exercise"] for i in items if i["exercise"]))
        form_set = sorted(set(i["form"] for i in items if i["form"]))
        if len(ex_set) != 1:
            basename_ambiguous.add(key)
            continue
        form = form_set[0] if len(form_set) == 1 else ""
        basename_unique[key] = {"exercise": ex_set[0], "form": form}

    return exact_map, basename_unique, basename_ambiguous


def load_gt_map(path):
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}

    if p.suffix.lower() == ".json":
        raw = load_json(p)
        if isinstance(raw, dict):
            out = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    out[normalize_key(k)] = {
                        "exercise": canonical_exercise_name(v.get("exercise", "")),
                        "form": canonical_form_name(v.get("form", "")),
                    }
                else:
                    out[normalize_key(k)] = {
                        "exercise": canonical_exercise_name(v),
                        "form": "",
                    }
            return out
        if isinstance(raw, list):
            out = {}
            for item in raw:
                if not isinstance(item, dict):
                    continue
                key = normalize_key(item.get("video_path", ""))
                if not key:
                    continue
                out[key] = {
                    "exercise": canonical_exercise_name(item.get("exercise", "")),
                    "form": canonical_form_name(item.get("form", "")),
                }
            return out
        return {}

    if p.suffix.lower() == ".csv":
        out = {}
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = normalize_key(row.get("video_path", ""))
                if not key:
                    continue
                out[key] = {
                    "exercise": canonical_exercise_name(row.get("exercise", "")),
                    "form": canonical_form_name(row.get("form", "")),
                }
        return out

    return {}


def sync_gt_template(predictions, template_path):
    p = Path(template_path)
    existing = load_json(p)
    if not isinstance(existing, dict):
        existing = {}

    out = {}
    changed = False

    # Keep rows for all current prediction keys, preserving existing labels.
    for r in predictions:
        key = str(r.get("video_path", ""))
        row = existing.get(key, {})
        if not isinstance(row, dict):
            row = {}
        new_row = {
            "exercise": canonical_exercise_name(row.get("exercise", "")),
            "form": canonical_form_name(row.get("form", "")),
        }
        out[key] = new_row
        if key not in existing or existing.get(key) != new_row:
            changed = True

    # Preserve extra keys that were already in template.
    for key, row in existing.items():
        if key in out:
            continue
        if not isinstance(row, dict):
            continue
        out[key] = {
            "exercise": canonical_exercise_name(row.get("exercise", "")),
            "form": canonical_form_name(row.get("form", "")),
        }
        if out[key] != row:
            changed = True

    if changed or not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    return out, changed


def conf_stats(values):
    if not values:
        return {}
    arr = np.array(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def extract_report_f1(report, labels=None):
    if not isinstance(report, dict):
        return {}

    out = {}
    macro = report.get("macro avg")
    weighted = report.get("weighted avg")
    if isinstance(macro, dict) and "f1-score" in macro:
        out["f1_macro"] = float(macro.get("f1-score", 0.0))
    if isinstance(weighted, dict) and "f1-score" in weighted:
        out["f1_weighted"] = float(weighted.get("f1-score", 0.0))

    if labels:
        per_label = {}
        for lbl in labels:
            row = report.get(lbl)
            if isinstance(row, dict) and "f1-score" in row:
                per_label[lbl] = float(row.get("f1-score", 0.0))
        if per_label:
            out["f1_by_class"] = per_label
    return out


def load_model_summary(model_path):
    p = Path(model_path)
    if not p.is_file():
        return {}
    try:
        import torch
    except Exception:
        return {"warning": "torch is not available"}

    try:
        ckpt = torch.load(p, map_location="cpu")
    except Exception as e:
        return {"warning": f"failed to load model: {e}"}

    ex_map = ckpt.get("ex_map", {}) if isinstance(ckpt.get("ex_map", {}), dict) else {}
    err_map = (
        ckpt.get("err_map", {}) if isinstance(ckpt.get("err_map", {}), dict) else {}
    )
    state_dict = ckpt.get("model", {})
    total_params = 0
    if isinstance(state_dict, dict):
        for v in state_dict.values():
            if hasattr(v, "numel"):
                total_params += int(v.numel())

    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    return {
        "path": str(p.resolve()),
        "arch": str(cfg.get("arch", "unknown")),
        "config": cfg,
        "F": int(ckpt.get("F", -1)),
        "T": (
            int(ckpt.get("T", -1))
            if not hasattr(ckpt.get("T", -1), "__len__")
            else int(ckpt.get("T")[0])
        ),
        "n_ex": int(ckpt.get("n_ex", len(ex_map))),
        "n_err": int(ckpt.get("n_err", len(err_map))),
        "exercise_classes": sorted(list(ex_map.keys())),
        "error_classes_count": len(err_map),
        "total_params": int(total_params),
    }


def load_dataset_summary(dataset_path, topn_errors):
    p = Path(dataset_path)
    if not p.is_file():
        return {}

    try:
        z = np.load(p, allow_pickle=True)
    except Exception as e:
        return {"warning": f"failed to load dataset: {e}"}

    out = {
        "path": str(p.resolve()),
        "keys": sorted(list(z.files)),
    }

    if "X" in z.files:
        x = z["X"]
        out["X_shape"] = [int(d) for d in x.shape]
    if "T" in z.files:
        out["T"] = int(z["T"][0])

    ex_map = z["ex_map"].item() if "ex_map" in z.files else {}
    err_map = z["err_map"].item() if "err_map" in z.files else {}
    y_ex = z["y_ex"] if "y_ex" in z.files else None
    y_form = z["y_form"] if "y_form" in z.files else None
    y_err = z["y_err"] if "y_err" in z.files else None

    if isinstance(ex_map, dict):
        out["exercise_classes"] = sorted(list(ex_map.keys()))
    if y_ex is not None and isinstance(ex_map, dict):
        ex_counts = {k: int((y_ex == v).sum()) for k, v in ex_map.items()}
        out["exercise_counts"] = dict(sorted(ex_counts.items()))
        vals = list(ex_counts.values())
        if vals and min(vals) > 0:
            out["exercise_imbalance_max_to_min"] = float(max(vals) / min(vals))

    if y_form is not None:
        y_form = np.array(y_form).astype(np.int64)
        out["form_counts"] = {
            "incorrect": int((y_form == 0).sum()),
            "correct": int((y_form == 1).sum()),
        }

    if y_err is not None and isinstance(err_map, dict):
        err_counts = {k: int(y_err[:, v].sum()) for k, v in err_map.items()}
        out["error_positive_counts_top"] = dict(
            sorted(err_counts.items(), key=lambda kv: kv[1], reverse=True)[
                : int(topn_errors)
            ]
        )

    return out


def write_csv(path, fieldnames, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_confusion_csv(path, labels, cm):
    rows = []
    for i, true_label in enumerate(labels):
        row = {"true\\pred": true_label}
        for j, pred_label in enumerate(labels):
            row[pred_label] = int(cm[i, j])
        rows.append(row)
    fieldnames = ["true\\pred"] + list(labels)
    write_csv(path, fieldnames, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    p.add_argument("--labels_reps", default=str(DEFAULT_LABELS_REPS))
    p.add_argument(
        "--gt_map",
        default="",
        help="Optional JSON/CSV mapping with columns: video_path, exercise, form",
    )
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--topn_errors", type=int, default=10)
    args = p.parse_args()

    predictions = load_predictions(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_exact, labels_base_unique, labels_base_ambiguous = load_labels_reps_maps(
        args.labels_reps
    )
    template_path = out_dir / "gt_map_template.json"
    template_map, template_changed = sync_gt_template(predictions, template_path)

    gt_map_source = ""
    if str(args.gt_map).strip():
        gt_map = load_gt_map(args.gt_map)
        gt_map_source = (
            str(Path(args.gt_map).resolve())
            if Path(args.gt_map).exists()
            else str(args.gt_map)
        )
    else:
        gt_map = load_gt_map(template_path)
        if gt_map:
            gt_map_source = str(template_path.resolve())

    gt_labeled_exercise = 0
    gt_labeled_form = 0
    for row in gt_map.values():
        if not isinstance(row, dict):
            continue
        if canonical_exercise_name(row.get("exercise", "")):
            gt_labeled_exercise += 1
        if canonical_form_name(row.get("form", "")):
            gt_labeled_form += 1

    status_counts = Counter()
    pred_ex_counts = Counter()
    pred_form_counts = Counter()
    ex_conf_by_class = defaultdict(list)
    form_conf_by_class = defaultdict(list)
    top1_error_counts = Counter()
    over_threshold_error_counts = Counter()
    source_counts = Counter()
    linked_mismatch_counts = Counter()

    eval_rows = []
    ex_pairs = []
    form_pairs = []
    match_source_counts = Counter()
    unmatched = 0
    ambiguous = 0

    for r in predictions:
        status = str(r.get("status", ""))
        status_counts[status] += 1

        pred_ex = canonical_exercise_name(r.get("exercise_pred", ""))
        pred_form = canonical_form_name(r.get("form_pred", ""))
        ex_conf = float(r.get("exercise_conf", 0.0) or 0.0)
        form_conf = float(r.get("form_conf", 0.0) or 0.0)
        video_path = str(r.get("video_path", ""))
        key_full = normalize_key(video_path)
        key_base = basename_key(video_path)

        if pred_ex:
            pred_ex_counts[pred_ex] += 1
            ex_conf_by_class[pred_ex].append(ex_conf)
        if pred_form:
            pred_form_counts[pred_form] += 1
            form_conf_by_class[pred_form].append(form_conf)

        source_counts[str(r.get("exercise_source", "unknown"))] += 1

        top_errors = r.get("top_errors", []) or []
        over_thr = r.get("errors_over_threshold", []) or []
        if top_errors:
            top1 = str((top_errors[0] or {}).get("name", "")).strip()
            if top1:
                top1_error_counts[top1] += 1
        for e in over_thr:
            name = str((e or {}).get("name", "")).strip()
            if name:
                over_threshold_error_counts[name] += 1

        # Diagnostic when form is linked to errors.
        if str(r.get("form_source", "")) == "errors_threshold":
            has_over_thr = len(over_thr) > 0
            if pred_form == "incorrect" and not has_over_thr:
                linked_mismatch_counts["incorrect_but_no_over_threshold"] += 1
            if pred_form == "correct" and has_over_thr:
                linked_mismatch_counts["correct_but_has_over_threshold"] += 1

        gt = None
        if key_full in gt_map:
            gt = gt_map[key_full]
            match_source_counts["gt_map_full"] += 1
        elif key_base in gt_map:
            gt = gt_map[key_base]
            match_source_counts["gt_map_basename"] += 1
        elif key_full in labels_exact:
            gt = labels_exact[key_full]
            match_source_counts["labels_full"] += 1
        elif key_base in labels_base_unique:
            gt = labels_base_unique[key_base]
            match_source_counts["labels_basename_unique"] += 1
        elif key_base in labels_base_ambiguous:
            ambiguous += 1
            match_source_counts["labels_basename_ambiguous"] += 1
        else:
            unmatched += 1

        gt_ex = canonical_exercise_name((gt or {}).get("exercise", "")) if gt else ""
        gt_form = canonical_form_name((gt or {}).get("form", "")) if gt else ""

        if gt_ex:
            ex_pairs.append((gt_ex, pred_ex))
        if gt_form:
            form_pairs.append((gt_form, pred_form))

        eval_rows.append(
            {
                "video_path": video_path,
                "status": status,
                "exercise_pred": pred_ex,
                "exercise_conf": ex_conf,
                "form_pred": pred_form,
                "form_conf": form_conf,
                "exercise_source": r.get("exercise_source", ""),
                "form_source": r.get("form_source", ""),
                "gt_exercise": gt_ex,
                "gt_form": gt_form,
                "top1_error": top_errors[0]["name"] if top_errors else "",
                "top1_error_prob": float(top_errors[0]["prob"]) if top_errors else "",
                "errors_over_threshold": "|".join(
                    str(e.get("name", "")) for e in over_thr
                ),
            }
        )

    summary = {
        "predictions_file": str(Path(args.predictions).resolve()),
        "gt_map_source": gt_map_source,
        "gt_map_entries": int(len(gt_map)),
        "gt_labeled_exercise_count": int(gt_labeled_exercise),
        "gt_labeled_form_count": int(gt_labeled_form),
        "gt_template_path": str(template_path.resolve()),
        "gt_template_updated": bool(template_changed),
        "n_rows": len(predictions),
        "status_counts": dict(sorted(status_counts.items())),
        "exercise_distribution": dict(sorted(pred_ex_counts.items())),
        "form_distribution": dict(sorted(pred_form_counts.items())),
        "exercise_source_counts": dict(sorted(source_counts.items())),
        "linked_form_diagnostics": dict(sorted(linked_mismatch_counts.items())),
        "top1_error_counts": dict(top1_error_counts.most_common(int(args.topn_errors))),
        "errors_over_threshold_counts": dict(
            over_threshold_error_counts.most_common(int(args.topn_errors))
        ),
        "exercise_conf_stats_overall": conf_stats(
            [float(r.get("exercise_conf", 0.0) or 0.0) for r in predictions]
        ),
        "form_conf_stats_overall": conf_stats(
            [float(r.get("form_conf", 0.0) or 0.0) for r in predictions]
        ),
        "exercise_conf_stats_by_pred_class": {
            k: conf_stats(v) for k, v in sorted(ex_conf_by_class.items())
        },
        "form_conf_stats_by_pred_class": {
            k: conf_stats(v) for k, v in sorted(form_conf_by_class.items())
        },
        "gt_match_sources": dict(sorted(match_source_counts.items())),
        "gt_unmatched_count": int(unmatched),
        "gt_ambiguous_basename_count": int(ambiguous),
        "model_summary": load_model_summary(args.model),
        "dataset_summary": load_dataset_summary(
            args.dataset, topn_errors=int(args.topn_errors)
        ),
    }

    # Exercise metrics with GT
    if ex_pairs:
        y_true_ex = [t for t, _ in ex_pairs]
        y_pred_ex = [p for _, p in ex_pairs]
        labels_ex = sorted(set(y_true_ex) | set(y_pred_ex))
        ex_acc = float(sum(int(t == p) for t, p in ex_pairs) / len(ex_pairs))
        summary["exercise_eval"] = {
            "n_matched": len(ex_pairs),
            "accuracy": ex_acc,
        }

        confusion_rows = []
        if HAS_SKLEARN:
            cm_ex = confusion_matrix(y_true_ex, y_pred_ex, labels=labels_ex)
            write_confusion_csv(out_dir / "confusion_exercise.csv", labels_ex, cm_ex)
            report_ex = classification_report(
                y_true_ex,
                y_pred_ex,
                labels=labels_ex,
                output_dict=True,
                zero_division=0,
            )
            summary["exercise_eval"]["classification_report"] = report_ex
            summary["exercise_eval"].update(
                extract_report_f1(report_ex, labels=labels_ex)
            )
        else:
            for t, p_ in ex_pairs:
                confusion_rows.append((t, p_))

        mismatch_counts = Counter((t, p_) for t, p_ in ex_pairs if t != p_)
        top_conf = []
        for (t, p_), c in mismatch_counts.most_common(20):
            top_conf.append({"true": t, "pred": p_, "count": int(c)})
        summary["exercise_eval"]["top_confusions"] = top_conf

    # Form metrics with GT
    if form_pairs:
        y_true_f = [t for t, _ in form_pairs]
        y_pred_f = [p_ for _, p_ in form_pairs]
        labels_f = ["incorrect", "correct"]
        form_acc = float(sum(int(t == p_) for t, p_ in form_pairs) / len(form_pairs))
        summary["form_eval"] = {
            "n_matched": len(form_pairs),
            "accuracy": form_acc,
        }
        if HAS_SKLEARN:
            cm_form = confusion_matrix(y_true_f, y_pred_f, labels=labels_f)
            write_confusion_csv(out_dir / "confusion_form.csv", labels_f, cm_form)
            report_f = classification_report(
                y_true_f, y_pred_f, labels=labels_f, output_dict=True, zero_division=0
            )
            summary["form_eval"]["classification_report"] = report_f
            summary["form_eval"].update(extract_report_f1(report_f, labels=labels_f))

    # Save outputs
    with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    eval_fieldnames = [
        "video_path",
        "status",
        "exercise_pred",
        "exercise_conf",
        "form_pred",
        "form_conf",
        "exercise_source",
        "form_source",
        "gt_exercise",
        "gt_form",
        "top1_error",
        "top1_error_prob",
        "errors_over_threshold",
    ]
    write_csv(out_dir / "eval_rows.csv", eval_fieldnames, eval_rows)

    print("Saved summary ->", out_dir / "eda_summary.json")
    print("Saved rows    ->", out_dir / "eval_rows.csv")
    if template_path.is_file():
        print("Saved GT tpl ->", template_path)
    if (out_dir / "confusion_exercise.csv").is_file():
        print("Saved conf ex ->", out_dir / "confusion_exercise.csv")
    if (out_dir / "confusion_form.csv").is_file():
        print("Saved conf fm ->", out_dir / "confusion_form.csv")

    print("")
    print("N predictions:", len(predictions))
    print("Status:", dict(sorted(status_counts.items())))
    print("Exercise distribution:", dict(sorted(pred_ex_counts.items())))
    print("Form distribution:", dict(sorted(pred_form_counts.items())))
    print("GT labeled exercise:", int(gt_labeled_exercise))
    print("GT labeled form:", int(gt_labeled_form))
    print("GT matched (exercise):", len(ex_pairs))
    if gt_labeled_exercise == 0:
        print(
            "No GT exercise labels filled -> confusion_exercise.csv will not be created."
        )
    if "exercise_eval" in summary:
        print("Exercise accuracy:", f"{summary['exercise_eval']['accuracy']:.4f}")
        if "f1_macro" in summary["exercise_eval"]:
            print("Exercise F1 macro:", f"{summary['exercise_eval']['f1_macro']:.4f}")
        if "f1_weighted" in summary["exercise_eval"]:
            print(
                "Exercise F1 weighted:",
                f"{summary['exercise_eval']['f1_weighted']:.4f}",
            )
    if "form_eval" in summary:
        print("Form accuracy:", f"{summary['form_eval']['accuracy']:.4f}")
        if "f1_macro" in summary["form_eval"]:
            print("Form F1 macro:", f"{summary['form_eval']['f1_macro']:.4f}")
        if "f1_weighted" in summary["form_eval"]:
            print("Form F1 weighted:", f"{summary['form_eval']['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
