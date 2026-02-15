import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import cv2

from extract_pose import extract_landmarks_from_video
from pose_normalize import normalize_sequence
from pose_smooth import smooth_sequence
from infer import MultiTaskGRU, build_model_features, sigmoid_np, softmax_np

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
LOWER_BODY_CLASSES = {"squat", "romanian_deadlift"}
DEFAULT_TAXONOMY = BASE_DIR / "data" / "processed" / "error_taxonomy.json"

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


def load_allowed_errors_by_exercise(taxonomy_path):
    p = Path(taxonomy_path)
    if not p.is_file():
        return {}

    try:
        with open(p, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
    except Exception:
        return {}

    out = {}
    for ex_key, data in (taxonomy or {}).items():
        ex = canonical_exercise_name(ex_key)
        errs = (data or {}).get("errors") or {}
        if not isinstance(errs, dict):
            continue
        out.setdefault(ex, set()).update(str(k) for k in errs.keys())
    return out


def merge_exercise_probs(id_to_ex, ex_probs):
    merged = {}
    for i in range(len(ex_probs)):
        raw_name = id_to_ex.get(i, f"unknown_{i}")
        if str(raw_name).startswith("unknown_"):
            ex_name = str(raw_name)
        else:
            ex_name = canonical_exercise_name(raw_name)
        merged[ex_name] = merged.get(ex_name, 0.0) + float(ex_probs[i])
    return merged


def discover_videos(root_dir):
    root = Path(root_dir)
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    return sorted(out)


def normalize_video_key(key):
    return str(key).replace("\\", "/").strip().lower()


def load_exercise_overrides(path):
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            raw = json.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    out = {}
    for k, v in raw.items():
        ex = canonical_exercise_name(v)
        if ex:
            out[normalize_video_key(k)] = ex
    return out


def load_error_threshold_map(path):
    p = Path(path)
    if not path or not p.is_file():
        return {
            "global": None,
            "errors": {},
            "exercise_errors": {},
        }

    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return {
            "global": None,
            "errors": {},
            "exercise_errors": {},
        }

    if not isinstance(raw, dict):
        return {
            "global": None,
            "errors": {},
            "exercise_errors": {},
        }

    out = {
        "global": None,
        "errors": {},
        "exercise_errors": {},
    }

    g = raw.get("global")
    if isinstance(g, (int, float)):
        out["global"] = float(g)

    errs = raw.get("errors", {})
    if isinstance(errs, dict):
        for k, v in errs.items():
            if isinstance(v, (int, float)):
                out["errors"][str(k)] = float(v)

    ex_errs = raw.get("exercise_errors", {})
    if isinstance(ex_errs, dict):
        for ex_name, e_map in ex_errs.items():
            ex = canonical_exercise_name(ex_name)
            if not ex or not isinstance(e_map, dict):
                continue
            out["exercise_errors"].setdefault(ex, {})
            for err_name, thr in e_map.items():
                if isinstance(thr, (int, float)):
                    out["exercise_errors"][ex][str(err_name)] = float(thr)
    return out


def get_error_threshold(ex_name, err_name, base_threshold, threshold_cfg):
    thr = float(base_threshold)
    if not isinstance(threshold_cfg, dict):
        return thr

    g = threshold_cfg.get("global")
    if isinstance(g, (int, float)):
        thr = float(g)

    errors_map = threshold_cfg.get("errors", {})
    if isinstance(errors_map, dict) and err_name in errors_map:
        v = errors_map.get(err_name)
        if isinstance(v, (int, float)):
            thr = float(v)

    ex_map = threshold_cfg.get("exercise_errors", {})
    if isinstance(ex_map, dict):
        per_ex = ex_map.get(canonical_exercise_name(ex_name), {})
        if isinstance(per_ex, dict) and err_name in per_ex:
            v = per_ex.get(err_name)
            if isinstance(v, (int, float)):
                thr = float(v)

    return thr


def resolve_effective_stride(video_path, stride, max_frames):
    stride = max(1, int(stride))
    max_frames = int(max_frames)
    if max_frames <= 0:
        return stride

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return stride
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        return stride

    need_stride = int(np.ceil(total_frames / max(1, max_frames)))
    return max(stride, need_stride)


def load_model_bundle(model_path, T_override=None, taxonomy_path=None):
    ckpt = torch.load(model_path, map_location="cpu")

    ex_map = ckpt.get("ex_map", None)
    if not isinstance(ex_map, dict) or len(ex_map) == 0:
        raise ValueError("Model checkpoint does not contain valid ex_map.")
    id_to_ex = {v: k for k, v in ex_map.items()}
    n_ex = int(ckpt.get("n_ex", len(ex_map)))
    if n_ex != len(ex_map):
        raise ValueError(f"Model mismatch: n_ex={n_ex}, len(ex_map)={len(ex_map)}.")

    err_map = ckpt.get("err_map", None)
    if isinstance(err_map, dict):
        id_to_err = {v: k for k, v in err_map.items()}
        n_err = int(ckpt.get("n_err", len(err_map)))
    else:
        id_to_err = None
        n_err = int(ckpt.get("n_err", 0))

    F = int(ckpt.get("F", 8))
    T_ckpt = ckpt.get("T", 120)
    T_default = (
        int(T_ckpt[0])
        if hasattr(T_ckpt, "__len__") and not isinstance(T_ckpt, (str, bytes))
        else int(T_ckpt)
    )
    T = int(T_override) if T_override is not None else int(T_default)

    cfg = ckpt.get("config", {})
    model = MultiTaskGRU(
        F=F,
        n_ex=n_ex,
        n_err=n_err,
        hidden=int(cfg.get("hidden", 128)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        bidir=bool(cfg.get("bidir", True)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    allowed_errors_by_exercise = load_allowed_errors_by_exercise(
        DEFAULT_TAXONOMY if taxonomy_path is None else taxonomy_path
    )

    return {
        "model": model,
        "F": F,
        "T": T,
        "n_err": n_err,
        "id_to_ex": id_to_ex,
        "id_to_err": id_to_err,
        "allowed_errors_by_exercise": allowed_errors_by_exercise,
    }


def infer_one_video(
    video_path,
    bundle,
    stride,
    max_frames,
    topk_errors,
    error_threshold,
    form_error_threshold,
    error_threshold_cfg,
    lower_body_rescore,
    lb_uniform_alpha,
    form_mode,
    exercise_override=None,
):
    base_stride = max(1, int(stride))
    seq = extract_landmarks_from_video(
        str(video_path), max_frames=max_frames, stride=base_stride
    )
    seq = smooth_sequence(seq, vis_thr=0.35, max_gap=8, ema_alpha=0.25)
    seq = normalize_sequence(seq, rotate=True)

    feats = build_model_features(
        seq, T=bundle["T"], F_expected=bundle["F"], sampling="head"
    )
    if feats.shape[-1] != bundle["F"]:
        raise ValueError(
            f"Feature dim mismatch: expected F={bundle['F']}, got {feats.shape[-1]}."
        )

    X = torch.tensor(feats[None, ...], dtype=torch.float32)

    with torch.no_grad():
        ex_logits_head, form_logits, err_logits = bundle["model"](X)
        ex_logits_head = ex_logits_head.numpy().reshape(-1)
        form_logits = form_logits.numpy().reshape(-1)
        err_logits = err_logits.numpy().reshape(-1) if err_logits is not None else None

    ex_probs_head = softmax_np(ex_logits_head)
    ex_probs = ex_probs_head
    if bool(lower_body_rescore) and int(seq.shape[0]) > int(bundle["T"]):
        head_ex_id = int(ex_probs_head.argmax())
        head_name_raw = bundle["id_to_ex"].get(head_ex_id, f"unknown_{head_ex_id}")
        if str(head_name_raw).startswith("unknown_"):
            head_name = str(head_name_raw)
        else:
            head_name = canonical_exercise_name(head_name_raw)
        if head_name in LOWER_BODY_CLASSES:
            seq_rescore = seq
            eff_stride = resolve_effective_stride(
                video_path, stride=base_stride, max_frames=max_frames
            )
            if eff_stride != base_stride:
                seq_rescore = extract_landmarks_from_video(
                    str(video_path), max_frames=max_frames, stride=eff_stride
                )
                seq_rescore = smooth_sequence(
                    seq_rescore, vis_thr=0.35, max_gap=8, ema_alpha=0.25
                )
                seq_rescore = normalize_sequence(seq_rescore, rotate=True)

            feats_uniform = build_model_features(
                seq_rescore, T=bundle["T"], F_expected=bundle["F"], sampling="uniform"
            )
            X_uniform = torch.tensor(feats_uniform[None, ...], dtype=torch.float32)
            with torch.no_grad():
                ex_logits_uniform, _form_u, _err_u = bundle["model"](X_uniform)
                ex_logits_uniform = ex_logits_uniform.numpy().reshape(-1)
            ex_probs_uniform = softmax_np(ex_logits_uniform)
            alpha = float(np.clip(lb_uniform_alpha, 0.0, 1.0))
            ex_probs = (1.0 - alpha) * ex_probs_head + alpha * ex_probs_uniform
            ex_probs = ex_probs / (np.sum(ex_probs) + 1e-9)

    form_probs = softmax_np(form_logits)

    ex_id = int(ex_probs.argmax())
    form_id = int(form_probs.argmax())
    raw_ex_name = bundle["id_to_ex"].get(ex_id, f"unknown_{ex_id}")
    if str(raw_ex_name).startswith("unknown_"):
        ex_name = str(raw_ex_name)
    else:
        ex_name = canonical_exercise_name(raw_ex_name)
    model_form_name = "correct" if form_id == 1 else "incorrect"
    model_form_conf = float(form_probs[form_id])
    model_form_probs = {
        "incorrect": float(form_probs[0]),
        "correct": float(form_probs[1]),
    }

    ex_probs_named = merge_exercise_probs(bundle["id_to_ex"], ex_probs)
    exercise_conf = float(ex_probs_named.get(ex_name, ex_probs[ex_id]))
    model_ex_name = ex_name
    model_ex_conf = exercise_conf
    exercise_source = "model_head"

    if exercise_override is not None:
        ex_override = canonical_exercise_name(exercise_override)
        if ex_override in ex_probs_named:
            ex_name = ex_override
            exercise_conf = float(ex_probs_named.get(ex_name, 0.0))
            exercise_source = "override"

    out = {
        "status": "ok",
        "exercise_pred": ex_name,
        "exercise_conf": exercise_conf,
        "exercise_pred_model": model_ex_name,
        "exercise_conf_model": model_ex_conf,
        "exercise_source": exercise_source,
        "exercise_probs": ex_probs_named,
        "form_pred": model_form_name,
        "form_conf": model_form_conf,
        "form_probs": model_form_probs,
        "form_pred_model": model_form_name,
        "form_conf_model": model_form_conf,
        "form_probs_model": model_form_probs,
        "form_source": "model_head",
        "top_errors": [],
        "errors_over_threshold": [],
    }

    if err_logits is not None and bundle["id_to_err"] is not None:
        err_probs = sigmoid_np(err_logits)
        allowed_map = bundle.get("allowed_errors_by_exercise", {})
        allowed_errors = None
        if ex_name in allowed_map:
            allowed_errors = set(allowed_map[ex_name])

        candidates = []
        for i in range(len(err_probs)):
            i_int = int(i)
            name = bundle["id_to_err"].get(i_int, str(i_int))
            if allowed_errors is not None and name not in allowed_errors:
                continue
            candidates.append((name, float(err_probs[i_int])))

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        topk = int(min(topk_errors, len(candidates)))

        top_errors = []
        for name, prob in candidates[:topk]:
            top_errors.append(
                {
                    "name": name,
                    "prob": prob,
                }
            )
        out["top_errors"] = top_errors

        over_thr = []
        for name, p in candidates:
            thr_err = get_error_threshold(
                ex_name=ex_name,
                err_name=name,
                base_threshold=float(error_threshold),
                threshold_cfg=error_threshold_cfg,
            )
            if p >= float(thr_err):
                over_thr.append(
                    {
                        "name": name,
                        "prob": p,
                    }
                )
        out["errors_over_threshold"] = over_thr

        if str(form_mode) == "errors_threshold":
            # Link form to current-exercise errors:
            # incorrect if at least one allowed error exceeds threshold.
            max_err_prob = float(candidates[0][1]) if candidates else 0.0
            linked_incorrect_prob = max_err_prob
            linked_correct_prob = float(max(0.0, 1.0 - linked_incorrect_prob))
            thr_form = (
                float(error_threshold)
                if form_error_threshold is None
                else float(form_error_threshold)
            )
            linked_form_name = (
                "incorrect" if linked_incorrect_prob >= thr_form else "correct"
            )
            linked_form_conf = (
                linked_incorrect_prob
                if linked_form_name == "incorrect"
                else linked_correct_prob
            )

            out["form_pred"] = linked_form_name
            out["form_conf"] = float(linked_form_conf)
            out["form_probs"] = {
                "incorrect": float(linked_incorrect_prob),
                "correct": float(linked_correct_prob),
            }
            out["form_source"] = "errors_threshold"

    return out


def write_csv(rows, out_csv, topk_errors):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_path",
        "status",
        "error_message",
        "exercise_pred",
        "exercise_conf",
        "form_pred",
        "form_conf",
        "errors_over_threshold",
    ]
    for i in range(1, int(topk_errors) + 1):
        fieldnames.extend([f"top_error_{i}", f"top_error_{i}_prob"])

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            row = {
                "video_path": r.get("video_path", ""),
                "status": r.get("status", ""),
                "error_message": r.get("error_message", ""),
                "exercise_pred": r.get("exercise_pred", ""),
                "exercise_conf": r.get("exercise_conf", ""),
                "form_pred": r.get("form_pred", ""),
                "form_conf": r.get("form_conf", ""),
                "errors_over_threshold": "|".join(
                    [e["name"] for e in r.get("errors_over_threshold", [])]
                ),
            }

            top = r.get("top_errors", [])
            for i in range(1, int(topk_errors) + 1):
                idx = i - 1
                if idx < len(top):
                    row[f"top_error_{i}"] = top[idx]["name"]
                    row[f"top_error_{i}_prob"] = top[idx]["prob"]
                else:
                    row[f"top_error_{i}"] = ""
                    row[f"top_error_{i}_prob"] = ""
            writer.writerow(row)


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") != "ok"]

    print(f"Total videos: {len(rows)}")
    print(f"Successful: {len(ok)}")
    print(f"Failed: {len(bad)}")

    if not ok:
        return

    ex_counts = {}
    form_counts = {}
    ex_conf = []
    form_conf = []

    for r in ok:
        ex = r.get("exercise_pred")
        fm = r.get("form_pred")
        ex_counts[ex] = ex_counts.get(ex, 0) + 1
        form_counts[fm] = form_counts.get(fm, 0) + 1
        ex_conf.append(float(r.get("exercise_conf", 0.0)))
        form_conf.append(float(r.get("form_conf", 0.0)))

    print("Exercise distribution:", dict(sorted(ex_counts.items())))
    print("Form distribution:", dict(sorted(form_counts.items())))
    print(f"Mean exercise confidence: {float(np.mean(ex_conf)):.3f}")
    print(f"Mean form confidence: {float(np.mean(form_conf)):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=str(BASE_DIR / "data" / "test"))
    p.add_argument("--model", default=str(BASE_DIR / "data" / "processed" / "model.pt"))
    p.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    p.add_argument("--exercise_overrides", default="")
    p.add_argument(
        "--out_json",
        default=str(BASE_DIR / "data" / "processed" / "test_predictions.json"),
    )
    p.add_argument(
        "--out_csv",
        default=str(BASE_DIR / "data" / "processed" / "test_predictions.csv"),
    )
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=180)
    p.add_argument("--topk_errors", type=int, default=5)
    p.add_argument("--error_threshold", type=float, default=0.5)
    p.add_argument(
        "--error_threshold_map",
        default="",
        help=(
            "Optional JSON with per-error thresholds. "
            "Supports keys: global, errors, exercise_errors."
        ),
    )
    p.add_argument(
        "--form_error_threshold",
        type=float,
        default=0.2,
        help=("Threshold only for form in errors_threshold mode " "(default: 0.2)."),
    )
    p.add_argument(
        "--form_mode",
        choices=["errors_threshold", "model_head"],
        default="errors_threshold",
        help="How to produce form_pred. Errors are always output in top_errors/errors_over_threshold.",
    )
    p.add_argument(
        "--lower_body_rescore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, rescore squat/deadlift classes using uniform temporal sampling.",
    )
    p.add_argument(
        "--lb_uniform_alpha",
        type=float,
        default=0.7,
        help="Blend weight for uniform sampling in lower-body rescore (0..1).",
    )
    args = p.parse_args()

    videos = discover_videos(args.input_dir)
    if not videos:
        print(f"No videos found in: {args.input_dir}")
        return

    bundle = load_model_bundle(
        args.model, T_override=args.T, taxonomy_path=args.taxonomy
    )
    threshold_cfg = load_error_threshold_map(args.error_threshold_map)
    overrides = load_exercise_overrides(
        args.exercise_overrides if str(args.exercise_overrides).strip() else None
    )
    root_dir = Path(args.input_dir).resolve()

    rows = []
    for video_path in videos:
        rel = video_path.resolve().relative_to(root_dir).as_posix()
        rel_key = normalize_video_key(rel)
        base_key = normalize_video_key(video_path.name)
        exercise_override = overrides.get(rel_key, overrides.get(base_key))
        print(f"[infer] {rel}")
        try:
            pred = infer_one_video(
                video_path=video_path,
                bundle=bundle,
                stride=int(args.stride),
                max_frames=int(args.max_frames),
                topk_errors=int(args.topk_errors),
                error_threshold=float(args.error_threshold),
                form_error_threshold=float(args.form_error_threshold),
                error_threshold_cfg=threshold_cfg,
                lower_body_rescore=bool(args.lower_body_rescore),
                lb_uniform_alpha=float(args.lb_uniform_alpha),
                form_mode=str(args.form_mode),
                exercise_override=exercise_override,
            )
            pred["video_path"] = rel
            rows.append(pred)
        except Exception as e:
            rows.append(
                {
                    "video_path": rel,
                    "status": "error",
                    "error_message": str(e),
                    "exercise_pred": "",
                    "exercise_conf": "",
                    "form_pred": "",
                    "form_conf": "",
                    "top_errors": [],
                    "errors_over_threshold": [],
                }
            )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    write_csv(rows, args.out_csv, args.topk_errors)

    print(f"\nSaved JSON -> {out_json}")
    print(f"Saved CSV  -> {args.out_csv}")
    print("")
    summarize(rows)


if __name__ == "__main__":
    main()
