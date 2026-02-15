import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path

from extract_pose import extract_landmarks_from_video
from pose_normalize import normalize_sequence
from pose_smooth import smooth_sequence
from features import landmarks_to_features, pad_or_trim, nan_to_num

BASE_DIR = Path(__file__).resolve().parent.parent
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


class MultiTaskGRU(nn.Module):
    def __init__(self, F, n_ex, n_err=0, hidden=128, num_layers=2, dropout=0.2, bidir=True):
        super().__init__()
        self.n_err = int(n_err)

        self.gru = nn.GRU(
            input_size=F,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidir,
        )
        out_dim = hidden * (2 if bidir else 1)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        self.ex_head = nn.Linear(out_dim, n_ex)
        self.form_head = nn.Linear(out_dim, 2)

        self.err_head = None
        if self.n_err > 0:
            self.err_head = nn.Linear(out_dim, self.n_err)

    def forward(self, x):
        h, _ = self.gru(x)
        emb = h.mean(dim=1)
        emb = self.dropout(self.norm(emb))
        ex_logits = self.ex_head(emb)
        form_logits = self.form_head(emb)
        err_logits = self.err_head(emb) if self.err_head is not None else None
        return ex_logits, form_logits, err_logits


def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


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


def temporal_resample_uniform(x, T):
    # Uniformly sample full sequence to preserve long-video motion coverage.
    t = x.shape[0]
    if t <= 0:
        return np.zeros((int(T), x.shape[1]), dtype=np.float32)
    if t == int(T):
        return x
    if t < int(T):
        out = np.zeros((int(T), x.shape[1]), dtype=np.float32)
        out[:t] = x
        return out
    idx = np.linspace(0, t - 1, int(T), dtype=np.int64)
    return x[idx]


def build_model_features(seq, T, F_expected, sampling="head"):
    # Feature mode by model input size:
    #   F=8   -> handcrafted features.py
    #   F=132 -> flattened normalized landmarks (33*4)
    if int(F_expected) == 8:
        feats = landmarks_to_features(seq)
    elif int(F_expected) == 132:
        feats = seq.reshape(seq.shape[0], -1).astype(np.float32)
    else:
        raise ValueError(
            f"Unsupported model feature size F={F_expected}. "
            "Supported values: 8 (handcrafted) or 132 (flattened landmarks)."
        )
    mode = str(sampling).strip().lower()
    if mode in ("head", "trim_head", "first"):
        feats = pad_or_trim(feats, T=int(T))
    elif mode in ("uniform", "full_uniform"):
        feats = temporal_resample_uniform(feats, T=int(T))
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")
    return nan_to_num(feats)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="path to mp4")
    p.add_argument("--model", default=str(BASE_DIR / "data" / "processed" / "model.pt"))
    p.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    p.add_argument("--force_exercise", default="", help="Optional override for exercise class.")
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=180)
    p.add_argument("--topk_errors", type=int, default=5)
    p.add_argument("--error_threshold", type=float, default=0.5)
    p.add_argument(
        "--lower_body_rescore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, rescore lower-body classes using uniform temporal sampling.",
    )
    p.add_argument(
        "--lb_uniform_alpha",
        type=float,
        default=0.7,
        help="Blend weight for uniform sampling in lower-body rescore (0..1).",
    )
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")

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
    T_default = int(T_ckpt[0]) if hasattr(T_ckpt, "__len__") and not isinstance(T_ckpt, (str, bytes)) else int(T_ckpt)
    T = int(args.T) if args.T is not None else int(T_default)

    # 1) pose
    base_stride = max(1, int(args.stride))
    seq = extract_landmarks_from_video(args.video, max_frames=args.max_frames, stride=base_stride)
    seq = smooth_sequence(seq, vis_thr=0.35, max_gap=8, ema_alpha=0.25)
    seq = normalize_sequence(seq, rotate=True)

    # 2) features
    feats = build_model_features(seq, T=T, F_expected=F, sampling="head")
    if feats.shape[-1] != F:
        raise ValueError(f"Feature dim mismatch: expected F={F}, got {feats.shape[-1]}.")

    X = torch.tensor(feats[None, ...], dtype=torch.float32)  # (1,T,F)

    # 3) model
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

    with torch.no_grad():
        ex_logits_head, form_logits, err_logits = model(X)
        ex_logits_head = ex_logits_head.numpy().reshape(-1)
        form_logits = form_logits.numpy().reshape(-1)
        err_logits = err_logits.numpy().reshape(-1) if err_logits is not None else None

    ex_probs_head = softmax_np(ex_logits_head)
    ex_probs = ex_probs_head
    if bool(args.lower_body_rescore) and int(seq.shape[0]) > int(T):
        head_ex_id = int(ex_probs_head.argmax())
        head_name_raw = id_to_ex.get(head_ex_id, f"unknown_{head_ex_id}")
        if str(head_name_raw).startswith("unknown_"):
            head_name = str(head_name_raw)
        else:
            head_name = canonical_exercise_name(head_name_raw)
        if head_name in LOWER_BODY_CLASSES:
            seq_rescore = seq
            eff_stride = resolve_effective_stride(args.video, stride=base_stride, max_frames=args.max_frames)
            if eff_stride != base_stride:
                seq_rescore = extract_landmarks_from_video(args.video, max_frames=args.max_frames, stride=eff_stride)
                seq_rescore = smooth_sequence(seq_rescore, vis_thr=0.35, max_gap=8, ema_alpha=0.25)
                seq_rescore = normalize_sequence(seq_rescore, rotate=True)

            feats_uniform = build_model_features(seq_rescore, T=T, F_expected=F, sampling="uniform")
            X_uniform = torch.tensor(feats_uniform[None, ...], dtype=torch.float32)
            with torch.no_grad():
                ex_logits_uniform, _form_u, _err_u = model(X_uniform)
                ex_logits_uniform = ex_logits_uniform.numpy().reshape(-1)
            ex_probs_uniform = softmax_np(ex_logits_uniform)
            alpha = float(np.clip(args.lb_uniform_alpha, 0.0, 1.0))
            ex_probs = (1.0 - alpha) * ex_probs_head + alpha * ex_probs_uniform
            ex_probs = ex_probs / (np.sum(ex_probs) + 1e-9)

    form_probs = softmax_np(form_logits)

    ex_id = int(ex_probs.argmax())
    form_id = int(form_probs.argmax())
    if ex_id not in id_to_ex:
        raise ValueError(
            f"Predicted class id {ex_id} is missing in ex_map. "
            f"Known ids: {sorted(id_to_ex.keys())}"
        )
    ex_name = canonical_exercise_name(id_to_ex[ex_id])
    ex_probs_named = merge_exercise_probs(id_to_ex, ex_probs)
    exercise_conf = float(ex_probs_named.get(ex_name, ex_probs[ex_id]))
    model_ex_name = ex_name
    model_ex_conf = exercise_conf
    exercise_source = "model_head"
    if str(args.force_exercise).strip():
        ex_override = canonical_exercise_name(args.force_exercise)
        if ex_override in ex_probs_named:
            ex_name = ex_override
            exercise_conf = float(ex_probs_named.get(ex_name, 0.0))
            exercise_source = "override"
    model_form_name = "correct" if form_id == 1 else "incorrect"
    model_form_conf = float(form_probs[form_id])
    allowed_errors_by_exercise = load_allowed_errors_by_exercise(args.taxonomy)

    print("\n=== Prediction ===")
    print("Exercise:", ex_name, f"(p={exercise_conf:.3f})")
    print("Exercise source:", exercise_source)
    print("Exercise (model head):", model_ex_name, f"(p={model_ex_conf:.3f})")
    print("\nExercise probs:", ex_probs_named)

    # optional errors
    if err_logits is not None and id_to_err is not None:
        err_probs = sigmoid_np(err_logits)  # multi-label probabilities
        allowed_errors = None
        if ex_name in allowed_errors_by_exercise:
            allowed_errors = set(allowed_errors_by_exercise[ex_name])

        candidates = []
        for i in range(len(err_probs)):
            i_int = int(i)
            name = id_to_err.get(i_int, str(i_int))
            if allowed_errors is not None and name not in allowed_errors:
                continue
            candidates.append((name, float(err_probs[i_int])))

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        topk = int(min(args.topk_errors, len(candidates)))
        over_thr = [(name, prob) for name, prob in candidates if float(prob) >= float(args.error_threshold)]

        max_err_prob = float(candidates[0][1]) if candidates else 0.0
        linked_incorrect_prob = max_err_prob
        linked_correct_prob = float(max(0.0, 1.0 - linked_incorrect_prob))
        linked_form_name = "incorrect" if len(over_thr) > 0 else "correct"
        linked_form_conf = linked_incorrect_prob if linked_form_name == "incorrect" else linked_correct_prob

        print("Form (linked to errors):", linked_form_name, f"(p={linked_form_conf:.3f})")
        print(
            "Form probs (linked):",
            {"incorrect": float(linked_incorrect_prob), "correct": float(linked_correct_prob)},
        )
        print("Form (model head):", model_form_name, f"(p={model_form_conf:.3f})")
        print("Errors >= threshold:", [name for name, _ in over_thr])

        print("\n=== Errors (filtered by current exercise) ===")
        for name, prob in candidates[:topk]:
            print(f"- {name}: p={prob:.3f}")
    else:
        print("Form:", model_form_name, f"(p={model_form_conf:.3f})")
        print("Form probs:", {"incorrect": float(form_probs[0]), "correct": float(form_probs[1])})


if __name__ == "__main__":
    main()
