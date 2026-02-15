import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from extract_pose import extract_landmarks_from_video
from pose_normalize import normalize_sequence

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "data" / "processed" / "labels_reps.json"
TAXONOMY_PATH = BASE_DIR / "data" / "processed" / "error_taxonomy.json"
OUT_NPZ = BASE_DIR / "data" / "processed" / "reps_dataset.npz"

EX_ALIASES = {
    "lat_pulldown": "latpulldown",
    "lateral_raise": "lateral_raises",
    "lateralraises": "lateral_raises",
    "deadlift": "romanian_deadlift",
    "dead_lift": "romanian_deadlift",
    "romanian_deadlift": "romanian_deadlift",
    "romanian deadlift": "romanian_deadlift",
    "rdl": "romanian_deadlift",
}
REQUIRED_REP_FIELDS = ("rep", "start", "bottom", "end", "errors")


def canonical_exercise_name(name):
    ex = str(name).strip().lower()
    return EX_ALIASES.get(ex, ex)


def load_json(p, default):
    p = Path(p)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def pad_or_trim(X, T=120):
    t = X.shape[0]
    if t == T:
        return X
    if t > T:
        return X[:T]
    out = np.zeros((T, X.shape[1]), dtype=X.dtype)
    out[:t] = X
    return out


def nan_to_num(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def landmarks_to_flat_features(seq):
    t = seq.shape[0]
    return seq.reshape(t, -1).astype(np.float32)


def normalize_taxonomy(taxonomy):
    out = {}
    for ex_key, data in (taxonomy or {}).items():
        ex = canonical_exercise_name(ex_key)
        if ex not in out:
            out[ex] = {"errors": {}}
        errs = (data or {}).get("errors") or {}
        for e_name, e_val in errs.items():
            out[ex]["errors"][e_name] = e_val
    return out


def build_error_map(taxonomy_norm, include_exercises):
    all_errors = []
    for ex, data in taxonomy_norm.items():
        if include_exercises is not None and ex not in include_exercises:
            continue
        all_errors.extend(list((data.get("errors") or {}).keys()))
    all_errors = sorted(set(all_errors))
    return {e: i for i, e in enumerate(all_errors)}


def parse_include_exercises(raw):
    if str(raw).strip().lower() == "all":
        return None
    exs = [canonical_exercise_name(x) for x in str(raw).split(",") if x.strip()]
    return set(exs) if exs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=str(LABELS_PATH))
    p.add_argument("--taxonomy", default=str(TAXONOMY_PATH))
    p.add_argument("--out", default=str(OUT_NPZ))
    p.add_argument("--T", type=int, default=120)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=2000)
    p.add_argument("--include_exercises", default="squat,bench_press,latpulldown")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    labels = load_json(args.labels, {})
    taxonomy_raw = load_json(args.taxonomy, {})
    taxonomy = normalize_taxonomy(taxonomy_raw)
    include_exercises = parse_include_exercises(args.include_exercises)

    if not labels:
        print("labels_reps.json is empty -> nothing to build")
        return

    err_map = build_error_map(taxonomy, include_exercises)
    if not err_map:
        print("No errors from taxonomy for selected exercises.")
        return

    # Build exercise set from labels after canonicalization/filtering.
    ex_names = []
    for entry in labels.values():
        ex = canonical_exercise_name(entry.get("exercise", ""))
        if not ex:
            continue
        if include_exercises is not None and ex not in include_exercises:
            continue
        ex_names.append(ex)
    ex_names = sorted(set(ex_names))
    ex_map = {ex: i for i, ex in enumerate(ex_names)}

    if not ex_map:
        print("No videos match selected exercises.")
        return

    X_list = []
    y_ex_list = []
    y_form_list = []
    y_err_list = []
    y_score_list = []

    pose_cache = {}
    warn = Counter()
    reps_by_ex = Counter()

    for video_path, entry in labels.items():
        ex_name = canonical_exercise_name(entry.get("exercise", ""))
        reps = entry.get("reps", [])
        if not ex_name or not isinstance(reps, list) or len(reps) == 0:
            warn["video_missing_or_empty_reps"] += 1
            continue
        if include_exercises is not None and ex_name not in include_exercises:
            continue
        if ex_name not in ex_map:
            warn["video_unknown_exercise"] += 1
            continue

        allowed_errors = set((taxonomy.get(ex_name, {}).get("errors") or {}).keys())

        # Extract pose once per video.
        if video_path not in pose_cache:
            if not Path(video_path).is_file():
                warn["video_file_missing"] += 1
                continue
            try:
                seq_raw = extract_landmarks_from_video(
                    video_path,
                    max_frames=int(args.max_frames),
                    stride=int(args.stride),
                )
                seq_norm = normalize_sequence(seq_raw.copy(), rotate=True)
                pose_cache[video_path] = seq_norm
            except Exception:
                warn["pose_extract_failed"] += 1
                continue
        else:
            seq_norm = pose_cache[video_path]

        if len(seq_norm) == 0:
            warn["empty_pose_sequence"] += 1
            continue

        for rep in reps:
            if not isinstance(rep, dict):
                warn["rep_not_dict"] += 1
                continue

            missing = [k for k in REQUIRED_REP_FIELDS if k not in rep]
            if missing:
                warn["rep_missing_required_fields"] += 1
                if args.strict:
                    raise ValueError(f"Missing fields {missing} in {video_path}")
                continue

            try:
                st = int(rep["start"])
                bt = int(rep["bottom"])
                en = int(rep["end"])
            except Exception:
                warn["rep_bad_indices"] += 1
                if args.strict:
                    raise
                continue

            if not (st <= bt <= en):
                warn["rep_interval_invalid"] += 1
                if args.strict:
                    raise ValueError(f"Invalid interval in {video_path}: {st}, {bt}, {en}")
                continue

            st = max(0, st)
            en = min(len(seq_norm) - 1, en)
            if en <= st + 2:
                warn["rep_too_short"] += 1
                continue

            rep_seq = seq_norm[st:en + 1]
            feats = landmarks_to_flat_features(rep_seq)
            feats = pad_or_trim(feats, T=int(args.T))
            feats = nan_to_num(feats)

            rep_errors = rep.get("errors") or {}
            if not isinstance(rep_errors, dict):
                rep_errors = {}
                warn["rep_errors_not_dict"] += 1

            unknown_errors = [e for e in rep_errors.keys() if e not in allowed_errors]
            if unknown_errors:
                warn["rep_unknown_errors"] += 1
                if args.strict:
                    raise ValueError(f"Unknown errors for {ex_name}: {unknown_errors}")

            y_err = np.zeros((len(err_map),), dtype=np.float32)
            positive_errors = sum(1 for v in rep_errors.values() if int(v) == 1)
            for e_name, v in rep_errors.items():
                if e_name in err_map:
                    bit = 1 if int(v) == 1 else 0
                    y_err[err_map[e_name]] = float(bit)

            y_form = 1 if positive_errors == 0 else 0
            score = rep.get("score", -1.0)
            score = -1.0 if score is None else float(score)

            X_list.append(feats)
            y_ex_list.append(ex_map[ex_name])
            y_form_list.append(y_form)
            y_err_list.append(y_err)
            y_score_list.append(score)
            reps_by_ex[ex_name] += 1

    if len(X_list) == 0:
        print("No valid reps collected. Check labels and filters.")
        return

    X = np.stack(X_list, axis=0)
    y_ex = np.array(y_ex_list, dtype=np.int64)
    y_form = np.array(y_form_list, dtype=np.int64)
    y_err = np.stack(y_err_list, axis=0)
    y_score = np.array(y_score_list, dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y_ex=y_ex,
        y_form=y_form,
        y_err=y_err,
        y_score=y_score,
        ex_map=ex_map,
        err_map=err_map,
        T=np.array([int(args.T)], dtype=np.int32),
    )

    print("Saved ->", out_path)
    print("N reps:", len(X))
    print("Exercises:", ex_map)
    print("Reps by exercise:", dict(sorted(reps_by_ex.items())))
    print("Errors:", len(err_map))
    print("Form distribution:", {"incorrect": int((y_form == 0).sum()), "correct": int((y_form == 1).sum())})
    if warn:
        print("Warnings:", dict(sorted(warn.items())))


if __name__ == "__main__":
    main()
