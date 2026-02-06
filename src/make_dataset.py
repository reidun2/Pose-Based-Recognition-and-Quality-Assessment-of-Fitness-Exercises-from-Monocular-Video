import json
import numpy as np
from pathlib import Path

from extract_pose import extract_landmarks_from_video
from pose_normalize import normalize_sequence

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "data" / "processed" / "labels_reps.json"
TAXONOMY_PATH = BASE_DIR / "data" / "processed" / "error_taxonomy.json"
OUT_NPZ = BASE_DIR / "data" / "processed" / "reps_dataset.npz"


def load_json(p, default):
    p = Path(p)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def pad_or_trim(X, T=120):
    # X: (t, F)
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
    """
    seq: (t, 33, 4) [x,y,z,vis] after normalize_sequence
    returns: (t, F) where F=33*4
    """
    t = seq.shape[0]
    return seq.reshape(t, -1).astype(np.float32)


def build_error_map(taxonomy):
    # taxonomy[exercise]["errors"] = {name: {...}, ...}
    all_errors = []
    for ex, data in taxonomy.items():
        errs = list((data.get("errors") or {}).keys())
        all_errors.extend(errs)
    # unique, stable order
    all_errors = sorted(set(all_errors))
    return {e: i for i, e in enumerate(all_errors)}


def main():
    labels = load_json(LABELS_PATH, {})
    taxonomy = load_json(TAXONOMY_PATH, {})

    if not labels:
        print("labels_reps.json is empty -> nothing to build")
        return

    err_map = build_error_map(taxonomy)

    # exercise map from labels
    ex_names = sorted({v["exercise"] for v in labels.values() if "exercise" in v})
    ex_map = {ex: i for i, ex in enumerate(ex_names)}

    X_list = []
    y_ex_list = []
    y_err_list = []
    y_score_list = []

    # cache pose per video (so we don't re-extract for each rep)
    pose_cache = {}

    T = 120
    stride = 1
    max_frames = 2000

    for video_path, entry in labels.items():
        ex_name = entry.get("exercise")
        reps = entry.get("reps", [])
        if not ex_name or not reps:
            continue

        # extract pose once
        if video_path not in pose_cache:
            seq_raw = extract_landmarks_from_video(video_path, max_frames=max_frames, stride=stride)
            seq_norm = normalize_sequence(seq_raw.copy(), rotate=True)
            pose_cache[video_path] = seq_norm
        else:
            seq_norm = pose_cache[video_path]

        for rep in reps:
            st = int(rep["start"])
            en = int(rep["end"])
            st = max(0, st)
            en = min(len(seq_norm) - 1, en)
            if en <= st + 2:
                continue

            rep_seq = seq_norm[st:en+1]  # (t,33,4)
            feats = landmarks_to_flat_features(rep_seq)  # (t,F)
            feats = pad_or_trim(feats, T=T)
            feats = nan_to_num(feats)

            # labels
            y_ex = ex_map[ex_name]

            # multi-label errors
            y_err = np.zeros((len(err_map),), dtype=np.float32)
            rep_errors = rep.get("errors", {}) or {}
            for e_name, v in rep_errors.items():
                if e_name in err_map:
                    y_err[err_map[e_name]] = 1.0 if int(v) == 1 else 0.0

            # optional score
            score = rep.get("score", None)
            if score is None:
                score = -1.0  # marker "no score"
            else:
                score = float(score)

            X_list.append(feats)
            y_ex_list.append(y_ex)
            y_err_list.append(y_err)
            y_score_list.append(score)

    X = np.stack(X_list, axis=0)               # (N,T,F)
    y_ex = np.array(y_ex_list, dtype=np.int64) # (N,)
    y_err = np.stack(y_err_list, axis=0)       # (N,K)
    y_score = np.array(y_score_list, dtype=np.float32)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        X=X,
        y_ex=y_ex,
        y_err=y_err,
        y_score=y_score,
        ex_map=ex_map,
        err_map=err_map,
        T=np.array([T], dtype=np.int32),
    )

    print("Saved ->", OUT_NPZ)
    print("N reps:", len(X))
    print("Exercises:", ex_map)
    print("Errors:", len(err_map))


if __name__ == "__main__":
    main()
