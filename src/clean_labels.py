import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = BASE_DIR / "data" / "processed" / "labels_reps.json"
DEFAULT_TAXONOMY = BASE_DIR / "data" / "processed" / "error_taxonomy.json"

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


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_video_path(video_path):
    raw = str(video_path).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p, p.as_posix()


def normalize_taxonomy(taxonomy):
    out = {}
    for ex_key, data in (taxonomy or {}).items():
        ex = canonical_exercise_name(ex_key)
        out.setdefault(ex, set()).update((data or {}).get("errors", {}).keys())
    return out


def to_bit(value):
    try:
        return 1 if int(value) == 1 else 0
    except Exception:
        s = str(value).strip().lower()
        return 1 if s in ("true", "yes", "y") else 0


def clean_rep(rep, allowed_errors, issues):
    if not isinstance(rep, dict):
        issues["rep_not_dict_dropped"] += 1
        return None

    missing = [k for k in REQUIRED_REP_FIELDS if k not in rep]
    if missing:
        issues["rep_missing_required_fields_dropped"] += 1
        return None

    try:
        rep_id = int(rep["rep"])
        st = int(rep["start"])
        bt = int(rep["bottom"])
        en = int(rep["end"])
    except Exception:
        issues["rep_bad_indices_dropped"] += 1
        return None

    if not (st <= bt <= en):
        issues["rep_invalid_interval_dropped"] += 1
        return None

    rep_errors = rep.get("errors") or {}
    if not isinstance(rep_errors, dict):
        rep_errors = {}
        issues["rep_errors_not_dict_reset"] += 1

    errors_clean = {}
    for err_name, value in rep_errors.items():
        key = str(err_name).strip()
        if key not in allowed_errors:
            issues["unknown_error_key_dropped"] += 1
            continue
        errors_clean[key] = to_bit(value)

    return {
        "rep": rep_id,
        "start": st,
        "bottom": bt,
        "end": en,
        "errors": errors_clean,
    }


def merge_entries(old_entry, new_entry):
    rep_map = {}
    for rep in old_entry.get("reps", []):
        rep_map[int(rep["rep"])] = rep
    for rep in new_entry.get("reps", []):
        rep_map[int(rep["rep"])] = rep
    merged_reps = [rep_map[k] for k in sorted(rep_map.keys())]
    return {
        "exercise": new_entry.get("exercise", old_entry.get("exercise", "")),
        "label": new_entry.get("label", old_entry.get("label", "raw")),
        "reps": merged_reps,
    }


def maybe_backup(path):
    src = Path(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = src.with_name(f"{src.stem}.backup_{stamp}{src.suffix}")
    shutil.copy2(src, backup)
    return backup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    p.add_argument("--out", default=None)
    p.add_argument("--backup", action="store_true")
    p.add_argument("--no-backup", dest="backup", action="store_false")
    p.set_defaults(backup=True)
    args = p.parse_args()

    labels_path = Path(args.labels)
    out_path = Path(args.out) if args.out else labels_path

    labels = load_json(labels_path)
    taxonomy = normalize_taxonomy(load_json(args.taxonomy))

    if not isinstance(labels, dict):
        raise ValueError("labels_reps.json must contain an object mapping video_path -> label entry.")
    if not taxonomy:
        raise ValueError("Taxonomy is empty after normalization.")

    issues = Counter()
    cleaned = {}
    reps_in = 0
    reps_out = 0

    for video_path, entry in labels.items():
        video_abs, video_key = normalize_video_path(video_path)
        if not video_abs.is_file():
            issues["video_file_missing_dropped"] += 1
            continue

        if not isinstance(entry, dict):
            issues["video_entry_not_dict_dropped"] += 1
            continue

        exercise = canonical_exercise_name(entry.get("exercise", ""))
        if not exercise:
            issues["video_missing_exercise_dropped"] += 1
            continue
        if exercise not in taxonomy:
            issues["exercise_missing_in_taxonomy_dropped"] += 1
            continue

        reps = entry.get("reps", [])
        if not isinstance(reps, list) or len(reps) == 0:
            issues["video_empty_reps_dropped"] += 1
            continue

        allowed_errors = taxonomy.get(exercise, set())
        reps_clean = []
        for rep in reps:
            reps_in += 1
            rep_clean = clean_rep(rep, allowed_errors, issues)
            if rep_clean is None:
                continue
            reps_clean.append(rep_clean)

        if not reps_clean:
            issues["video_all_reps_invalid_dropped"] += 1
            continue

        reps_out += len(reps_clean)
        new_entry = {
            "exercise": exercise,
            "label": entry.get("label", "raw"),
            "reps": reps_clean,
        }
        if video_key in cleaned:
            issues["duplicate_video_key_merged"] += 1
            cleaned[video_key] = merge_entries(cleaned[video_key], new_entry)
        else:
            cleaned[video_key] = new_entry

    if args.backup and out_path.resolve() == labels_path.resolve():
        backup_path = maybe_backup(labels_path)
        print("Backup saved ->", backup_path)

    save_json(out_path, cleaned)

    print("Saved cleaned labels ->", out_path)
    print("Videos:", len(labels), "->", len(cleaned))
    print("Reps:", reps_in, "->", reps_out)
    print("Issues:", dict(sorted(issues.items())))


if __name__ == "__main__":
    main()
