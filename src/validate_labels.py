import argparse
import json
from collections import Counter
from pathlib import Path


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


def canonical_exercise_name(name):
    ex = str(name).strip().lower()
    return EX_ALIASES.get(ex, ex)


def load_json(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_taxonomy(taxonomy):
    out = {}
    for ex_key, data in (taxonomy or {}).items():
        ex = canonical_exercise_name(ex_key)
        out.setdefault(ex, set()).update((data or {}).get("errors", {}).keys())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="data/processed/labels_reps.json")
    p.add_argument("--taxonomy", default="data/processed/error_taxonomy.json")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    labels = load_json(args.labels)
    taxonomy = normalize_taxonomy(load_json(args.taxonomy))

    issues = Counter()
    reps_total = 0
    videos_total = len(labels)
    exercise_counts = Counter()

    if videos_total == 0:
        raise ValueError("labels_reps.json is empty")

    for video_path, entry in labels.items():
        if not Path(video_path).is_file():
            issues["video_file_missing"] += 1
            continue

        ex = canonical_exercise_name(entry.get("exercise", ""))
        reps = entry.get("reps", [])
        if not ex:
            issues["missing_exercise"] += 1
            continue
        if not isinstance(reps, list) or len(reps) == 0:
            issues["empty_reps"] += 1
            continue
        exercise_counts[ex] += 1

        known_errors = taxonomy.get(ex, set())
        if ex not in taxonomy:
            issues["exercise_missing_in_taxonomy"] += 1

        for rep in reps:
            reps_total += 1
            if not isinstance(rep, dict):
                issues["rep_not_dict"] += 1
                continue

            for k in ("rep", "start", "bottom", "end", "errors"):
                if k not in rep:
                    issues["missing_required_rep_fields"] += 1

            try:
                st = int(rep.get("start"))
                bt = int(rep.get("bottom"))
                en = int(rep.get("end"))
                if not (st <= bt <= en):
                    issues["bad_intervals"] += 1
            except Exception:
                issues["bad_intervals"] += 1

            rep_errors = rep.get("errors", {}) or {}
            if not isinstance(rep_errors, dict):
                issues["errors_not_dict"] += 1
                continue

            unknown = [e for e in rep_errors.keys() if e not in known_errors]
            if unknown:
                issues["unknown_error_keys"] += 1

    print("Videos:", videos_total)
    print("Reps:", reps_total)
    print("Exercises:", dict(sorted(exercise_counts.items())))
    print("Issues:", dict(sorted(issues.items())))

    if args.strict and issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
