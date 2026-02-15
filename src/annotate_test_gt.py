import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "test"
DEFAULT_TAXONOMY = BASE_DIR / "data" / "processed" / "error_taxonomy.json"
DEFAULT_OUT_JSON = BASE_DIR / "data" / "processed" / "eval_eda" / "gt_map_template.json"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

EX_ALIASES = {
    "lat_pulldown": "latpulldown",
    "lateral_raise": "lateral_raises",
    "lateralraises": "lateral_raises",
    "deadlift": "romanian_deadlift",
    "dead_lift": "romanian_deadlift",
    "romanian deadlift": "romanian_deadlift",
    "rdl": "romanian_deadlift",
}

PREFERRED_EX_ORDER = [
    "squat",
    "bench_press",
    "latpulldown",
    "lateral_raises",
    "romanian_deadlift",
]


def canonical_exercise_name(name):
    ex = str(name).strip().lower()
    return EX_ALIASES.get(ex, ex)


def normalize_video_key(path_like):
    return str(path_like).replace("\\", "/").strip().lower()


def discover_videos(root_dir):
    root = Path(root_dir)
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    return sorted(out)


def load_json(path, default):
    p = Path(path)
    if not p.is_file():
        return default
    with open(p, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_exercise_classes(taxonomy_path):
    taxonomy = load_json(taxonomy_path, {})
    classes = set()
    if isinstance(taxonomy, dict):
        for ex_key in taxonomy.keys():
            ex = canonical_exercise_name(ex_key)
            if ex:
                classes.add(ex)

    if not classes:
        classes = set(PREFERRED_EX_ORDER)

    ordered = [ex for ex in PREFERRED_EX_ORDER if ex in classes]
    remaining = sorted([ex for ex in classes if ex not in ordered])
    return ordered + remaining


def make_square_with_padding(frame, size=720):
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    square = cv2.copyMakeBorder(
        resized,
        top,
        size - new_h - top,
        left,
        size - new_w - left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return square


def draw_text_block(frame, lines, x, y, line_h=24, color=(255, 255, 255)):
    yy = y
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        yy += line_h


def key_to_digit(key):
    if ord("1") <= key <= ord("9"):
        return key - ord("0")
    return None


def canonical_form_name(name):
    s = str(name).strip().lower()
    if s in ("c", "correct", "1", "true", "good"):
        return "correct"
    if s in ("i", "incorrect", "0", "false", "bad"):
        return "incorrect"
    return ""


def find_start_index(videos_rel, labels):
    for i, rel in enumerate(videos_rel):
        row = labels.get(rel, {})
        ex = str(row.get("exercise", "")).strip()
        fm = str(row.get("form", "")).strip()
        if not ex or not fm:
            return i
    return 0


def resolve_label_keys(videos_rel, labels):
    base_counts = {}
    for rel in videos_rel:
        base = Path(rel).name
        base_counts[base] = base_counts.get(base, 0) + 1

    key_by_rel = {}
    for rel in videos_rel:
        base = Path(rel).name
        if rel in labels:
            key_by_rel[rel] = rel
        elif base in labels and base_counts.get(base, 0) == 1:
            key_by_rel[rel] = base
        else:
            key_by_rel[rel] = rel
    return key_by_rel


def get_row_for_rel(rel, labels, key_by_rel):
    key = key_by_rel[rel]
    row = labels.get(key, {})
    if not isinstance(row, dict):
        row = {}
    row = {
        "exercise": canonical_exercise_name(row.get("exercise", "")),
        "form": canonical_form_name(row.get("form", "")),
    }
    labels[key] = row
    return key, row


def load_prediction_maps(path):
    data = load_json(path, [])
    if not isinstance(data, list):
        return {}, {}

    rel_map = {}
    base_items = {}
    base_counts = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("video_path", "")).strip()
        if not rel:
            continue
        rel_key = normalize_video_key(rel)
        base_key = normalize_video_key(Path(rel).name)

        try:
            ex_conf = float(item.get("exercise_conf", 0.0) or 0.0)
        except Exception:
            ex_conf = 0.0
        try:
            form_conf = float(item.get("form_conf", 0.0) or 0.0)
        except Exception:
            form_conf = 0.0

        row = {
            "exercise": canonical_exercise_name(item.get("exercise_pred", "")),
            "exercise_conf": ex_conf,
            "form": canonical_form_name(item.get("form_pred", "")),
            "form_conf": form_conf,
        }
        rel_map[rel_key] = row
        base_items[base_key] = row
        base_counts[base_key] = base_counts.get(base_key, 0) + 1

    base_unique = {}
    for key, row in base_items.items():
        if base_counts.get(key, 0) == 1:
            base_unique[key] = row
    return rel_map, base_unique


def get_prediction_for_rel(rel, pred_rel_map, pred_base_unique):
    rel_key = normalize_video_key(rel)
    if rel_key in pred_rel_map:
        return pred_rel_map[rel_key]
    base_key = normalize_video_key(Path(rel).name)
    return pred_base_unique.get(base_key)


def apply_prefill(
    videos_rel,
    labels,
    key_by_rel,
    classes,
    pred_rel_map,
    pred_base_unique,
    prefill_exercise,
    prefill_exercise_min_conf,
    prefill_form_mode,
    prefill_form_min_conf,
    prefill_overwrite,
):
    changed_rows = 0
    changed_exercise = 0
    changed_form = 0

    for rel in videos_rel:
        pred = get_prediction_for_rel(rel, pred_rel_map, pred_base_unique)
        if not isinstance(pred, dict):
            continue

        label_key, row = get_row_for_rel(rel, labels, key_by_rel)
        ex_cur = canonical_exercise_name(row.get("exercise", ""))
        fm_cur = canonical_form_name(row.get("form", ""))
        row_changed = False

        if bool(prefill_exercise) and (bool(prefill_overwrite) or not ex_cur):
            ex_new = canonical_exercise_name(pred.get("exercise", ""))
            ex_conf = float(pred.get("exercise_conf", 0.0) or 0.0)
            if (
                ex_new in classes
                and ex_conf >= float(prefill_exercise_min_conf)
                and ex_new != ex_cur
            ):
                row["exercise"] = ex_new
                row_changed = True
                changed_exercise += 1

        if str(prefill_form_mode) != "none" and (bool(prefill_overwrite) or not fm_cur):
            fm_new = canonical_form_name(pred.get("form", ""))
            fm_conf = float(pred.get("form_conf", 0.0) or 0.0)
            form_ok = str(prefill_form_mode) == "all" or (
                str(prefill_form_mode) == "high_conf"
                and fm_conf >= float(prefill_form_min_conf)
            )
            if fm_new in ("correct", "incorrect") and form_ok and fm_new != fm_cur:
                row["form"] = fm_new
                row_changed = True
                changed_form += 1

        if row_changed:
            labels[label_key] = row
            changed_rows += 1

    return {
        "changed_rows": int(changed_rows),
        "changed_exercise": int(changed_exercise),
        "changed_form": int(changed_form),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    p.add_argument("--out_json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--size", type=int, default=720)
    p.add_argument(
        "--seek_step", type=int, default=30, help="Frame step for seek keys ',' and '.'"
    )
    p.add_argument(
        "--fps_playback",
        type=float,
        default=30.0,
        help="Playback FPS for waitKey delay",
    )
    p.add_argument(
        "--prefill_predictions",
        default="",
        help="Optional test_predictions.json to prefill labels.",
    )
    p.add_argument(
        "--prefill_exercise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefill missing exercise from predictions.",
    )
    p.add_argument("--prefill_exercise_min_conf", type=float, default=0.85)
    p.add_argument(
        "--prefill_form",
        choices=["none", "all", "high_conf"],
        default="none",
        help="How to prefill form from predictions.",
    )
    p.add_argument("--prefill_form_min_conf", type=float, default=0.80)
    p.add_argument(
        "--prefill_overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow prefill to overwrite existing labels.",
    )
    p.add_argument(
        "--prefill_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply prefill and exit without opening video UI.",
    )
    args = p.parse_args()

    videos = discover_videos(args.input_dir)
    if not videos:
        print(f"No videos found in: {args.input_dir}")
        return

    root = Path(args.input_dir).resolve()
    videos_rel = [v.resolve().relative_to(root).as_posix() for v in videos]
    classes = load_exercise_classes(args.taxonomy)

    labels = load_json(args.out_json, {})
    if not isinstance(labels, dict):
        labels = {}
    key_by_rel = resolve_label_keys(videos_rel, labels)
    normalized = {}
    for rel in videos_rel:
        key, row = get_row_for_rel(rel, labels, key_by_rel)
        normalized[key] = row
    labels = normalized

    display_labels = {}
    for rel in videos_rel:
        key = key_by_rel[rel]
        display_labels[rel] = labels.get(key, {"exercise": "", "form": ""})

    pred_rel_map = {}
    pred_base_unique = {}
    if str(args.prefill_predictions).strip():
        pred_rel_map, pred_base_unique = load_prediction_maps(args.prefill_predictions)
        print(
            f"[prefill] loaded predictions: {len(pred_rel_map)} "
            f"from {args.prefill_predictions}"
        )
        changes = apply_prefill(
            videos_rel=videos_rel,
            labels=labels,
            key_by_rel=key_by_rel,
            classes=classes,
            pred_rel_map=pred_rel_map,
            pred_base_unique=pred_base_unique,
            prefill_exercise=bool(args.prefill_exercise),
            prefill_exercise_min_conf=float(args.prefill_exercise_min_conf),
            prefill_form_mode=str(args.prefill_form),
            prefill_form_min_conf=float(args.prefill_form_min_conf),
            prefill_overwrite=bool(args.prefill_overwrite),
        )
        print(
            f"[prefill] changed rows={changes['changed_rows']} "
            f"(exercise={changes['changed_exercise']}, form={changes['changed_form']})"
        )

    idx = find_start_index(videos_rel, display_labels)
    paused = False
    save_json(args.out_json, labels)

    if bool(args.prefill_only):
        print(f"Saved -> {args.out_json}")
        return

    window = "Test GT Annotator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, int(args.size), int(args.size))

    delay = int(max(1, round(1000.0 / max(1.0, float(args.fps_playback)))))

    while True:
        video_path = videos[idx]
        rel = videos_rel[idx]
        label_key, row = get_row_for_rel(rel, labels, key_by_rel)

        def commit_row():
            labels[label_key] = row
            save_json(args.out_json, labels)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[warn] cannot open video: {video_path}")
            idx = (idx + 1) % len(videos)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 0:
            total_frames = 0
        frame = None
        frame_idx = 0

        while True:
            if not paused:
                ret, fr = cap.read()
                if ret:
                    frame = fr
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                else:
                    paused = True

            if frame is None:
                canvas = np.zeros((int(args.size), int(args.size), 3), dtype=np.uint8)
            else:
                canvas = make_square_with_padding(frame, size=int(args.size))

            ex = row.get("exercise", "")
            fm = row.get("form", "")
            done = "yes" if ex and fm else "no"

            head = [
                f"Video {idx + 1}/{len(videos)} | labeled={done}",
                rel,
                f"frame: {frame_idx}/{max(0, total_frames - 1)} | paused={paused}",
                f"exercise: {ex or '-'} | form: {fm or '-'}",
            ]
            pred_hint = get_prediction_for_rel(rel, pred_rel_map, pred_base_unique)
            if isinstance(pred_hint, dict):
                head.append(
                    "suggest: "
                    f"ex={pred_hint.get('exercise','-')} ({float(pred_hint.get('exercise_conf', 0.0)):.2f}) | "
                    f"form={pred_hint.get('form','-')} ({float(pred_hint.get('form_conf', 0.0)):.2f})"
                )
            draw_text_block(canvas, head, 16, 30)

            y0 = 138
            cls_lines = ["Exercises (keys 1..9):"]
            for i, ex_name in enumerate(classes[:9], start=1):
                mark = "*" if ex_name == ex else " "
                cls_lines.append(f"{mark} {i}. {ex_name}")
            draw_text_block(canvas, cls_lines, 16, y0, line_h=22)

            controls = [
                "Controls:",
                "space play/pause | r restart | ,/. seek -/+",
                "c/9 form=correct | i/0 form=incorrect | x clear",
                "m manual input(exercise/form in terminal)",
                "n next video | p prev video | s/Enter save | q quit",
                "autosave: ON (on each label change)",
            ]
            draw_text_block(canvas, controls, 16, int(args.size) - 110, line_h=22)

            cv2.imshow(window, canvas)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                save_json(args.out_json, labels)
                cap.release()
                cv2.destroyAllWindows()
                print(f"Saved -> {args.out_json}")
                return
            key = cv2.waitKey(delay if not paused else 20) & 0xFF

            if key in (ord("q"), 27):
                save_json(args.out_json, labels)
                cap.release()
                cv2.destroyAllWindows()
                print(f"Saved -> {args.out_json}")
                return

            if key in (ord("s"), ord("S"), 13):
                save_json(args.out_json, labels)
                print(f"[save] {args.out_json}")
                continue

            if key == ord(" "):
                paused = not paused
                continue

            if key == ord("r"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
                continue

            if key == ord(","):
                cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - int(args.seek_step)))
                paused = True
                continue

            if key == ord("."):
                cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur + int(args.seek_step)))
                paused = True
                continue

            if key in (ord("c"), ord("C"), ord("9")):
                row["form"] = "correct"
                commit_row()
                print(
                    f"[set] {label_key} -> exercise={row.get('exercise','')} form=correct"
                )
                continue

            if key in (ord("i"), ord("I"), ord("0")):
                row["form"] = "incorrect"
                commit_row()
                print(
                    f"[set] {label_key} -> exercise={row.get('exercise','')} form=incorrect"
                )
                continue

            if key == ord("x"):
                row["exercise"] = ""
                row["form"] = ""
                commit_row()
                print(f"[set] {label_key} -> exercise= form=")
                continue

            if key == ord("m"):
                paused = True
                print("")
                print(f"[manual] {rel}")
                ex_in = input(f"exercise [{row.get('exercise','')}]: ").strip()
                fm_in = input(
                    f"form(correct/incorrect) [{row.get('form','')}]: "
                ).strip()

                ex_new = (
                    canonical_exercise_name(ex_in) if ex_in else row.get("exercise", "")
                )
                fm_new = canonical_form_name(fm_in) if fm_in else row.get("form", "")

                if ex_new in classes:
                    row["exercise"] = ex_new
                elif ex_in:
                    print(f"[warn] unknown exercise: {ex_in}")

                if fm_new in ("correct", "incorrect"):
                    row["form"] = fm_new
                elif fm_in:
                    print(f"[warn] unknown form: {fm_in}")

                commit_row()
                print(
                    f"[set] {label_key} -> exercise={row.get('exercise','')} form={row.get('form','')}"
                )
                continue

            dig = key_to_digit(key)
            if dig is not None and 1 <= dig <= len(classes[:9]):
                row["exercise"] = classes[dig - 1]
                commit_row()
                print(
                    f"[set] {label_key} -> exercise={row.get('exercise','')} form={row.get('form','')}"
                )
                continue

            if key == ord("n"):
                save_json(args.out_json, labels)
                idx = (idx + 1) % len(videos)
                paused = False
                break

            if key == ord("p"):
                save_json(args.out_json, labels)
                idx = (idx - 1) % len(videos)
                paused = False
                break

        cap.release()


if __name__ == "__main__":
    main()
