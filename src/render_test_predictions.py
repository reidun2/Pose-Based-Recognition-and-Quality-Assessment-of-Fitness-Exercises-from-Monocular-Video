import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from extract_pose import extract_landmarks_from_video
from pose_normalize import normalize_sequence
from pose_smooth import smooth_sequence
from reps import analyze_reps

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
DEFAULT_TAXONOMY = BASE_DIR / "data" / "processed" / "error_taxonomy.json"

CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


def canonical_exercise_name(name):
    ex = str(name).strip().lower()
    if ex == "lat_pulldown":
        return "latpulldown"
    if ex in ("lateral_raise", "lateralraises"):
        return "lateral_raises"
    if ex in ("deadlift", "dead_lift", "romanian deadlift", "rdl"):
        return "romanian_deadlift"
    return ex


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


def discover_videos(root_dir):
    root = Path(root_dir)
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    return sorted(out)


def load_predictions(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Predictions file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("Predictions JSON must be an array.")
    return rows


def make_square_with_padding(frame, size=640):
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
    return square, scale, left, top, w, h


def remap_landmarks_to_square(lm, orig_w, orig_h, scale, left, top, size=640):
    lm2 = lm.copy()
    x = lm[:, 0] * orig_w
    y = lm[:, 1] * orig_h
    x2 = x * scale + left
    y2 = y * scale + top
    lm2[:, 0] = x2 / size
    lm2[:, 1] = y2 / size
    return lm2


def draw_pose(frame, lm, v_min=0.35):
    h, w = frame.shape[:2]

    def ok(i):
        x, y, _z, v = lm[i]
        return (not np.isnan(x)) and (not np.isnan(y)) and (v >= v_min)

    for a, b in CONNECTIONS:
        if ok(a) and ok(b):
            ax, ay = int(lm[a, 0] * w), int(lm[a, 1] * h)
            bx, by = int(lm[b, 0] * w), int(lm[b, 1] * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)

    for i in range(33):
        if ok(i):
            cx, cy = int(lm[i, 0] * w), int(lm[i, 1] * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)


def filter_errors_for_exercise(err_list, allowed_errors):
    if allowed_errors is None:
        return list(err_list)
    out = []
    for e in err_list:
        name = str(e.get("name", "unknown"))
        if name in allowed_errors:
            out.append(e)
    return out


def choose_errors(pred, allowed_errors_by_exercise, topk=5):
    ex = canonical_exercise_name(pred.get("exercise_pred", ""))
    allowed_errors = None
    if ex in allowed_errors_by_exercise:
        allowed_errors = set(allowed_errors_by_exercise[ex])

    over = filter_errors_for_exercise(
        pred.get("errors_over_threshold", []) or [], allowed_errors
    )
    if len(over) > 0:
        lines = []
        for e in over:
            name = str(e.get("name", "unknown"))
            prob = e.get("prob", None)
            if prob is None:
                lines.append(name)
            else:
                lines.append(f"{name} ({float(prob):.3f})")
        return "Errors >= threshold", lines

    top = filter_errors_for_exercise(pred.get("top_errors", []) or [], allowed_errors)
    lines = []
    for e in top[: int(topk)]:
        name = str(e.get("name", "unknown"))
        prob = e.get("prob", None)
        if prob is None:
            lines.append(name)
        else:
            lines.append(f"{name} ({float(prob):.3f})")
    if len(lines) == 0:
        lines = ["none"]
    return f"Top errors ({int(topk)})", lines


def draw_overlay(
    frame, video_rel, pred, allowed_errors_by_exercise, topk_errors, rep_idx, rep_total
):
    ex = pred.get("exercise_pred", "unknown")
    ex_conf = float(pred.get("exercise_conf", 0.0))
    form = pred.get("form_pred", "unknown")
    form_conf = float(pred.get("form_conf", 0.0))

    title, err_lines = choose_errors(pred, allowed_errors_by_exercise, topk=topk_errors)

    y = 28
    cv2.putText(
        frame,
        f"Video: {video_rel}",
        (14, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    y += 26
    cv2.putText(
        frame,
        f"Exercise: {ex} ({ex_conf:.3f})",
        (14, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    y += 26
    cv2.putText(
        frame,
        f"Form: {form} ({form_conf:.3f})",
        (14, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    y += 26
    if rep_total > 0:
        rep_txt = (
            f"Rep: {rep_idx}/{rep_total}"
            if rep_idx is not None
            else f"Rep: -/{rep_total}"
        )
    else:
        rep_txt = "Rep: n/a"
    cv2.putText(
        frame, rep_txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2
    )
    y += 30
    cv2.putText(
        frame, title + ":", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2
    )
    y += 24
    for line in err_lines:
        cv2.putText(
            frame,
            f"- {line}",
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
        )
        y += 22
        if y > frame.shape[0] - 16:
            break


def safe_stem(path):
    s = path.stem
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "video"


def resolve_pose_max_frames(video_path, stride, max_frames):
    max_frames = int(max_frames)
    if max_frames > 0:
        return max_frames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 5000
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        return 5000

    sampled = int(np.ceil(total_frames / max(1, int(stride))))
    return max(1, sampled + 2)


def detect_reps(seq, exercise_name, stride):
    ex = canonical_exercise_name(exercise_name)
    if not ex or ex.startswith("unknown_"):
        return []

    rotate = ex not in ("latpulldown", "lateral_raises")
    seq_smooth = smooth_sequence(seq, vis_thr=0.35, max_gap=8, ema_alpha=0.25)
    seq_norm = normalize_sequence(seq_smooth.copy(), rotate=rotate)
    rep_results, _ = analyze_reps(seq_norm, ex, stride=int(stride))

    out = []
    for r in rep_results:
        try:
            rep_id = int(r.get("rep"))
            st = int(r.get("start"))
            en = int(r.get("end"))
        except Exception:
            continue
        if st <= en:
            out.append((rep_id, st, en))
    return out


def current_rep_idx(sample_t, reps):
    t = int(sample_t)
    for rep_id, st, en in reps:
        if int(st) <= t <= int(en):
            return int(rep_id)
    return None


def render_video(
    video_path,
    rel_key,
    pred,
    allowed_errors_by_exercise,
    out_path,
    stride,
    max_frames,
    size,
    v_min,
    topk_errors,
):
    pose_max_frames = resolve_pose_max_frames(
        video_path, stride=stride, max_frames=max_frames
    )
    seq = extract_landmarks_from_video(
        str(video_path),
        max_frames=int(pose_max_frames),
        stride=int(stride),
    )
    rep_exercise = pred.get("exercise_pred", "")
    reps = detect_reps(seq, rep_exercise, stride=stride)
    rep_total = len(reps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(src_fps) if src_fps and src_fps > 0 else 30.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(size), int(size)),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for: {out_path}")

    frame_idx = 0
    lm_idx = 0
    cur_lm = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        square, scale, left, top, orig_w, orig_h = make_square_with_padding(
            frame, size=size
        )

        if frame_idx % int(stride) == 0:
            if lm_idx < len(seq):
                cur_lm = seq[lm_idx]
            else:
                cur_lm = None
            lm_idx += 1

        if cur_lm is not None:
            lm_sq = remap_landmarks_to_square(
                cur_lm, orig_w, orig_h, scale, left, top, size=size
            )
            draw_pose(square, lm_sq, v_min=v_min)

        sample_t = frame_idx // int(stride)
        rep_idx = current_rep_idx(sample_t, reps)
        draw_overlay(
            square,
            rel_key,
            pred,
            allowed_errors_by_exercise,
            topk_errors=topk_errors,
            rep_idx=rep_idx,
            rep_total=rep_total,
        )
        writer.write(square)
        frame_idx += 1

    writer.release()
    cap.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=str(BASE_DIR / "data" / "test"))
    p.add_argument(
        "--predictions",
        default=str(BASE_DIR / "data" / "processed" / "test_predictions.json"),
    )
    p.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    p.add_argument(
        "--out_dir",
        default=str(BASE_DIR / "data" / "processed" / "test_predictions_overlay"),
    )
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=0, help="0 = process full video")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--v_min", type=float, default=0.35)
    p.add_argument("--topk_errors", type=int, default=5)
    args = p.parse_args()

    videos = discover_videos(args.input_dir)
    if not videos:
        print(f"No videos found in: {args.input_dir}")
        return

    rows = load_predictions(args.predictions)
    allowed_errors_by_exercise = load_allowed_errors_by_exercise(args.taxonomy)
    pred_map = {}
    for r in rows:
        key = str(r.get("video_path", "")).replace("\\", "/")
        pred_map[key.lower()] = r

    root_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0
    failed = 0

    for video_path in videos:
        rel_key = video_path.resolve().relative_to(root_dir).as_posix()
        pred = pred_map.get(rel_key.lower())
        if pred is None:
            print(f"[skip] No prediction for {rel_key}")
            skipped += 1
            continue

        out_name = safe_stem(video_path) + "_pred.mp4"
        out_path = out_dir / out_name
        print(f"[render] {rel_key} -> {out_path.name}")
        try:
            render_video(
                video_path=video_path,
                rel_key=rel_key,
                pred=pred,
                allowed_errors_by_exercise=allowed_errors_by_exercise,
                out_path=out_path,
                stride=int(args.stride),
                max_frames=int(args.max_frames),
                size=int(args.size),
                v_min=float(args.v_min),
                topk_errors=int(args.topk_errors),
            )
            rendered += 1
        except Exception as e:
            print(f"[error] {rel_key}: {e}")
            failed += 1

    print("")
    print(f"Rendered: {rendered}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
