import os, glob, json
import cv2
import numpy as np
from pathlib import Path

from extract_pose import extract_landmarks_from_video
from pose_normalize import normalize_sequence
from reps import analyze_reps
from auto_errors import auto_label_rep
from pose_smooth import smooth_sequence

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw_videos"
OUT_REP_LABELS = BASE_DIR / "data" / "processed" / "labels_reps.json"
TAXONOMY_PATH  = BASE_DIR / "data" / "processed" / "error_taxonomy.json"

EX_CYCLE_KEY = ord('e')

CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

# ----------------- Drawing -----------------
def draw_pose(frame, lm, v_min=0.3):
    h, w = frame.shape[:2]

    def ok(i):
        x, y, z, v = lm[i]
        return (
            not np.isnan(x)
            and not np.isnan(y)
            and v >= v_min
            and abs(z) < 2.0   # убираем “вылеты” в глубину
        )


    for a, b in CONNECTIONS:
        if ok(a) and ok(b):
            ax, ay = int(lm[a, 0] * w), int(lm[a, 1] * h)
            bx, by = int(lm[b, 0] * w), int(lm[b, 1] * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)

    for i in range(33):
        if ok(i):
            cx, cy = int(lm[i, 0] * w), int(lm[i, 1] * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

def make_square_with_padding(frame, size=640):
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    left = (size - new_w) // 2
    top  = (size - new_h) // 2

    square = cv2.copyMakeBorder(
        resized,
        top,  size - new_h - top,
        left, size - new_w - left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
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

# ----------------- IO -----------------
def load_json(p, default):
    p = Path(p)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def list_videos(root=DATA_PATH, exercise=None):
    """
    If exercise is None -> all exercises
    If exercise = "squat" -> only data/raw_videos/squat/*.mp4
    """
    root = str(root)
    videos = []

    if exercise is not None:
        ex_dir = os.path.join(root, exercise)
        if not os.path.isdir(ex_dir):
            return []
        pattern = os.path.join(ex_dir, "*.mp4")
        for fp in sorted(glob.glob(pattern)):
            videos.append((exercise, fp))
        return videos

    ex_dirs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
    for ex_dir in ex_dirs:
        ex_name = os.path.basename(ex_dir)
        pattern = os.path.join(ex_dir, "*.mp4")
        for fp in sorted(glob.glob(pattern)):
            videos.append((ex_name, fp))
    return videos

# ----------------- HUD -----------------
def draw_errors_panel_bottom(frame, errors_list, rep_errors, max_items=9):
    h, w = frame.shape[:2]
    line_h = 24
    pad = 12

    items = errors_list[:max_items]
    if not items:
        return

    y0 = h - pad - line_h * (len(items) + 1)
    y0 = max(y0, 10)

    cv2.putText(frame, "Errors (1..9 toggle):", (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    y = y0 + line_h
    for i, e in enumerate(items, start=1):
        v = int(rep_errors.get(e, 0))
        txt = f"{i}. {e}: {v}"
        cv2.putText(frame, txt, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,255) if v else (200,200,200), 2)
        y += line_h

def draw_spine(frame, lm):
    h, w = frame.shape[:2]
    idxs = [11, 12, 23, 24]
    pts = []

    for i in idxs:
        if not ok(i):
            return
        pts.append((int(lm[i,0]*w), int(lm[i,1]*h)))

    cx = int((pts[0][0] + pts[1][0]) / 2)
    sx = int((pts[2][0] + pts[3][0]) / 2)

    cy = int((pts[0][1] + pts[1][1]) / 2)
    sy = int((pts[2][1] + pts[3][1]) / 2)

    cv2.line(frame, (cx, cy), (sx, sy), (0,255,0), 3)


def draw_hud(frame, folder_exercise, label, rep_idx_1based, rep_total, v_min, cur_errors, errors_list,
             vid_i=None, vid_total=None):
    y = 24
    if vid_i is not None and vid_total is not None:
        cv2.putText(frame, f"Video: {vid_i}/{vid_total}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += 22

    cv2.putText(frame, f"Folder: {folder_exercise}/{label} | Rep: {rep_idx_1based}/{rep_total} | v_min={v_min:.2f}",
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    y += 24

    cv2.putText(frame, "Keys: space pause | r replay | n/p rep | N/P video | e switch folder | 1..9 toggle | s save | q quit",
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    y += 24

    active = [e for e in errors_list if int(cur_errors.get(e, 0)) == 1]
    active_txt = ", ".join(active[:6]) + (" ..." if len(active) > 6 else "")
    cv2.putText(frame, f"Active: {active_txt if active else 'none'}",
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

def build_video_entry(folder_exercise, label, rep_results, rep_to_errors):
    reps_out = []
    for r in rep_results:
        rep_id = int(r["rep"])
        if rep_id in rep_to_errors:
            reps_out.append({
                "rep": rep_id,
                "start": int(r["start"]),
                "bottom": int(r["bottom"]),
                "end": int(r["end"]),
                "errors": rep_to_errors.get(rep_id, {})
            })
    return {"exercise": folder_exercise, "label": label, "reps": reps_out}

# ----------------- Playback -----------------
def play_rep_clip(
    cap,
    seq_draw,
    rep_start,
    rep_end,
    stride=1,
    v_min=0.3,
    win_name="Rep Annotator",
    square_size=640,
    overlay_fn=None,
):
    cap.set(cv2.CAP_PROP_POS_FRAMES, rep_start * stride)

    paused = False
    last_frame = None
    end_frame = rep_end * stride

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            t = cur_frame // stride

            square, scale, left, top, orig_w, orig_h = make_square_with_padding(frame, size=square_size)

            if 0 <= t < len(seq_draw):
                lm = seq_draw[t]
                lm_sq = remap_landmarks_to_square(lm, orig_w, orig_h, scale, left, top, size=square_size)
                draw_pose(square, lm_sq, v_min=v_min)

            if overlay_fn is not None:
                overlay_fn(square)

            last_frame = square
            cv2.imshow(win_name, square)

            if cur_frame >= end_frame:
                paused = True

        key_full = cv2.waitKey(20)
        key = key_full & 0xFF

        if key == ord(' '):
            paused = not paused
            continue
        if key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, rep_start * stride)
            paused = False
            continue
        if key in [ord('q'), 27]:
            return last_frame, key_full

        if key not in [255, 0]:
            return last_frame, key_full

    return last_frame, -1

# ----------------- Main -----------------
def main():
    taxonomy = load_json(TAXONOMY_PATH, {})
    rep_db  = load_json(OUT_REP_LABELS, {})

    # start in squat
    SUPPORTED_EXERCISES = ["squat", "bench_press", "latpulldown"]  # add more later
    exercise_i = 0
    exercise_filter = SUPPORTED_EXERCISES[exercise_i]

    videos = list_videos(DATA_PATH, exercise=exercise_filter)
    if not videos:
        print("No videos found for:", exercise_filter)
        return

    i_vid = 0
    stride = 1
    max_frames = 2000
    v_min = 0.3
    square_size = 640

    while True:
        # refresh current list if empty
        if not videos:
            videos = list_videos(DATA_PATH, exercise=exercise_filter)
            if not videos:
                print("No videos found for:", exercise_filter)
                return
            i_vid = 0

        folder_ex, video_path = videos[i_vid]   # folder_ex == exercise_filter
        label = "raw"
        video_key = str(video_path).replace("\\", "/")

        errors_list = list(taxonomy.get(folder_ex, {}).get("errors", {}).keys())

        print(f"\n[{i_vid+1}/{len(videos)}] folder={folder_ex}: {video_path}")

        # 1) pose
        from pose_smooth import smooth_sequence
        from pose_smooth import stabilize_lr_flip   # <-- добавь импорт (если вставил туда)

        seq_raw = extract_landmarks_from_video(video_path, max_frames=max_frames, stride=stride)

        # 1) FIX: стабилизируем лев/прав ДО сглаживания и нормализации
        seq_raw = stabilize_lr_flip(seq_raw, hysteresis_frames=3)

        # 2) дальше как было
        if exercise_filter == "latpulldown":
            seq_raw = lock_lr_orientation(seq_raw)

        seq_raw = smooth_sequence(seq_raw, vis_thr=0.35, max_gap=8, ema_alpha=0.25)
        seq_norm = normalize_sequence(seq_raw.copy(), rotate=(exercise_filter != "latpulldown"))


        # 2) reps for the CURRENT folder exercise
        rep_results, _ = analyze_reps(seq_norm, folder_ex, stride=stride)
        print(f"Detected reps: {len(rep_results)}")

        # load existing labels for this video
        entry = rep_db.get(video_key, {"exercise": folder_ex, "label": label, "reps": []})
        rep_to_errors = {int(r["rep"]): r.get("errors", {}) for r in entry.get("reps", [])}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video.")
            i_vid = (i_vid + 1) % len(videos)
            continue

        # if no reps: show first frame and allow switching folder with 'e'
        if len(rep_results) == 0:
            ret, frame0 = cap.read()
            cap.release()
            if ret:
                square0, *_ = make_square_with_padding(frame0, size=square_size)
                cv2.imshow("Rep Annotator", square0)
                while True:
                    kf = cv2.waitKey(0)
                    k = kf & 0xFF
                    if k in [ord('q'), 27]:
                        cv2.destroyAllWindows()
                        return
                    if k == ord('N'):
                        i_vid = (i_vid + 1) % len(videos)
                        break
                    if k == ord('P'):
                        i_vid = (i_vid - 1) % len(videos)
                        break
                    if k == EX_CYCLE_KEY:
                        exercise_i = (exercise_i + 1) % len(SUPPORTED_EXERCISES)
                        exercise_filter = SUPPORTED_EXERCISES[exercise_i]
                        videos = list_videos(DATA_PATH, exercise=exercise_filter)
                        i_vid = 0
                        print(f"[SWITCH FOLDER] -> {exercise_filter} | videos={len(videos)}")
                        break
            else:
                i_vid = (i_vid + 1) % len(videos)
            continue

        i_rep = 0
        vid_step = 0   # 0 stay, +1 next, -1 prev, 999 means switched folder

        while True:
            rep = rep_results[i_rep]
            rep_num = int(rep["rep"])
            st = int(rep["start"])
            bt = int(rep["bottom"])
            en = int(rep["end"])

            cur_errors = rep_to_errors.get(rep_num)
            if cur_errors is None:
                cur_errors = auto_label_rep(seq_norm, folder_ex, st, en)
            else:
                cur_errors = dict(cur_errors)

            for e in errors_list:
                cur_errors.setdefault(e, 0)

            rep_to_errors[rep_num] = cur_errors

            vid_i = i_vid + 1
            vid_total = len(videos)
            rep_idx_1based = i_rep + 1
            rep_total = len(rep_results)

            def overlay(frame):
                draw_hud(frame, folder_ex, label, rep_idx_1based, rep_total, v_min, cur_errors, errors_list,
                         vid_i=vid_i, vid_total=vid_total)
                draw_errors_panel_bottom(frame, errors_list, cur_errors)

            frame, kf = play_rep_clip(
                cap, seq_raw,
                st, en,
                stride=stride,
                v_min=v_min,
                win_name="Rep Annotator",
                square_size=square_size,
                overlay_fn=overlay
            )

            if kf == -1:
                continue

            k = kf & 0xFF

            # quit
            if k in [ord('q'), 27]:
                rep_db[video_key] = build_video_entry(folder_ex, label, rep_results, rep_to_errors)
                save_json(OUT_REP_LABELS, rep_db)
                cap.release()
                cv2.destroyAllWindows()
                return

            # toggle errors 1..9
            if ord('1') <= k <= ord('9'):
                idx = k - ord('1')
                if idx < len(errors_list):
                    e = errors_list[idx]
                    cur_errors[e] = 1 - int(cur_errors.get(e, 0))
                    rep_to_errors[rep_num] = cur_errors
                continue

            # save
            if k in [ord('s'), ord('S')]:
                rep_to_errors[rep_num] = cur_errors
                rep_db[video_key] = build_video_entry(folder_ex, label, rep_results, rep_to_errors)
                save_json(OUT_REP_LABELS, rep_db)
                print("Saved ->", OUT_REP_LABELS)
                continue

            # next/prev rep
            if k == ord('n'):
                i_rep = min(len(rep_results) - 1, i_rep + 1)
                continue
            if k == ord('p'):
                i_rep = max(0, i_rep - 1)
                continue

            # next/prev video
            if k == ord('N'):
                rep_db[video_key] = build_video_entry(folder_ex, label, rep_results, rep_to_errors)
                save_json(OUT_REP_LABELS, rep_db)
                vid_step = +1
                break
            if k == ord('P'):
                rep_db[video_key] = build_video_entry(folder_ex, label, rep_results, rep_to_errors)
                save_json(OUT_REP_LABELS, rep_db)
                vid_step = -1
                break

            # switch folder exercise
            if k == EX_CYCLE_KEY:
                rep_db[video_key] = build_video_entry(folder_ex, label, rep_results, rep_to_errors)
                save_json(OUT_REP_LABELS, rep_db)

                exercise_i = (exercise_i + 1) % len(SUPPORTED_EXERCISES)
                exercise_filter = SUPPORTED_EXERCISES[exercise_i]
                videos = list_videos(DATA_PATH, exercise=exercise_filter)
                i_vid = 0
                print(f"[SWITCH FOLDER] -> {exercise_filter} | videos={len(videos)}")

                vid_step = 999
                break

        cap.release()

        if vid_step == 999:
            continue
        if vid_step == +1:
            i_vid = (i_vid + 1) % len(videos)
        elif vid_step == -1:
            i_vid = (i_vid - 1) % len(videos)
        else:
            # if user didn't request video switch, stay on current video index
            i_vid = i_vid % len(videos)

if __name__ == "__main__":
    main()
