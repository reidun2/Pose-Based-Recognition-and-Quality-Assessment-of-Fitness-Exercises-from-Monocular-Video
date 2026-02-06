import os
import glob
import cv2
import numpy as np
from pathlib import Path

from extract_pose import extract_landmarks_from_video

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw_videos"

# ---------- Square helpers ----------
def make_square_with_padding(frame, size=640):
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    left = (size - new_w) // 2
    top = (size - new_h) // 2

    square = cv2.copyMakeBorder(
        resized,
        top, size - new_h - top,
        left, size - new_w - left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return square, scale, left, top

def remap_landmarks_to_square(lm, orig_w, orig_h, scale, left, top, size=640):
    lm2 = lm.copy()
    x = lm[:, 0] * orig_w
    y = lm[:, 1] * orig_h
    x2 = x * scale + left
    y2 = y * scale + top
    lm2[:, 0] = x2 / size
    lm2[:, 1] = y2 / size
    return lm2
# ----------------------------------

CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

def draw_pose(frame, lm, v_min=0.3):
    h, w = frame.shape[:2]

    def ok(i):
        x, y, z, v = lm[i]
        return (not np.isnan(x)) and (not np.isnan(y)) and (v >= v_min)

    for a, b in CONNECTIONS:
        if ok(a) and ok(b):
            ax, ay = int(lm[a, 0] * w), int(lm[a, 1] * h)
            bx, by = int(lm[b, 0] * w), int(lm[b, 1] * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)

    for i in range(33):
        if ok(i):
            cx, cy = int(lm[i, 0] * w), int(lm[i, 1] * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

def get_exercises(root=DATA_PATH):
    if not Path(root).exists():
        return []
    ex_dirs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
    return [os.path.basename(d) for d in ex_dirs]

def list_videos_filtered(root=DATA_PATH, exercise=None, label_filter="all"):
    """
    exercise: None -> all exercises, else exercise folder name
    label_filter: 'all' | 'correct' | 'incorrect'
    """
    videos = []

    ex_list = get_exercises(root) if exercise in [None, "all"] else [exercise]
    for ex_name in ex_list:
        ex_dir = os.path.join(root, ex_name)
        if not os.path.isdir(ex_dir):
            continue

        labels = ["correct", "incorrect"] if label_filter == "all" else [label_filter]
        for lab in labels:
            pattern = os.path.join(ex_dir, lab, "*.mp4")
            for fp in sorted(glob.glob(pattern)):
                videos.append((ex_name, lab, fp))

    return videos

def choose_exercise_ui(root=DATA_PATH):
    exs = get_exercises(root)
    if not exs:
        return "all"

    print("\nAvailable exercises:")
    print("  0) ALL")
    for i, ex in enumerate(exs, start=1):
        print(f"  {i}) {ex}")

    while True:
        ans = input("Choose exercise number (0 for ALL): ").strip()
        if ans == "0":
            return "all"
        if ans.isdigit():
            k = int(ans)
            if 1 <= k <= len(exs):
                return exs[k-1]
        print("Invalid choice. Try again.")

def next_label_filter(cur):
    # cycle: all -> correct -> incorrect -> all
    if cur == "all":
        return "correct"
    if cur == "correct":
        return "incorrect"
    return "all"

def main():
    root = str(DATA_PATH)

    # initial selection
    selected_ex = choose_exercise_ui(root)
    label_filter = "all"

    def refresh_list():
        vids = list_videos_filtered(root, exercise=selected_ex, label_filter=label_filter)
        return vids

    videos = refresh_list()
    if not videos:
        print(f"No videos for selection: exercise={selected_ex}, label={label_filter}")
        return

    print("\nControls:")
    print("  n -> next video")
    print("  p -> previous video")
    print("  space -> pause/resume")
    print("  +/- -> change visibility threshold")
    print("  c -> cycle label filter (all/correct/incorrect)")
    print("  e -> choose exercise again")
    print("  q or ESC -> quit")
    print()

    idx_video = 0
    paused = False
    v_min = 0.3
    stride = 2
    max_frames = 180
    size = 640

    frame_to_show = np.zeros((size, size, 3), dtype=np.uint8)

    while True:
        if not videos:
            frame_to_show[:] = 0
            cv2.putText(frame_to_show, "No videos in current filter.", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Skeleton Review", frame_to_show)
            key = cv2.waitKey(50) & 0xFF
            if key in [ord('q'), 27]:
                cv2.destroyAllWindows()
                return
            if key == ord('e'):
                selected_ex = choose_exercise_ui(root)
                videos = refresh_list()
                idx_video = 0
            if key == ord('c'):
                label_filter = next_label_filter(label_filter)
                videos = refresh_list()
                idx_video = 0
            continue

        ex_name, label, video_path = videos[idx_video]
        print(f"\n[{idx_video+1}/{len(videos)}] {ex_name}/{label}: {video_path}")

        seq = extract_landmarks_from_video(video_path, max_frames=max_frames, stride=stride)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video, skipping.")
            idx_video = (idx_video + 1) % len(videos)
            continue

        frame_idx = 0
        lm_idx = 0
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                orig_h, orig_w = frame.shape[:2]
                square_frame, scale, left, top = make_square_with_padding(frame, size=size)

                if frame_idx % stride == 0 and lm_idx < len(seq):
                    lm_orig = seq[lm_idx]
                    lm_sq = remap_landmarks_to_square(lm_orig, orig_w, orig_h, scale, left, top, size=size)
                    draw_pose(square_frame, lm_sq, v_min=v_min)
                    lm_idx += 1

                frame_idx += 1
                frame_to_show = square_frame

            # overlay UI text
            txt1 = f"EX={selected_ex} | FILTER={label_filter} | NOW={ex_name}/{label}"
            txt2 = f"v_min={v_min:.2f} stride={stride} size={size} | Keys: n/p space +/- c e q"
            cv2.putText(frame_to_show, txt1, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
            cv2.putText(frame_to_show, txt2, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

            cv2.imshow("Skeleton Review", frame_to_show)
            key = cv2.waitKey(20) & 0xFF

            if key in [ord('q'), 27]:
                cap.release()
                cv2.destroyAllWindows()
                return

            if key == ord(' '):
                paused = not paused

            if key == ord('n'):
                break

            if key == ord('p'):
                cap.release()
                idx_video = (idx_video - 1) % len(videos)
                break

            if key in [ord('+'), ord('=')]:
                v_min = min(1.0, v_min + 0.05)
            if key in [ord('-'), ord('_')]:
                v_min = max(0.0, v_min - 0.05)

            if key == ord('c'):
                label_filter = next_label_filter(label_filter)
                videos = refresh_list()
                idx_video = 0
                cap.release()
                break

            if key == ord('e'):
                selected_ex = choose_exercise_ui(root)
                videos = refresh_list()
                idx_video = 0
                cap.release()
                break

        cap.release()
        idx_video = (idx_video + 1) % len(videos)

if __name__ == "__main__":
    main()
