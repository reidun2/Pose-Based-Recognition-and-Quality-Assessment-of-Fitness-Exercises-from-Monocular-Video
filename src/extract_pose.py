# src/extract_pose.py
import cv2
import numpy as np
from pathlib import Path

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core.image import Image
from mediapipe.tasks.python.vision.core.image import ImageFormat


def frame_quality(lm33):
    """
    lm33: (33,4) x,y,z,conf
    quality = mean(conf) ignoring NaNs
    """
    v = lm33[:, 3]
    if np.all(np.isnan(v)):
        return 0.0
    return float(np.nanmean(v))


def _ema_update(prev, cur, alpha=0.25):
    """
    EMA on x,y,z only. Keeps conf from cur.
    prev/cur: (33,4)
    """
    if prev is None:
        return cur
    out = cur.copy()
    # where cur coords valid, do EMA; where NaN, keep prev
    for k in range(3):  # x,y,z
        c = cur[:, k]
        p = prev[:, k]
        ok = ~np.isnan(c)
        out[ok, k] = alpha * c[ok] + (1.0 - alpha) * p[ok]
        out[~ok, k] = p[~ok]
    # conf: prefer cur, but if NaN -> prev
    c = cur[:, 3]
    p = prev[:, 3]
    ok = ~np.isnan(c)
    out[ok, 3] = c[ok]
    out[~ok, 3] = p[~ok]
    return out


def extract_landmarks_from_video(
    video_path: str,
    model_path: str = "models/pose_landmarker_full.task",
    max_frames: int = 5000,
    stride: int = 2,
    conf_det: float = 0.55,
    conf_pres: float = 0.55,
    conf_track: float = 0.55,
    max_hold: int = 12,            # сколько кадров держать last_good при провале детекции
    ema_alpha: float = 0.25,       # 0.15..0.35 обычно норм
    update_q_thr: float = 0.35,    # обновлять last_good только если качество >= порога
    render_q_thr: float = 0.18,    # если качество совсем низкое -> вернуть NaN (чтобы не рисовать мусор)
    debug_print_every: int = 0     # 0 выключить, иначе печать каждые N кадров
):
    model_path = Path(model_path)
    if not model_path.is_file():
        model_path = Path(__file__).resolve().parent.parent / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model_path = str(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=conf_det,
        min_pose_presence_confidence=conf_pres,
        min_tracking_confidence=conf_track,
        output_segmentation_masks=False,
    )

    landmarks_seq = []
    frame_idx = 0

    dt_ms = max(1, int(1000.0 * stride / fps))
    timestamp_ms = 0

    last_good = None          # последний хороший (после EMA)
    hold_left = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            timestamp_ms += dt_ms

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks:
                # провал: держим last_good если можно
                if last_good is not None and hold_left > 0:
                    out = last_good.copy()
                    hold_left -= 1
                else:
                    out = np.full((33, 4), np.nan, dtype=np.float32)
            else:
                lm_list = result.pose_landmarks[0]
                cur = np.zeros((33, 4), dtype=np.float32)
                for i, lm in enumerate(lm_list):
                    vis = getattr(lm, "visibility", None)
                    pres = getattr(lm, "presence", None)
                    if vis is None and pres is None:
                        conf = 1.0
                    elif vis is None:
                        conf = float(pres)
                    elif pres is None:
                        conf = float(vis)
                    else:
                        conf = float(min(vis, pres))
                    cur[i] = [lm.x, lm.y, lm.z, conf]

                q = frame_quality(cur)

                # если детект есть, но он мусорный (низкая уверенность) —
                # не обновляем last_good, чтобы не было "прыжка"
                if (q >= update_q_thr) or (last_good is None):
                    sm = _ema_update(last_good, cur, alpha=ema_alpha)
                    last_good = sm
                    hold_left = max_hold
                    out = sm
                else:
                    # слишком низкое качество: лучше держать предыдущее
                    if last_good is not None:
                        out = last_good.copy()
                        hold_left = max(0, hold_left - 1)
                    else:
                        out = cur

            # если уверенность совсем низкая — возвращаем NaN, чтобы в UI просто не рисовать
            if last_good is not None:
                if frame_quality(out) < render_q_thr:
                    out = np.full((33, 4), np.nan, dtype=np.float32)

            landmarks_seq.append(out)

            if debug_print_every and (len(landmarks_seq) % int(debug_print_every) == 0):
                q = frame_quality(out)
                miss = 1 if (result is None or not result.pose_landmarks) else 0
                print(f"[pose] t={len(landmarks_seq)} q={q:.3f} miss={miss} hold={hold_left}")

            if len(landmarks_seq) >= max_frames:
                break

    cap.release()

    if len(landmarks_seq) == 0:
        return np.zeros((0, 33, 4), dtype=np.float32)

    return np.stack(landmarks_seq, axis=0)
