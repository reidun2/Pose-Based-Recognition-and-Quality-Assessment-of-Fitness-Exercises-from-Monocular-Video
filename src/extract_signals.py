# src/extract_signals.py
import numpy as np

# MediaPipe indices:
# squat legs: L hip/knee/ankle = 23/25/27, R = 24/26/28
# bench elbows: L shoulder/elbow/wrist = 11/13/15, R = 12/14/16

EPS = 1e-9


def angle_3pts(a, b, c, eps=EPS):
    """Angle ABC in degrees for 2D points a,b,c (np arrays (2,))."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba) + eps
    nbc = np.linalg.norm(bc) + eps
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def fill_nans_1d(x):
    x = np.asarray(x, dtype=np.float32).copy()
    ok = ~np.isnan(x)
    if not ok.any():
        return x
    for i in range(1, len(x)):
        if np.isnan(x[i]):
            x[i] = x[i - 1]
    for i in range(len(x) - 2, -1, -1):
        if np.isnan(x[i]):
            x[i] = x[i + 1]
    return x


def smooth_1d(x, win=11):
    x = fill_nans_1d(x)
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, k, mode="same").astype(np.float32)


def _angle_signal(seq, a_idx, b_idx, c_idx):
    """Generic joint angle signal for seq (T,33,4). Uses x,y only."""
    T = seq.shape[0]
    sig = np.full((T,), np.nan, dtype=np.float32)
    for t in range(T):
        lm = seq[t]
        if np.isnan(lm[[a_idx, b_idx, c_idx], :2]).any():
            continue
        a = lm[a_idx, :2]
        b = lm[b_idx, :2]
        c = lm[c_idx, :2]
        sig[t] = angle_3pts(a, b, c)
    return sig


def _mean_coord_signal(seq, idxs, coord=1):
    """Mean coord (0=x,1=y) across idxs; if one NaN uses others."""
    T = seq.shape[0]
    sig = np.full((T,), np.nan, dtype=np.float32)
    for t in range(T):
        lm = seq[t]
        vals = []
        for i in idxs:
            v = lm[i, coord]
            if not np.isnan(v):
                vals.append(float(v))
        if vals:
            sig[t] = float(np.mean(vals))
    return sig


# -------------------- SQUAT EXTRACTS --------------------

def squat_knee_angle_left(seq):   # hip-knee-ankle
    return _angle_signal(seq, 23, 25, 27)

def squat_knee_angle_right(seq):
    return _angle_signal(seq, 24, 26, 28)

def squat_knee_angle_mean(seq):
    L = squat_knee_angle_left(seq)
    R = squat_knee_angle_right(seq)
    return np.nanmean(np.stack([L, R], axis=0), axis=0).astype(np.float32)

def squat_hip_y_mean(seq):
    return _mean_coord_signal(seq, [23, 24], coord=1)

def squat_knee_y_mean(seq):
    return _mean_coord_signal(seq, [25, 26], coord=1)


# -------------------- BENCH PRESS EXTRACTS --------------------

def bench_elbow_angle_left(seq):  # shoulder-elbow-wrist
    return _angle_signal(seq, 11, 13, 15)

def bench_elbow_angle_right(seq):
    return _angle_signal(seq, 12, 14, 16)

def bench_elbow_angle_mean(seq):
    L = bench_elbow_angle_left(seq)
    R = bench_elbow_angle_right(seq)
    return np.nanmean(np.stack([L, R], axis=0), axis=0).astype(np.float32)

def bench_wrists_y_mean(seq):
    return _mean_coord_signal(seq, [15, 16], coord=1)

def bench_wrists_x_mean(seq):
    return _mean_coord_signal(seq, [15, 16], coord=0)

def bench_shoulders_x_mean(seq):
    return _mean_coord_signal(seq, [11, 12], coord=0)

def bench_shoulders_y_mean(seq):
    return _mean_coord_signal(seq, [11, 12], coord=1)


# -------------------- LAT PULLDOWN EXTRACTS --------------------
# (Если уже добавляешь latpulldown: базовые сигналы те же "руки" + плечи)

def lat_elbow_angle_left(seq):
    return _angle_signal(seq, 11, 13, 15)

def lat_elbow_angle_right(seq):
    return _angle_signal(seq, 12, 14, 16)

def lat_elbow_angle_mean(seq):
    L = lat_elbow_angle_left(seq)
    R = lat_elbow_angle_right(seq)
    return np.nanmean(np.stack([L, R], axis=0), axis=0).astype(np.float32)

def lat_wrists_y_mean(seq):
    return _mean_coord_signal(seq, [15, 16], coord=1)

def lat_shoulders_y_mean(seq):
    return _mean_coord_signal(seq, [11, 12], coord=1)


# -------------------- DISPATCHER --------------------

def get_primary_signal(seq, exercise_name: str):
    """
    Returns (sig, meta) where meta includes smoothing window recommendation etc.
    """
    ex = exercise_name.lower()

    if ex == "squat":
        sig = squat_knee_angle_mean(seq)
        return sig, {"smooth_win": 11, "type": "angle", "bottom_is_min": True}

    if ex == "bench_press":
        sig = bench_elbow_angle_mean(seq)
        return sig, {"smooth_win": 11, "type": "angle", "bottom_is_min": True}

    if ex == "latpulldown":
        # часто кисти/локти лучше всего отражают тягу вниз
        sig = lat_wrists_y_mean(seq)  # y обычно растёт вниз -> "низ" = MAX
        return sig, {"smooth_win": 11, "type": "y", "bottom_is_min": False}

    return None, {}
