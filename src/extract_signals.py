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


# -------------------- LATERAL RAISES EXTRACTS --------------------

def lateral_wrists_y_mean(seq):
    return _mean_coord_signal(seq, [15, 16], coord=1)


def lateral_shoulders_y_mean(seq):
    return _mean_coord_signal(seq, [11, 12], coord=1)


def lateral_raise_depth(seq):
    """
    Wrist Y depth relative to shoulder Y.
    Higher value means hands are lower than shoulders.
    Lateral raise peak is a local minimum of this signal.
    """
    return lateral_wrists_y_mean(seq) - lateral_shoulders_y_mean(seq)


def lateral_raise_depth_with_fallback(seq, vis_thr=0.35):
    """
    Robust lateral raise depth:
    - per side fallback wrist -> elbow
    - weighted fusion by landmark visibility
    - safe one-arm behavior (if only one side visible, use that side)
    Lower value means higher raise.
    """
    T = seq.shape[0]
    sig = np.full((T,), np.nan, dtype=np.float32)
    left = np.full((T,), np.nan, dtype=np.float32)
    right = np.full((T,), np.nan, dtype=np.float32)
    left_w = np.full((T,), 0.0, dtype=np.float32)
    right_w = np.full((T,), 0.0, dtype=np.float32)

    for t in range(T):
        lm = seq[t]
        sh_vals = []
        sh_x_vals = []
        for sh in (11, 12):
            y = lm[sh, 1]
            x = lm[sh, 0]
            v = lm[sh, 3]
            if np.isfinite(y) and np.isfinite(v) and v >= vis_thr:
                sh_vals.append(float(y))
            if np.isfinite(x) and np.isfinite(v) and v >= vis_thr:
                sh_x_vals.append(float(x))
        if not sh_vals:
            continue
        sh_y = float(np.mean(sh_vals))
        sh_x = float(np.mean(sh_x_vals)) if sh_x_vals else np.nan

        for side_i, (wr, el) in enumerate(((15, 13), (16, 14))):
            y_wr, v_wr = lm[wr, 1], lm[wr, 3]
            x_wr = lm[wr, 0]
            y_el, v_el = lm[el, 1], lm[el, 3]
            x_el = lm[el, 0]

            y_sel = np.nan
            x_sel = np.nan
            w_sel = 0.0
            if np.isfinite(y_wr) and np.isfinite(v_wr) and v_wr >= vis_thr:
                y_sel = float(y_wr)
                x_sel = float(x_wr) if np.isfinite(x_wr) else np.nan
                w_sel = float(v_wr)
            elif np.isfinite(y_el) and np.isfinite(v_el) and v_el >= vis_thr:
                y_sel = float(y_el)
                x_sel = float(x_el) if np.isfinite(x_el) else np.nan
                w_sel = float(v_el) * 0.9  # elbow fallback slightly lower confidence

            if np.isfinite(y_sel):
                # Blend vertical depth with horizontal abduction.
                # Helps front-view / one-arm clips where y-only can flatten.
                if np.isfinite(sh_x):
                    d = float((y_sel - sh_y) - 0.20 * abs(float(x_sel - sh_x)))
                else:
                    d = float(y_sel - sh_y)
                if side_i == 0:
                    left[t] = d
                    left_w[t] = w_sel
                else:
                    right[t] = d
                    right_w[t] = w_sel

    l_ok = np.isfinite(left)
    r_ok = np.isfinite(right)
    l_cov = float(np.mean(l_ok)) if T > 0 else 0.0
    r_cov = float(np.mean(r_ok)) if T > 0 else 0.0
    l_rng = float(np.nanpercentile(left, 90) - np.nanpercentile(left, 10)) if np.isfinite(left).any() else 0.0
    r_rng = float(np.nanpercentile(right, 90) - np.nanpercentile(right, 10)) if np.isfinite(right).any() else 0.0
    l_score = l_cov * max(0.0, l_rng)
    r_score = r_cov * max(0.0, r_rng)

    use_blend = (l_score > 1e-6 and r_score > 1e-6 and (min(l_score, r_score) / max(l_score, r_score)) >= 0.72)

    if use_blend:
        for t in range(T):
            have_l = l_ok[t]
            have_r = r_ok[t]
            if have_l and have_r:
                wl = max(1e-4, float(left_w[t]))
                wr = max(1e-4, float(right_w[t]))
                sig[t] = float((left[t] * wl + right[t] * wr) / (wl + wr))
            elif have_l:
                sig[t] = float(left[t])
            elif have_r:
                sig[t] = float(right[t])
    else:
        prefer_left = l_score >= r_score
        primary = left if prefer_left else right
        secondary = right if prefer_left else left
        p_ok = np.isfinite(primary)
        s_ok = np.isfinite(secondary)
        for t in range(T):
            if p_ok[t]:
                sig[t] = float(primary[t])
            elif s_ok[t]:
                sig[t] = float(secondary[t])

    return sig


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

    if ex in ("lateral_raises", "lateral_raise", "lateralraises"):
        sig = lateral_raise_depth_with_fallback(seq, vis_thr=0.35)
        return sig, {"smooth_win": 9, "type": "y", "bottom_is_min": True, "prominence": 0.025}

    if ex in ("deadlift", "dead_lift", "romanian_deadlift", "romanian deadlift", "rdl"):
        sig = squat_knee_angle_mean(seq)
        return sig, {"smooth_win": 11, "type": "angle", "bottom_is_min": True, "prominence": 7.0}

    return None, {}
