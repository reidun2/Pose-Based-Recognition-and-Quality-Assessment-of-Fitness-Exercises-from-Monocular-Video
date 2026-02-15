import numpy as np

# ----------------- helpers -----------------

def angle(a, b, c):
    """Angle ABC in degrees for 2D points a,b,c (each: (x,y))."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba) + 1e-9
    nbc = np.linalg.norm(bc) + 1e-9
    cosang = np.dot(ba, bc) / (nba * nbc)
    cosang = np.clip(cosang, -1, 1)
    return float(np.degrees(np.arccos(cosang)))

def rep_window(seq, st, en):
    st = max(0, int(st)); en = min(len(seq)-1, int(en))
    return seq[st:en+1]

def nanmean_xy(lm, idxs, axis=0):
    # lm: (33,2) or (33,4) but we use [:2] before
    vals = []
    for i in idxs:
        v = lm[i, axis]
        if not np.isnan(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan

def elbow_angle_frame(lm33, side="left"):
    # left: 11-13-15, right: 12-14-16
    if side == "left":
        sh, el, wr = 11, 13, 15
    else:
        sh, el, wr = 12, 14, 16
    pts = lm33[[sh, el, wr], :2]
    if np.isnan(pts).any():
        return np.nan
    return angle(pts[0], pts[1], pts[2])

def series_elbow_angles(W, side="left"):
    out = np.full((len(W),), np.nan, dtype=np.float32)
    for t in range(len(W)):
        out[t] = elbow_angle_frame(W[t], side=side)
    return out

def moving_std_1d(x, win=11):
    x = np.asarray(x, dtype=np.float32)
    # fill NaNs forward/backward
    ok = ~np.isnan(x)
    if not ok.any():
        return np.full_like(x, np.nan)
    xf = x.copy()
    for i in range(1, len(xf)):
        if np.isnan(xf[i]):
            xf[i] = xf[i-1]
    for i in range(len(xf)-2, -1, -1):
        if np.isnan(xf[i]):
            xf[i] = xf[i+1]
    win = max(5, int(win))
    if win % 2 == 0:
        win += 1
    half = win // 2
    out = np.zeros_like(xf)
    for i in range(len(xf)):
        a = max(0, i-half)
        b = min(len(xf), i+half+1)
        out[i] = np.std(xf[a:b])
    return out

def torso_lean_deg_frame(lm33):
    # hips -> shoulders, angle vs vertical (0=vertical)
    sx = nanmean_xy(lm33, [11,12], axis=0)
    sy = nanmean_xy(lm33, [11,12], axis=1)
    hx = nanmean_xy(lm33, [23,24], axis=0)
    hy = nanmean_xy(lm33, [23,24], axis=1)
    if np.isnan([sx,sy,hx,hy]).any():
        return np.nan
    vx = sx - hx
    vy = sy - hy
    norm = np.sqrt(vx*vx + vy*vy) + 1e-9
    cosang = np.clip(abs(vy) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

# ----------------- lat pulldown auto errors -----------------

def auto_label_lat_pulldown(seq_norm, st, en):
    """
    Heuristics for lat pulldown based on normalized landmarks.
    Returns dict error_name -> 0/1.
    """
    W = rep_window(seq_norm, st, en)
    if len(W) < 5:
        return {}

    # Define "top" and "bottom" inside the rep by wrist y relative to shoulders.
    # If y grows downward (typical), then bottom = max(depth), top = min(depth).
    depth = np.full((len(W),), np.nan, dtype=np.float32)
    wy = np.full((len(W),), np.nan, dtype=np.float32)
    sy = np.full((len(W),), np.nan, dtype=np.float32)
    wx = np.full((len(W),), np.nan, dtype=np.float32)
    sx = np.full((len(W),), np.nan, dtype=np.float32)

    for t in range(len(W)):
        lm = W[t]
        wy[t] = nanmean_xy(lm, [15,16], axis=1)
        sy[t] = nanmean_xy(lm, [11,12], axis=1)
        wx[t] = nanmean_xy(lm, [15,16], axis=0)
        sx[t] = nanmean_xy(lm, [11,12], axis=0)
        if np.isfinite(wy[t]) and np.isfinite(sy[t]):
            depth[t] = wy[t] - sy[t]

    # If depth is mostly NaN -> can't label
    if not np.isfinite(depth).any():
        return {}

    t_top = int(np.nanargmin(depth))
    t_bot = int(np.nanargmax(depth))

    # elbow angles at top (want near extension)
    eaL_top = elbow_angle_frame(W[t_top], "left")
    eaR_top = elbow_angle_frame(W[t_top], "right")
    ea_top = np.nanmean([eaL_top, eaR_top])

    # depth delta
    depth_top = float(depth[t_top]) if np.isfinite(depth[t_top]) else np.nan
    depth_bot = float(depth[t_bot]) if np.isfinite(depth[t_bot]) else np.nan
    depth_delta = (depth_bot - depth_top) if (np.isfinite(depth_bot) and np.isfinite(depth_top)) else np.nan

    # torso lean
    lean = np.full((len(W),), np.nan, dtype=np.float32)
    for t in range(len(W)):
        lean[t] = torso_lean_deg_frame(W[t])
    lean_top = float(lean[t_top]) if np.isfinite(lean[t_top]) else np.nan
    lean_bot = float(lean[t_bot]) if np.isfinite(lean[t_bot]) else np.nan
    lean_add = (lean_bot - lean_top) if (np.isfinite(lean_bot) and np.isfinite(lean_top)) else np.nan

    # swing (motion cheating): torso lean variability OR hip-x variability
    hipx = np.full((len(W),), np.nan, dtype=np.float32)
    for t in range(len(W)):
        hipx[t] = nanmean_xy(W[t], [23,24], axis=0)
    lean_std = float(np.nanstd(lean)) if np.isfinite(lean).any() else np.nan
    hipx_std = float(np.nanstd(hipx)) if np.isfinite(hipx).any() else np.nan

    # uneven pull at bottom: wrists y diff or elbows y diff
    lw_y = W[t_bot, 15, 1]; rw_y = W[t_bot, 16, 1]
    le_y = W[t_bot, 13, 1]; re_y = W[t_bot, 14, 1]
    wrist_diff = abs(float(lw_y - rw_y)) if np.isfinite(lw_y) and np.isfinite(rw_y) else 0.0
    elbow_diff = abs(float(le_y - re_y)) if np.isfinite(le_y) and np.isfinite(re_y) else 0.0

    # shoulder shrug: shoulders go up (y decreases) at bottom compared to top
    sy_top = float(sy[t_top]) if np.isfinite(sy[t_top]) else np.nan
    sy_bot = float(sy[t_bot]) if np.isfinite(sy[t_bot]) else np.nan
    shrug = (sy_bot < sy_top - 0.015) if (np.isfinite(sy_bot) and np.isfinite(sy_top)) else False

    # ---------------- thresholds (tune) ----------------
    TH_TOP_EXT = 155.0       # deg
    TH_DEPTH_DELTA = 0.06    # wrists should go down relative to shoulders
    TH_LEAN_ADD = 18.0       # deg additional lean at bottom
    TH_LEAN_STD = 6.0        # deg
    TH_HIPX_STD = 0.015      # normalized
    TH_UNEVEN = 0.035        # normalized y diff

    out = {}
    out["incomplete_top_extension"] = int(np.isfinite(ea_top) and (ea_top < TH_TOP_EXT))
    out["insufficient_pull_depth"]  = int(np.isfinite(depth_delta) and (depth_delta < TH_DEPTH_DELTA))
    out["excessive_lean_back"]      = int(np.isfinite(lean_add) and (lean_add > TH_LEAN_ADD))
    out["swinging_momentum"]        = int((np.isfinite(lean_std) and lean_std > TH_LEAN_STD) or
                                         (np.isfinite(hipx_std) and hipx_std > TH_HIPX_STD))
    out["uneven_pull"]             = int((wrist_diff > TH_UNEVEN) or (elbow_diff > TH_UNEVEN))
    out["shoulder_shrug"]          = int(bool(shrug))

    return out

# ----------------- main router -----------------

def auto_label_rep(seq_norm, ex_name, st, en):
    """
    seq_norm: normalized sequence (better for rules)
    Returns dict error_name -> 0/1
    """
    W = rep_window(seq_norm, st, en)
    if len(W) == 0:
        return {}

    # default empty
    out = {}

    ex = str(ex_name).lower()
    if ex in ("lateral_raise", "lateralraises"):
        ex = "lateral_raises"
    elif ex == "lat_pulldown":
        ex = "latpulldown"
    elif ex in ("deadlift", "dead_lift", "romanian_deadlift", "romanian deadlift", "rdl"):
        ex = "romanian_deadlift"

    if ex == "squat":
        # Landmarks indices (MediaPipe):
        # hip: 23/24, knee: 25/26, ankle: 27/28, shoulder: 11/12
        hip = W[:, 23, :2]
        knee = W[:, 25, :2]
        ankle = W[:, 27, :2]
        shoulder = W[:, 11, :2]

        dx = knee[:, 0] - ankle[:, 0]
        out["knee_over_toe"] = int(np.nanmax(np.abs(dx)) > 0.10)

        depth = hip[:, 1] - knee[:, 1]
        out["insufficient_depth"] = int(np.nanmin(depth) > -0.02)

        torso_dx = shoulder[:, 0] - hip[:, 0]
        torso_dy = shoulder[:, 1] - hip[:, 1]
        torso_angle = np.degrees(np.arctan2(np.abs(torso_dx), np.abs(torso_dy) + 1e-9))
        out["rounded_back"] = int(np.nanmax(torso_angle) > 35)

    elif ex == "bench_press":
        out["elbow_flaring"] = 0
        out["uneven_lockout"] = 0
        out["bar_not_touching_chest"] = 0

    elif ex == "latpulldown":
        # indices: shoulders 11,12 elbows 13,14 wrists 15,16 hips 23,24
        sh_y = W[:, [11, 12], 1]
        el_y = W[:, [13, 14], 1]
        wr_y = W[:, [15, 16], 1]
        hip_x = W[:, [23, 24], 0]

        sh_y_m = np.nanmean(sh_y, axis=1)
        el_y_m = np.nanmean(el_y, axis=1)
        wr_y_m = np.nanmean(wr_y, axis=1)
        hip_x_m = np.nanmean(hip_x, axis=1)

        # 1) bottom quality: elbows must be below shoulders at some point in rep
        # y растет вниз => "ниже" = больше
        bottom_ok = np.nanmax(el_y_m - sh_y_m) > 0.02
        out["elbows_not_below_shoulders"] = int(not bottom_ok)

        # 2) torso swing: hips x jitter too big (качает корпусом)
        swing = float(np.nanstd(hip_x_m)) if np.isfinite(np.nanstd(hip_x_m)) else 0.0
        out["swinging_momentum"] = int(swing > 0.020)  # подстрой 0.015..0.03

        # 3) shoulder shrug: shoulders move up noticeably
        # берем изменение sh_y в окне
        sh_range = float(np.nanmax(sh_y_m) - np.nanmin(sh_y_m)) if np.isfinite(np.nanmax(sh_y_m)) else 0.0
        out["shoulder_shrug"] = int(sh_range > 0.05)  # подстрой 0.03..0.07

    elif ex == "lateral_raises":
        vis_thr = 0.35
        depth = np.full((len(W),), np.nan, dtype=np.float32)
        side_y = np.full((len(W), 2), np.nan, dtype=np.float32)  # left/right selected y (wrist or elbow)

        for t in range(len(W)):
            lm = W[t]
            sh_vals = []
            for sh in (11, 12):
                y = lm[sh, 1]
                v = lm[sh, 3]
                if np.isfinite(y) and np.isfinite(v) and v >= vis_thr:
                    sh_vals.append(float(y))
            if not sh_vals:
                continue
            sh_m = float(np.mean(sh_vals))

            selected = []
            for side_i, (wr, el) in enumerate(((15, 13), (16, 14))):
                y_wr, v_wr = lm[wr, 1], lm[wr, 3]
                y_el, v_el = lm[el, 1], lm[el, 3]
                if np.isfinite(y_wr) and np.isfinite(v_wr) and v_wr >= vis_thr:
                    y_sel = float(y_wr)
                elif np.isfinite(y_el) and np.isfinite(v_el) and v_el >= vis_thr:
                    y_sel = float(y_el)
                else:
                    y_sel = np.nan
                side_y[t, side_i] = y_sel
                if np.isfinite(y_sel):
                    selected.append(y_sel)

            if selected:
                depth[t] = float(np.mean(selected) - sh_m)

        if np.isfinite(depth).any():
            top_i = int(np.nanargmin(depth))
            top_depth = float(depth[top_i])
        else:
            top_i = 0
            top_depth = 1.0

        top_l = side_y[top_i, 0]
        top_r = side_y[top_i, 1]
        top_asym = abs(float(top_l - top_r)) if np.isfinite(top_l) and np.isfinite(top_r) else 0.0

        eaL = series_elbow_angles(W, side="left")
        eaR = series_elbow_angles(W, side="right")
        min_candidates = []
        if np.isfinite(eaL).any():
            min_candidates.append(float(np.nanmin(eaL)))
        if np.isfinite(eaR).any():
            min_candidates.append(float(np.nanmin(eaR)))
        min_ea = min(min_candidates) if min_candidates else 180.0

        out["insufficient_rom"] = int(top_depth > 0.04)
        out["asymmetry"] = int(top_asym > 0.05)
        out["too_much_elbow_bend"] = int(min_ea < 120.0)
        if np.isfinite(depth).any():
            top_d = float(np.nanmin(depth))
            end_d = float(depth[-1]) if np.isfinite(depth[-1]) else float(np.nanmedian(depth))
            out["incomplete_lowering"] = int((end_d - top_d) < 0.05)
        else:
            out["incomplete_lowering"] = 0

        sh_y_m = np.nanmean(W[:, [11, 12], 1], axis=1)
        if np.isfinite(sh_y_m).any():
            sh_top = float(sh_y_m[top_i]) if 0 <= top_i < len(sh_y_m) and np.isfinite(sh_y_m[top_i]) else float(np.nanmedian(sh_y_m))
            out["shoulder_shrug"] = int((float(np.nanmedian(sh_y_m)) - sh_top) > 0.018)
        else:
            out["shoulder_shrug"] = 0

        hip_x = np.nanmean(W[:, [23, 24], 0], axis=1)
        out["torso_sway"] = int(np.isfinite(hip_x).any() and (float(np.nanstd(hip_x)) > 0.02))

    elif ex == "romanian_deadlift":
        sh = W[:, [11, 12], :2]
        hip = W[:, [23, 24], :2]

        sh_x = np.nanmean(sh[:, :, 0], axis=1)
        sh_y = np.nanmean(sh[:, :, 1], axis=1)
        hip_x = np.nanmean(hip[:, :, 0], axis=1)
        hip_y = np.nanmean(hip[:, :, 1], axis=1)

        # torso angle vs floor: 90 upright, 0 near parallel
        torso_floor = np.degrees(np.arctan2(np.abs(sh_y - hip_y), np.abs(sh_x - hip_x) + 1e-9))
        start_torso = float(torso_floor[0]) if np.isfinite(torso_floor[0]) else np.nan
        end_torso = float(torso_floor[-1]) if np.isfinite(torso_floor[-1]) else np.nan
        bottom_torso = float(np.nanmin(torso_floor)) if np.isfinite(torso_floor).any() else np.nan

        out["insufficient_hip_hinge"] = int(np.isfinite(bottom_torso) and (bottom_torso > 50.0))
        out["rounded_back"] = int(np.isfinite([start_torso, end_torso]).all() and (abs(end_torso - start_torso) > 18.0))
        out["incomplete_lockout"] = int(np.isfinite(end_torso) and (end_torso < 70.0))

        def knee_angles(side="left"):
            if side == "left":
                h, k, a = 23, 25, 27
            else:
                h, k, a = 24, 26, 28
            vals = np.full((len(W),), np.nan, dtype=np.float32)
            for t in range(len(W)):
                pts = W[t, [h, k, a], :2]
                if np.isnan(pts).any():
                    continue
                vals[t] = angle(pts[0], pts[1], pts[2])
            return vals

        kL = knee_angles("left")
        kR = knee_angles("right")
        k = np.nanmean(np.stack([kL, kR], axis=0), axis=0)
        out["excessive_knee_bend"] = int(np.isfinite(k).any() and (float(np.nanmin(k)) < 145.0))

        # proxy for bar path: wrists should stay relatively close to hips in side view
        l_dist = np.abs(W[:, 15, 0] - W[:, 23, 0])
        r_dist = np.abs(W[:, 16, 0] - W[:, 24, 0])
        wh_dist = np.nanmean(np.stack([l_dist, r_dist], axis=0), axis=0)
        out["bar_too_far_from_body"] = int(np.isfinite(wh_dist).any() and (np.nanmax(wh_dist) > 0.22))

    return out
