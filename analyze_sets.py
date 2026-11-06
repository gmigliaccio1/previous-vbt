import argparse
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

G = 9.80665
EXPECTED_DT = 0.005

def clean_read(path: str) -> pd.DataFrame:
    good_lines = ["t_s,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"]
    dropped = 0
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if(len(parts) != 7):
                dropped += 1
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                dropped += 1
                continue
            if vals[0] > 2 * 3600:
                dropped += 1
                continue
            good_lines.append(line.strip())
        if dropped :
            print(f"[warn] dropped {dropped} corrupted lines from {path}")
        df = pd.read_csv(io.StringIO("\n". join(good_lines)))
        df["t_s"] = df["t_s"] - df["t_s"].iloc[0]
        return df

def vertical_proxy(df: pd.DataFrame, lp_seconds: float = 2.0):
    acc = df[["ax_g", "ay_g", "az_g"]].to_numpy()
    win = max(1, int(lp_seconds / EXPECTED_DT))
    grav = (pd.DataFrame(acc).rolling(win, center=True, min_periods=1).mean().to_numpy())
    grav_norm = grav / np.linalg.norm(grav, axis=1, keepdims=True)
    proj = np.sum(acc*grav_norm, axis=1)
    grav_mag = np.linalg.norm(grav, axis=1)
    net_g = proj - grav_mag
    return net_g * G

def vertical_accel_from_orientation(acc, gyro_dps, t, alpha=0.98):
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0, 0.02)
    g_est = complementary_filter(acc, gyro_dps, dt, alpha=alpha)
    proj = np.sum(acc * g_est, axis=1)  
    net_g = proj - 1.0 
    return net_g * G

def find_quiet_windows(df: pd.DataFrame, gyro_thresh=8.0, mag_tol=0.04, min_dur=0.8):
    acc = df[["ax_g", "ay_g", "az_g"]].to_numpy()
    mag = np.linalg.norm(acc, axis=1)
    gyro = np.sqrt(df["gx_dps"] ** 2 + df["gy_dps"] ** 2 + df["gz_dps"] ** 2).to_numpy()
    quiet = (gyro < gyro_thresh) & (np.abs(mag - 1.0) < mag_tol)
 
    t = df["t_s"].to_numpy()
    windows = []
    in_win = False
    start = 0
    for i, q in enumerate(quiet):
        if q and not in_win:
            start = i
            in_win = True
        elif not q and in_win:
            in_win = False
            if t[i - 1] - t[start] >= min_dur:
                windows.append((t[start], t[i - 1]))
    if in_win and t[-1] - t[start] >= min_dur:
        windows.append((t[start], t[-1]))
    return windows

def naive_zupt_velocity(net_ms2: np.ndarray, t: np.ndarray, quiet_mask: np.ndarray):
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0, 0.02)
    v = np.zeros_like(net_ms2)
    for i in range(1, len(net_ms2)):
        if quiet_mask[i]:
            v[i] = 0.0
        else:
            v[i] = v[i - 1] + net_ms2[i] * dt[i]
    return v

def rotate_by_gyro(g_vec, gyro_dps, dt):
    gyro_rad = np.radians(gyro_dps)
    delta = -dt * np.cross(gyro_rad, g_vec)
    return g_vec + delta

def complementary_filter(acc, gyro_dps, dt, alpha=0.995):
    g_est = np.zeros_like(acc)
    g_est[0] = acc[0] / np.linalg.norm(acc[0])
    for i in range(1, len(acc)):
        gyro_pred = rotate_by_gyro(g_est[i - 1], gyro_dps[i], dt[i])
        
        acc_norm = np.linalg.norm(acc[i])
        gyro_mag = np.linalg.norm(gyro_dps[i])
        
        # Durante la spinta (accelerazione != 1g o rotazione), usa solo il giroscopio
        if abs(acc_norm - 1.0) > 0.08 or gyro_mag > 10.0:
            curr_alpha = 1.0
        else:
            curr_alpha = alpha

        acc_dir = acc[i] / acc_norm
        g_est[i] = curr_alpha * gyro_pred + (1 - curr_alpha) * acc_dir
        g_est[i] /= np.linalg.norm(g_est[i])
    return g_est

def find_rep_peaks(t, v, min_height=0.35, min_distance_s=2.5):
    dt = np.median(np.diff(t))
    distance_samples = max(1, int(min_distance_s / dt))
    peak_idx, _ = find_peaks(v, height=min_height, distance=distance_samples)
    return peak_idx

def rep_window_zero_crossing(t, v, peak_idx, search_bound_s=2.0):
    dt = np.median(np.diff(t))
    max_steps = int(search_bound_s / dt)

    start = peak_idx
    steps = 0
    while start > 0 and v[start] > 0 and steps < max_steps:
        start -= 1
        steps += 1

    end = peak_idx
    steps = 0
    while end < len(v) - 1 and v[end] > 0 and steps < max_steps:
        end += 1
        steps += 1

    return start, end

def summarize_rep(t,v, gyro, start_idx, end_idx):
    v_window = v[start_idx:end_idx+1]
    t_window = t[start_idx:end_idx+1]
    gyro_window = gyro[start_idx:end_idx+1]
    concentric = v_window[v_window > 0]
    return {
        "start_s": t_window[0],
        "end_s": t_window[-1],
        "duration_s": t_window[-1] - t_window[0],
        "mean_vel": float(np.mean(concentric)) if len(concentric) else 0.0,
        "peak_vel": float(np.max(v_window)),
        "mean_gyro": float(np.mean(gyro_window)),
    }

def calibrate_from_rest(acc, gyro_dps, seconds=2.0, fs=200, max_search_s=10.0):
    # Cerca il silenzio solo nei primi N secondi del file
    max_samples = int(max_search_s * fs)
    acc_sub = acc[:max_samples]
    gyro_sub = gyro_dps[:max_samples]
    
    win_len = int(seconds * fs)
    gyro_mag = np.linalg.norm(gyro_sub, axis=1)
    acc_mag = np.linalg.norm(acc_sub, axis=1)
    
    score = pd.Series(gyro_mag).rolling(win_len).mean().to_numpy() + \
            np.abs(pd.Series(acc_mag).rolling(win_len).mean().to_numpy() - 1.0) * 10.0
    
    score = np.nan_to_num(score, nan=1e9)
    best_end = np.argmin(score)
    best_start = max(0, best_end - win_len)
    
    gyro_bias = np.median(gyro_sub[best_start:best_end], axis=0)
    acc_scale = 1.0 / np.median(np.linalg.norm(acc_sub[best_start:best_end], axis=1))
    
    return acc * acc_scale, gyro_dps - gyro_bias

def anchored_rep_velocity(net, t, quiet_windows, peak_idx):
    """Velocity for one rep, forced to zero at the still moment before and after it.
    Returns (start_index, velocity_segment), or None if a still moment is missing."""
    starts = np.array([np.searchsorted(t, s) for s, e in quiet_windows])
    ends = np.array([np.searchsorted(t, e) for s, e in quiet_windows])
    before = ends[ends <= peak_idx]
    after = starts[starts > peak_idx]
    if len(before) == 0 or len(after) == 0:
        return None
    a, b = before[-1], after[0]
    dt = np.clip(np.diff(t[a:b + 1], prepend=t[a]), 0, 0.02)
    v_seg = np.cumsum(net[a:b + 1] * dt)          
    v_seg -= v_seg[0]
    v_seg -= np.linspace(0, v_seg[-1], len(v_seg)) 
    return a, v_seg

def concentric_window(v_seg, peak_local):
    i = peak_local
    while i > 0 and v_seg[i] > 0:
        i -= 1
    j = peak_local
    while j < len(v_seg) - 1 and v_seg[j] > 0:
        j += 1
    return i, j

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--label", default=None, help="e.g. '200kg x5'")
    ap.add_argument("--out", default=None, help="PNG output path (default: <csv>_plot.png)")
    ap.add_argument("--min-height", type=float, default=0.35)
    args = ap.parse_args()
 
    df = clean_read(args.csv_path)
    t = df["t_s"].to_numpy()
    acc = df[["ax_g", "ay_g", "az_g"]].to_numpy()
    gyro_dps = df[["gx_dps", "gy_dps", "gz_dps"]].to_numpy()

    acc, gyro_dps = calibrate_from_rest(acc, gyro_dps)
    df[["ax_g", "ay_g", "az_g"]] = acc
    df[["gx_dps", "gy_dps", "gz_dps"]] = gyro_dps

    gyro_mag = np.linalg.norm(gyro_dps, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)

    # net = vertical_proxy(df)
    net = vertical_accel_from_orientation(acc, gyro_dps, t)
    quiet_windows = find_quiet_windows(df)
    quiet_mask = np.zeros_like(t, dtype=bool)
    for s, e in quiet_windows:
        quiet_mask |= (t >= s) & (t <= e)
    v = naive_zupt_velocity(net, t, quiet_mask)

    peaks = find_rep_peaks(t, v, min_height=args.min_height)
    v_fixed = v.copy()
    print(f"\n detected  {len(peaks)} rep(s):")
    print(f"  {'rep':>3} {'start_s':>8} {'end_s':>8} {'dur_s':>6} {'mean_v':>7} {'peak_v':>7} {'m_gyro':>7}")
    for i, p in enumerate(peaks):
        seg = anchored_rep_velocity(net, t, quiet_windows, p)
        if seg is None:
            print(f"  {i+1:>3}  no still moment before/after this rep, skipped")
            continue
        a, v_seg = seg
        v_fixed[a:a + len(v_seg)] = v_seg
        i0, i1 = concentric_window(v_seg, p - a)
        r = summarize_rep(t, v_fixed, gyro_mag, a + i0, a + i1)
        print(f"  {i+1:>3} {r['start_s']:>8.2f} {r['end_s']:>8.2f} "
              f"{r['duration_s']:>6.2f} {r['mean_vel']:>7.3f} {r['peak_vel']:>7.3f} {r['mean_gyro']:>7.1f}")
    
    label = args.label or args.csv_path
    print(f"\n{label}")
    print(f"  duration: {t[-1]:.1f}s, samples: {len(df)}")
    print(f"  {len(quiet_windows)} candidate quiet window(s) found:")
    for s, e in quiet_windows:
        print(f"    {s:6.1f}s -> {e:6.1f}s  (dur {e - s:.1f}s)")
    print(f"  naive velocity range: {v.min():.2f} to {v.max():.2f} m/s "
          f"(compare against Qwik as a sanity check only)")
 
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
 
    axes[0].plot(t, df["ax_g"], label="ax", alpha=0.7)
    axes[0].plot(t, df["ay_g"], label="ay", alpha=0.7)
    axes[0].plot(t, df["az_g"], label="az", alpha=0.7)
    axes[0].plot(t, acc_mag, label="|a|", color="black", lw=1)
    axes[0].set_ylabel("accel (g)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{label} -- raw accelerometer")
 
    axes[1].plot(t, gyro_mag, color="tab:red")
    axes[1].set_ylabel("gyro |w| (deg/s)")
    axes[1].set_title("rotation magnitude -- spikes = fast movement / rerack / plate clank")
 
    axes[2].plot(t, net, color="tab:blue")
    axes[2].set_ylabel("net accel (m/s^2)")
    axes[2].set_title("rough 'vertical' proxy (first pass -- see docstring)")
 
    axes[3].plot(t, v, color="tab:green")
    axes[3].set_ylabel("velocity (m/s)")
    axes[3].set_title("naive ZUPT velocity (first pass -- NOT the final algorithm)")
    axes[3].set_xlabel("time (s)")
 
    for ax in axes:
        for s, e in quiet_windows:
            ax.axvspan(s, e, color="gray", alpha=0.15)
 
    plt.tight_layout()
    out_path = args.out or args.csv_path.replace(".csv", "_plot.png")
    plt.savefig(out_path, dpi=130)
    print(f"  saved plot -> {out_path}")
 
 
if __name__ == "__main__":
    main()