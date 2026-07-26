#!/usr/bin/env python3
"""
Aggressive Riding Detection Pipeline
=====================================
Traffic- and Road-Context Aware Aggressive Riding Detection for Two-Wheelers
Using Smartphone Sensors and DTW-based Weak Supervision

Usage:
    python aggressive_riding_pipeline.py aggressive_trip.zip unlabeled_trip.zip

Stages:
    1.  Sensor data processing (merge all CSVs into unified timeline)
    2.  Orientation-independent vehicle-frame transformation (quaternion + GPS bearing)
    3.  Jerk computation
    4.  Aggressive reference window generation
    5.  Feature extraction per aggressive window
    6.  Template selection (top 30% most aggressive windows)
    7.  Unlabeled trip windowing
    8.  Multivariate DTW matching (aggr. templates vs unlabeled windows)
    9.  DTW threshold estimation (inter-template distances)
    10. Label assignment (Aggressive / Non-Aggressive)
    11. Similarity score computation
    12. Sample-level label aggregation via voting
    13. Final labeled dataset generation
    14. Summary statistics
    15. Interactive Folium map (red = aggressive, green = normal)
    16. Plaintext summary report
"""

import os
import sys
import argparse
import zipfile
import shutil
import warnings
import textwrap
from datetime import datetime, timezone, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import folium

warnings.filterwarnings("ignore")

# ─── Pipeline-wide constants ──────────────────────────────────────────────────
SAMPLE_RATE = 40                           # Hz (motion sensors)
WINDOW_SEC = 5                             # seconds per window
STRIDE_SEC = 1                             # stride in seconds
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC  # 200 samples
STRIDE_SAMPLES = SAMPLE_RATE * STRIDE_SEC  # 40 samples
DT = 1.0 / SAMPLE_RATE                    # time step
TOP_AGG_FRACTION = 0.30                   # keep top-30% as templates
IST = timezone(timedelta(hours=5, minutes=30))

# ─── Stage 1: Sensor data processing ─────────────────────────────────────────

def ns_to_ist(ns: float) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).astimezone(IST)


def load_sensor_csv(path: str, value_cols: list, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.strip(), inplace=True)
    df = df[["time"] + value_cols].copy()
    df["datetime_ist"] = df["time"].apply(ns_to_ist)
    df.set_index("datetime_ist", inplace=True)
    df.drop(columns=["time"], inplace=True)
    df.columns = [f"{prefix}_{c.lower()}" for c in value_cols]

    if prefix == "Location":
        spd_col = "Location_speed"
        if spd_col in df.columns:
            df[spd_col] = df[spd_col] * 3.6      # m/s -> km/h
            df = df[df[spd_col] >= 0]
        bear_acc = "Location_bearingaccuracy"
        if bear_acc in df.columns:
            invalid = df[bear_acc] < 0
            if "Location_bearing" in df.columns:
                df.loc[invalid, "Location_bearing"] = np.nan
            df.drop(columns=[bear_acc], inplace=True)

    return df


def reindex_interpolate(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = df.reindex(idx, method="nearest")
    df = df.interpolate(method="time", limit_direction="both")
    return df


def process_zip(zip_path: str, label: str, out_dir: str) -> str:
    """
    Stage 1 – Extract ZIP, load all sensor CSVs, synchronise at 40 Hz,
    merge into a single DataFrame and save to <label>_merged_raw_data.csv.
    Returns the path to the saved CSV.
    """
    print(f"\n[Stage 1] Processing ZIP: {zip_path}  (label={label})")
    extract_dir = os.path.join(out_dir, f"{label}_extracted")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Locate sensor files
    sensor_map = {k: None for k in
                  ["Accelerometer", "Gyroscope", "Magnetometer",
                   "Location", "Gravity", "Orientation"]}
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            fl = fn.lower()
            fp = os.path.join(root, fn)
            if "accelerometer.csv" in fl:
                sensor_map["Accelerometer"] = fp
            elif "gyroscope.csv" in fl:
                sensor_map["Gyroscope"] = fp
            elif "magnetometer.csv" in fl:
                sensor_map["Magnetometer"] = fp
            elif "location.csv" in fl:
                sensor_map["Location"] = fp
            elif "gravity.csv" in fl:
                sensor_map["Gravity"] = fp
            elif "orientation.csv" in fl:
                sensor_map["Orientation"] = fp

    missing = [k for k, v in sensor_map.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing sensor files in {zip_path}: {missing}")

    # Load all sensors
    accel_df = load_sensor_csv(sensor_map["Accelerometer"], ["x", "y", "z"], "Accelerometer")
    grav_df  = load_sensor_csv(sensor_map["Gravity"],       ["x", "y", "z"], "Gravity")
    gyro_df  = load_sensor_csv(sensor_map["Gyroscope"],     ["x", "y", "z"], "Gyroscope")
    mag_df   = load_sensor_csv(sensor_map["Magnetometer"],  ["x", "y", "z"], "Magnetometer")
    ori_df   = load_sensor_csv(sensor_map["Orientation"],   ["qw", "qx", "qy", "qz"], "Orientation")
    loc_df   = load_sensor_csv(
        sensor_map["Location"],
        ["latitude", "longitude", "speed", "bearing", "bearingAccuracy", "horizontalAccuracy"],
        "Location"
    )

    # Raw location for time-range clamping
    raw_loc = pd.read_csv(sensor_map["Location"])
    raw_loc.rename(columns=lambda c: c.strip(), inplace=True)
    raw_loc["datetime_ist"] = raw_loc["time"].apply(ns_to_ist)
    raw_loc.set_index("datetime_ist", inplace=True)

    # Common time range (intersection of all sensor ranges)
    all_frames = [accel_df, grav_df, gyro_df, mag_df, raw_loc, ori_df]
    start = max(df.index.min() for df in all_frames)
    end   = min(df.index.max() for df in all_frames)
    common_idx = pd.date_range(start=start, end=end, freq="25ms")  # 40 Hz

    # Reindex and interpolate motion sensors
    accel_df = reindex_interpolate(accel_df, common_idx)
    grav_df  = reindex_interpolate(grav_df,  common_idx)
    gyro_df  = reindex_interpolate(gyro_df,  common_idx)
    mag_df   = reindex_interpolate(mag_df,   common_idx)
    ori_df   = reindex_interpolate(ori_df,   common_idx)

    # Merge high-frequency sensors
    merged = pd.concat([accel_df, grav_df, gyro_df, mag_df, ori_df], axis=1)

    # GPS: nearest + smooth speed
    loc_df = loc_df.reindex(common_idx, method="nearest")
    loc_df = loc_df.interpolate(method="time", limit_direction="both")
    if "Location_speed" in loc_df.columns:
        loc_df["Location_speed"] = (loc_df["Location_speed"]
                                    .ffill()
                                    .rolling(window=3, center=True, min_periods=1)
                                    .mean())
    loc_df = loc_df.ffill().bfill()

    merged = pd.concat([merged, loc_df], axis=1)

    # Drop rows with missing motion data
    motion_cols = [c for c in merged.columns if not c.startswith("Location_")]
    merged.dropna(subset=motion_cols, inplace=True)

    merged.reset_index(inplace=True)
    merged.rename(columns={"index": "datetime_ist"}, inplace=True)

    out_csv = os.path.join(out_dir, f"{label}_merged_raw_data.csv")
    merged.to_csv(out_csv, index=False)
    print(f"  Saved {len(merged)} rows -> {out_csv}")
    return out_csv


# ─── Stage 2: Orientation-independent vehicle-frame transformation ─────────────

def rotate_quaternion(vec: np.ndarray, qw, qx, qy, qz) -> np.ndarray:
    try:
        rot = R.from_quat([qx, qy, qz, qw])   # scipy: (x, y, z, w)
        return rot.apply(vec)
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def rotate_z_axis(vec: np.ndarray, bearing_deg: float) -> np.ndarray:
    """Rotate around Z-axis to align with vehicle forward direction."""
    try:
        theta = np.deg2rad(bearing_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        Rz = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1]
        ])
        return Rz @ vec
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def compute_vehicle_frame(merged_csv: str, label: str, out_dir: str) -> str:
    """
    Stage 2 – Transform accelerometer and gyroscope readings from device frame
    to vehicle frame using quaternion orientation + GPS bearing.
    Returns path to the vehicle-frame CSV.
    """
    print(f"\n[Stage 2] Vehicle-frame transformation for: {label}")
    df = pd.read_csv(merged_csv)

    accel_veh, gyro_veh = [], []

    for i in range(len(df)):
        row = df.iloc[i]
        try:
            ax, ay, az = row["Accelerometer_x"], row["Accelerometer_y"], row["Accelerometer_z"]
            gx, gy, gz = row["Gyroscope_x"],     row["Gyroscope_y"],     row["Gyroscope_z"]
            qw, qx     = row["Orientation_qw"],   row["Orientation_qx"]
            qy, qz     = row["Orientation_qy"],   row["Orientation_qz"]
            bearing    = row["Location_bearing"]

            # Step 1: quaternion rotation -> world frame
            a_world = rotate_quaternion(np.array([ax, ay, az]), qw, qx, qy, qz)
            g_world = rotate_quaternion(np.array([gx, gy, gz]), qw, qx, qy, qz)

            # Step 2: Z-axis rotation -> vehicle heading frame
            a_veh = rotate_z_axis(a_world, bearing)
            g_veh = rotate_z_axis(g_world, bearing)

            accel_veh.append(a_veh)
            gyro_veh.append(g_veh)
        except Exception:
            accel_veh.append([np.nan, np.nan, np.nan])
            gyro_veh.append([np.nan, np.nan, np.nan])

    avdf = pd.DataFrame(accel_veh, columns=["Accel_vehicle_x", "Accel_vehicle_y", "Accel_vehicle_z"])
    gvdf = pd.DataFrame(gyro_veh,  columns=["Gyro_vehicle_x",  "Gyro_vehicle_y",  "Gyro_vehicle_z"])
    df   = pd.concat([df, avdf, gvdf], axis=1)

    out_csv = os.path.join(out_dir, f"{label}_vehicle_frame.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved {len(df)} rows -> {out_csv}")
    return out_csv


# ─── Stage 3: Jerk ────────────────────────────────────────────────────────────

def add_jerk(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 3 – Append jerk = d(Accel_vehicle_x)/dt column."""
    df = df.copy()
    df["jerk"] = df["Accel_vehicle_x"].diff() / DT
    df["jerk"].fillna(0, inplace=True)
    return df


# ─── Stage 4: Aggressive reference windows ────────────────────────────────────

def create_windows(df: pd.DataFrame, window_id_prefix: str) -> list[dict]:
    """
    Stage 4 / Stage 7 – Sliding-window segmentation.
    Returns list of dicts: {window_id, start_index, end_index}.
    """
    n = len(df)
    windows = []
    idx = 0
    win_num = 1
    while idx + WINDOW_SAMPLES <= n:
        windows.append({
            "window_id":    f"{window_id_prefix}_{win_num}",
            "start_index":  idx,
            "end_index":    idx + WINDOW_SAMPLES - 1,
        })
        idx += STRIDE_SAMPLES
        win_num += 1
    return windows


# ─── Stage 5: Feature extraction ─────────────────────────────────────────────

def extract_features(df: pd.DataFrame, win: dict) -> dict:
    """Stage 5 – Compute summary statistics for a single window."""
    s, e = win["start_index"], win["end_index"] + 1
    seg = df.iloc[s:e]

    spd  = seg.get("Location_speed", pd.Series(dtype=float))
    acx  = seg["Accel_vehicle_x"]
    jerk = seg["jerk"]
    gx   = seg["Gyro_vehicle_x"]
    gy   = seg["Gyro_vehicle_y"]
    gz   = seg["Gyro_vehicle_z"]

    feats = {
        **win,
        "mean_speed":   spd.mean(),  "max_speed":  spd.max(),
        "min_speed":    spd.min(),   "std_speed":  spd.std(),
        "mean_accel":   acx.mean(),  "max_accel":  acx.abs().max(),
        "min_accel":    acx.min(),   "std_accel":  acx.std(),
        "mean_jerk":    jerk.mean(), "max_jerk":   jerk.abs().max(),
        "std_jerk":     jerk.std(),
        "gyro_x_std":   gx.std(),
        "gyro_y_std":   gy.std(),
        "gyro_z_std":   gz.std(),
    }
    return feats


# ─── Stage 6: Select strong aggressive templates ──────────────────────────────

def _safe_norm(series: pd.Series) -> pd.Series:
    rng = series.max() - series.min()
    if rng == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / rng


def select_templates(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 6 – Score each aggressive window and keep the top-30%.
    Score = 0.4*norm(max_accel) + 0.4*norm(max_jerk) + 0.2*norm(gyro_z_std)
    """
    df = feat_df.copy()
    df["_s_accel"] = _safe_norm(df["max_accel"])
    df["_s_jerk"]  = _safe_norm(df["max_jerk"])
    df["_s_gyro"]  = _safe_norm(df["gyro_z_std"])
    df["aggression_score_template"] = (
        0.4 * df["_s_accel"] +
        0.4 * df["_s_jerk"]  +
        0.2 * df["_s_gyro"]
    )
    df.sort_values("aggression_score_template", ascending=False, inplace=True)
    cutoff = max(1, int(np.ceil(TOP_AGG_FRACTION * len(df))))
    templates = df.head(cutoff).drop(columns=["_s_accel", "_s_jerk", "_s_gyro"])
    print(f"  Selected {len(templates)} of {len(df)} aggressive windows as templates")
    return templates.reset_index(drop=True)


# ─── Stage 8: Multivariate DTW matching ──────────────────────────────────────

DTW_FEATURES = ["Accel_vehicle_x", "Gyro_vehicle_z", "Location_speed", "jerk"]


def _normalise_signal(arr: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation per channel."""
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std[std == 0] = 1.0
    return (arr - mean) / std


def get_window_signal(df: pd.DataFrame, win: dict) -> np.ndarray:
    """Extract normalised multi-channel signal for a window (shape: [T, 4])."""
    s, e = win["start_index"], win["end_index"] + 1
    seg = df.iloc[s:e][DTW_FEATURES].fillna(0).values.astype(float)
    return _normalise_signal(seg)


def dtw_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    dist, _ = fastdtw(sig_a, sig_b, dist=euclidean)
    return dist


def match_windows(unlabeled_df: pd.DataFrame, unlabeled_wins: list,
                  agg_df: pd.DataFrame, template_wins: list,
                  template_signals: list) -> pd.DataFrame:
    """
    Stage 8 – For every unlabeled window, find the minimum DTW distance
    to any aggressive template. Returns results DataFrame.
    """
    print(f"\n[Stage 8] DTW matching: {len(unlabeled_wins)} unlabeled windows "
          f"vs {len(template_wins)} templates …")

    results = []
    for i, uwin in enumerate(unlabeled_wins):
        u_sig = get_window_signal(unlabeled_df, uwin)
        best_dist  = np.inf
        best_match = None
        for j, tmpl_sig in enumerate(template_signals):
            d = dtw_distance(u_sig, tmpl_sig)
            if d < best_dist:
                best_dist  = d
                best_match = template_wins[j]["window_id"]

        results.append({
            **uwin,
            "best_match": best_match,
            "dtw_distance": best_dist,
        })
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(unlabeled_wins)} windows matched …")

    print(f"  Matching complete.")
    return pd.DataFrame(results)


# ─── Stage 9: Threshold estimation ───────────────────────────────────────────

def estimate_threshold(agg_df: pd.DataFrame, template_wins: list,
                       template_signals: list) -> tuple[float, float, float]:
    """
    Stage 9 – Compute inter-template DTW distances.
    threshold = mean + 2 * std
    Returns (threshold, mean_dist, std_dist).
    """
    print("\n[Stage 9] Estimating DTW threshold from inter-template distances …")
    inter_dists = []
    pairs = list(combinations(range(len(template_signals)), 2))
    # Cap at 300 pairs to avoid very long runtimes on large template sets
    if len(pairs) > 300:
        rng = np.random.default_rng(42)
        pairs = [pairs[i] for i in rng.choice(len(pairs), 300, replace=False)]

    for (i, j) in pairs:
        d = dtw_distance(template_signals[i], template_signals[j])
        inter_dists.append(d)

    if not inter_dists:
        # Only one template – use a heuristic
        return (50.0, 25.0, 12.5)

    mean_d = np.mean(inter_dists)
    std_d  = np.std(inter_dists)
    threshold = mean_d + 2 * std_d
    print(f"  Inter-template DTW: mean={mean_d:.2f}, std={std_d:.2f} -> threshold={threshold:.2f}")
    return threshold, mean_d, std_d


# ─── Stage 10 + 11: Labels and similarity score ──────────────────────────────

def assign_labels(match_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Stages 10 & 11 – Label each window and compute similarity score."""
    df = match_df.copy()
    df["label"] = np.where(df["dtw_distance"] < threshold, "Aggressive", "Non-Aggressive")
    df["similarity_score"] = np.exp(-df["dtw_distance"] / threshold)
    return df


# ─── Stage 12: Sample-level labels ───────────────────────────────────────────

def sample_level_labels(labeled_wins: pd.DataFrame, total_samples: int) -> pd.DataFrame:
    """
    Stage 12 – Voting: for each sample accumulate aggressive votes and
    total covering windows, then compute per-sample aggression score.
    """
    agg_votes   = np.zeros(total_samples, dtype=float)
    total_votes = np.zeros(total_samples, dtype=float)

    for _, row in labeled_wins.iterrows():
        s = int(row["start_index"])
        e = int(row["end_index"]) + 1
        is_agg = 1.0 if row["label"] == "Aggressive" else 0.0
        agg_votes[s:e]   += is_agg
        total_votes[s:e] += 1.0

    aggression_score = np.where(
        total_votes > 0,
        agg_votes / total_votes,
        0.0
    )
    aggressive_label = np.where(aggression_score > 0.5, "Aggressive", "Non-Aggressive")

    return pd.DataFrame({
        "aggression_score":  aggression_score,
        "aggressive_label":  aggressive_label,
    })


# ─── Stage 14: Summary statistics ────────────────────────────────────────────

def compute_summary(final_df: pd.DataFrame) -> dict:
    """Stage 14 – Compute trip-level summary statistics."""
    dt_per_sample = DT  # seconds
    total_samples = len(final_df)

    agg_mask = final_df["aggressive_label"] == "Aggressive"
    agg_samples = agg_mask.sum()
    non_agg_samples = total_samples - agg_samples

    agg_dur = agg_samples * dt_per_sample
    non_agg_dur = non_agg_samples * dt_per_sample
    total_dur = total_samples * dt_per_sample

    pct_agg     = 100.0 * agg_samples / total_samples if total_samples > 0 else 0.0
    pct_non_agg = 100.0 - pct_agg

    # Aggressive episodes (consecutive runs of "Aggressive" samples)
    episodes = []
    in_episode = False
    ep_start = 0
    labels = final_df["aggressive_label"].values
    for i, lbl in enumerate(labels):
        if lbl == "Aggressive" and not in_episode:
            in_episode = True
            ep_start = i
        elif lbl != "Aggressive" and in_episode:
            in_episode = False
            episodes.append((ep_start, i - 1))
    if in_episode:
        episodes.append((ep_start, len(labels) - 1))

    num_episodes = len(episodes)
    ep_durations = [(e - s + 1) * dt_per_sample for s, e in episodes]
    avg_ep_dur = np.mean(ep_durations) if ep_durations else 0.0
    max_ep_dur = np.max(ep_durations) if ep_durations else 0.0

    def fmt_dur(secs):
        m = int(secs // 60)
        s = secs % 60
        return f"{m}m {s:.1f}s"

    return {
        "total_duration":    fmt_dur(total_dur),
        "agg_duration":      fmt_dur(agg_dur),
        "non_agg_duration":  fmt_dur(non_agg_dur),
        "pct_aggressive":    pct_agg,
        "pct_non_agg":       pct_non_agg,
        "num_episodes":      num_episodes,
        "avg_episode_dur":   fmt_dur(avg_ep_dur),
        "max_episode_dur":   fmt_dur(max_ep_dur),
        "episodes":          episodes,
    }


# ─── Stage 15: Folium map ─────────────────────────────────────────────────────

def _window_meta_for_sample(sample_idx: int, labeled_wins: pd.DataFrame) -> dict:
    """Return best-match DTW window info for a given sample index."""
    covering = labeled_wins[
        (labeled_wins["start_index"] <= sample_idx) &
        (labeled_wins["end_index"]   >= sample_idx)
    ]
    if covering.empty:
        return {"best_match": "N/A", "dtw_distance": np.nan}
    row = covering.loc[covering["dtw_distance"].idxmin()]
    return {"best_match": row["best_match"], "dtw_distance": row["dtw_distance"]}


def build_map(final_df: pd.DataFrame, labeled_wins: pd.DataFrame, out_path: str) -> None:
    """
    Stage 15 – Build interactive Folium route map.
    Red segments = aggressive, Green = non-aggressive.
    Clicking a point shows behaviour, coordinates, speed stats,
    aggression score, and DTW match info.
    """
    print(f"\n[Stage 15] Building interactive Folium map -> {out_path}")

    lat_col = "Location_latitude"
    lon_col = "Location_longitude"
    spd_col = "Location_speed"

    df = final_df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    if df.empty:
        print("  WARNING: No GPS data available for map.")
        return

    center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    # Draw route segments coloured by label
    COLOR = {"Aggressive": "red", "Non-Aggressive": "green"}

    # Group consecutive rows with the same label into segments
    # Use a speed-aggregation window of ~40 samples (1 s) for popup stats
    POPUP_WIN = 40

    prev_label = None
    seg_lats, seg_lons, seg_indices = [], [], []

    def flush_segment(lats, lons, idxs, label):
        if len(lats) < 2:
            return
        # Representative index at midpoint
        mid = idxs[len(idxs) // 2]
        seg_slice = df.iloc[max(0, mid - POPUP_WIN // 2): mid + POPUP_WIN // 2]
        spd       = seg_slice[spd_col] if spd_col in seg_slice.columns else pd.Series()
        dtw_info  = _window_meta_for_sample(mid, labeled_wins)
        agg_score = df.loc[mid, "aggression_score"] if "aggression_score" in df.columns else 0.0

        popup_html = textwrap.dedent(f"""
            <b>Driving Behaviour</b>: {label}<br>
            <b>Latitude</b>:  {lats[len(lats)//2]:.6f}<br>
            <b>Longitude</b>: {lons[len(lons)//2]:.6f}<br>
            <b>Min Speed</b>: {spd.min():.1f} km/h<br>
            <b>Max Speed</b>: {spd.max():.1f} km/h<br>
            <b>Avg Speed</b>: {spd.mean():.1f} km/h<br>
            <b>Aggression Score</b>: {agg_score:.3f}<br>
            <b>DTW Best Match</b>: {dtw_info['best_match']}<br>
            <b>DTW Distance</b>: {dtw_info['dtw_distance']:.2f}
        """).strip()

        folium.PolyLine(
            locations=list(zip(lats, lons)),
            color=COLOR.get(label, "blue"),
            weight=4,
            opacity=0.8,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(m)

    for i, row in df.iterrows():
        label = row["aggressive_label"]
        lat, lon = row[lat_col], row[lon_col]

        if label != prev_label:
            flush_segment(seg_lats, seg_lons, seg_indices, prev_label)
            seg_lats, seg_lons, seg_indices = [], [], []
            prev_label = label

        seg_lats.append(lat)
        seg_lons.append(lon)
        seg_indices.append(i)

    flush_segment(seg_lats, seg_lons, seg_indices, prev_label)

    # Start / end markers
    start_row = df.iloc[0]
    end_row   = df.iloc[-1]
    folium.Marker(
        location=[start_row[lat_col], start_row[lon_col]],
        popup="<b>Trip Start</b>",
        icon=folium.Icon(color="blue", icon="play"),
    ).add_to(m)
    folium.Marker(
        location=[end_row[lat_col], end_row[lon_col]],
        popup="<b>Trip End</b>",
        icon=folium.Icon(color="purple", icon="stop"),
    ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:40px;left:40px;z-index:1000;
                background:white;padding:12px;border-radius:6px;
                border:2px solid #ccc;font-size:13px;line-height:1.8;">
        <b>Legend</b><br>
        <span style="color:red;font-size:18px;">&#9644;</span> Aggressive<br>
        <span style="color:green;font-size:18px;">&#9644;</span> Non-Aggressive<br>
        <span style="color:blue;font-size:18px;">&#9679;</span> Trip Start<br>
        <span style="color:purple;font-size:18px;">&#9679;</span> Trip End
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(out_path)
    print(f"  Map saved -> {out_path}")


# ─── Stage 16: Summary report ─────────────────────────────────────────────────

def write_report(stats: dict, labeled_wins: pd.DataFrame,
                 final_df: pd.DataFrame, out_path: str) -> None:
    """Stage 16 – Write plaintext summary report."""
    # Top matching templates (most commonly matched in aggressive windows)
    agg_wins  = labeled_wins[labeled_wins["label"] == "Aggressive"]
    top_tmpl  = (agg_wins["best_match"].value_counts().head(5).to_string()
                 if not agg_wins.empty else "N/A")

    # Most aggressive segment (max aggression_score)
    lat_col, lon_col = "Location_latitude", "Location_longitude"
    if "aggression_score" in final_df.columns and lat_col in final_df.columns:
        peak_idx  = final_df["aggression_score"].idxmax()
        peak_row  = final_df.loc[peak_idx]
        peak_loc  = f"Lat {peak_row[lat_col]:.6f}, Lon {peak_row[lon_col]:.6f}"
        peak_score = f"{peak_row['aggression_score']:.3f}"
    else:
        peak_loc, peak_score = "N/A", "N/A"

    report = textwrap.dedent(f"""
    ============================================================
     AGGRESSIVE RIDING DETECTION – SUMMARY REPORT
    ============================================================

    TRIP DURATION
    -------------
    Total Trip Duration     : {stats['total_duration']}
    Aggressive Duration     : {stats['agg_duration']}
    Non-Aggressive Duration : {stats['non_agg_duration']}
    % Aggressive            : {stats['pct_aggressive']:.1f}%
    % Non-Aggressive        : {stats['pct_non_agg']:.1f}%

    AGGRESSIVE EPISODES
    --------------------
    Number of Episodes      : {stats['num_episodes']}
    Avg Episode Duration    : {stats['avg_episode_dur']}
    Max Episode Duration    : {stats['max_episode_dur']}

    TOP MATCHING AGGRESSIVE TEMPLATES
    -----------------------------------
    {top_tmpl}

    MOST AGGRESSIVE SEGMENT
    ------------------------
    Location                : {peak_loc}
    Aggression Score        : {peak_score}

    ============================================================
    Generated by Aggressive Riding Detection Pipeline
    ============================================================
    """).strip()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Stage 16] Summary report saved -> {out_path}")
    print()
    print(report)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(aggressive_zip: str, unlabeled_zip: str, out_dir: str = "output") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ── Stage 1: Sensor processing ────────────────────────────────────────────
    agg_raw_csv = process_zip(aggressive_zip, "aggressive", out_dir)
    unl_raw_csv = process_zip(unlabeled_zip,  "unlabeled",  out_dir)

    # ── Stage 2: Vehicle-frame transformation ─────────────────────────────────
    agg_vf_csv = compute_vehicle_frame(agg_raw_csv, "aggressive", out_dir)
    unl_vf_csv = compute_vehicle_frame(unl_raw_csv, "unlabeled",  out_dir)

    # Load vehicle-frame DataFrames
    print("\n[Stage 3] Computing jerk …")
    agg_df = pd.read_csv(agg_vf_csv)
    unl_df = pd.read_csv(unl_vf_csv)

    # ── Stage 3: Jerk ─────────────────────────────────────────────────────────
    agg_df = add_jerk(agg_df)
    unl_df = add_jerk(unl_df)

    # ── Stage 4: Aggressive windows ───────────────────────────────────────────
    print("\n[Stage 4] Generating aggressive reference windows …")
    agg_windows = create_windows(agg_df, "Agg")
    print(f"  Created {len(agg_windows)} aggressive windows")

    # ── Stage 5: Feature extraction ───────────────────────────────────────────
    print("\n[Stage 5] Extracting features from aggressive windows …")
    feat_rows = [extract_features(agg_df, w) for w in agg_windows]
    feat_df = pd.DataFrame(feat_rows)

    # ── Stage 6: Template selection ───────────────────────────────────────────
    print("\n[Stage 6] Selecting strong aggressive templates …")
    template_df = select_templates(feat_df)
    templates_csv = os.path.join(out_dir, "aggressive_templates.csv")
    template_df.to_csv(templates_csv, index=False)
    print(f"  Saved -> {templates_csv}")

    # Pre-compute template signals once
    template_wins = [
        {"window_id": row["window_id"],
         "start_index": int(row["start_index"]),
         "end_index": int(row["end_index"])}
        for _, row in template_df.iterrows()
    ]
    template_signals = [get_window_signal(agg_df, w) for w in template_wins]

    # ── Stage 7: Unlabeled windows ────────────────────────────────────────────
    print("\n[Stage 7] Creating windows from unlabeled trip …")
    unl_windows = create_windows(unl_df, "U")
    print(f"  Created {len(unl_windows)} unlabeled windows")

    # ── Stage 8: DTW matching ─────────────────────────────────────────────────
    match_df = match_windows(unl_df, unl_windows, agg_df, template_wins, template_signals)

    # ── Stage 9: Threshold estimation ─────────────────────────────────────────
    threshold, mean_d, std_d = estimate_threshold(agg_df, template_wins, template_signals)

    # ── Stage 10 + 11: Labels and similarity score ────────────────────────────
    print("\n[Stage 10 & 11] Assigning labels and computing similarity scores …")
    labeled_wins = assign_labels(match_df, threshold)
    n_agg = (labeled_wins["label"] == "Aggressive").sum()
    n_non = (labeled_wins["label"] == "Non-Aggressive").sum()
    print(f"  Aggressive windows: {n_agg} / Non-Aggressive: {n_non}")

    wins_csv = os.path.join(out_dir, "window_level_labels.csv")
    labeled_wins.to_csv(wins_csv, index=False)
    print(f"  Saved -> {wins_csv}")

    # ── Stage 12: Sample-level labels ─────────────────────────────────────────
    print("\n[Stage 12] Aggregating sample-level labels via voting …")
    sample_labels = sample_level_labels(labeled_wins, len(unl_df))

    # ── Stage 13: Final dataset ───────────────────────────────────────────────
    print("\n[Stage 13] Building final labeled dataset …")
    keep_cols = [
        "datetime_ist",
        "Location_latitude", "Location_longitude", "Location_speed",
        "Accel_vehicle_x", "Gyro_vehicle_z", "jerk",
    ]
    # Keep only columns that actually exist
    keep_cols = [c for c in keep_cols if c in unl_df.columns]
    final_df = unl_df[keep_cols].copy()
    final_df["aggression_score"] = sample_labels["aggression_score"].values
    final_df["aggressive_label"] = sample_labels["aggressive_label"].values

    final_csv = os.path.join(out_dir, "labeled_unlabeled_trip.csv")
    final_df.to_csv(final_csv, index=False)
    print(f"  Saved {len(final_df)} rows -> {final_csv}")

    # ── Stage 14: Summary statistics ──────────────────────────────────────────
    print("\n[Stage 14] Computing summary statistics …")
    stats = compute_summary(final_df)

    # ── Stage 15: Map ─────────────────────────────────────────────────────────
    map_path = os.path.join(out_dir, "aggressive_behavior_map.html")
    build_map(final_df, labeled_wins, map_path)

    # ── Stage 16: Report ──────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, "summary_report.txt")
    write_report(stats, labeled_wins, final_df, report_path)

    # ── Final output summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  OUTPUT FILES")
    print("=" * 60)
    outputs = [
        ("aggressive_merged_raw_data.csv", agg_raw_csv),
        ("aggressive_vehicle_frame.csv",   agg_vf_csv),
        ("unlabeled_merged_raw_data.csv",  unl_raw_csv),
        ("unlabeled_vehicle_frame.csv",    unl_vf_csv),
        ("aggressive_templates.csv",       templates_csv),
        ("window_level_labels.csv",        wins_csv),
        ("labeled_unlabeled_trip.csv",     final_csv),
        ("aggressive_behavior_map.html",   map_path),
        ("summary_report.txt",             report_path),
    ]
    for name, path in outputs:
        exists = "OK" if os.path.isfile(path) else "MISSING"
        print(f"  [{exists}] {name}")
    print("=" * 60)
    print("  Pipeline complete.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggressive Riding Detection Pipeline (DTW weak supervision)"
    )
    parser.add_argument(
        "aggressive_zip",
        help="ZIP file containing the known-aggressive reference trip",
    )
    parser.add_argument(
        "unlabeled_zip",
        help="ZIP file containing the unlabeled trip to analyse",
    )
    parser.add_argument(
        "--out-dir",
        default="output",
        help="Directory for all output files (default: output/)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.aggressive_zip):
        sys.exit(f"ERROR: aggressive ZIP not found: {args.aggressive_zip}")
    if not os.path.isfile(args.unlabeled_zip):
        sys.exit(f"ERROR: unlabeled ZIP not found: {args.unlabeled_zip}")

    run_pipeline(args.aggressive_zip, args.unlabeled_zip, args.out_dir)
