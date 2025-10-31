#!/usr/bin/env python3
"""
Compact temperature recorder:
- Reads variable-rate Fahrenheit data from a serial device
- Resamples to fixed rate using SciPy interpolation
- Saves NPZ with fields: time (1xT), data_F (1xT), labels (1xT; NaN for baseline)
- File name encodes folder, duration, fs, sample index, label, and timestamp

Assumptions about incoming serial lines (robust parsing):
1) JSON: {"temp_f": 98.6, "timestamp": 1730400000.123} (keys also accepted: tempF, fahrenheit)
2) CSV:  timestamp,tempC,tempF,avgF   (we use tempF if present; else last value)
3) Single numeric: "98.6" (we timestamp on host time)
"""

import json
import os
import time
from datetime import datetime
from collections import deque

import numpy as np
from scipy.interpolate import interp1d
import serial  # pip install pyserial

# ============== CONFIGURATION ==============
# Output settings
OUTPUT_FOLDER = "noise_baseline_1031_20251031"  # Output folder for NPZ files
DURATION_MINUTES = 1  # Per-sample duration (minutes)
NUM_SAMPLES = 1  # How many samples to record
SAMPLE_RATE = 200.0  # Fixed resample rate (Hz)
LABEL = False  # Label flag (currently unused in data, baseline → NaN labels)

# Serial communication settings
PORT = 'COM6'  # Serial port (Windows: 'COM6', Mac/Linux: '/dev/ttyUSB0' or '/dev/cu.usbserial-*')
BAUD_RATE = 115200  # Baud rate
WARMUP_SEC = 0.5  # Discard initial seconds each run
# ==========================================

if OUTPUT_FOLDER is None:
    OUTPUT_FOLDER = os.path.join("data", datetime.now().strftime("%Y%m%d-%H%M%S"))
else:
    OUTPUT_FOLDER = os.path.join("data", OUTPUT_FOLDER)

# -------------------------- Parsing helpers --------------------------

def _float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None

def parse_line(line: str):
    """
    Try JSON → CSV → single number.
    Returns (t_host, temp_F) where t_host is time.time() if device timestamp missing.
    """
    s = line.strip()
    now = time.time()
    
    # CSV
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        nums = [ _float_or_none(p) for p in parts ]
        if any(n is not None for n in nums):
            # Heuristics: if 3+ columns, prefer the third as tempF; else last numeric
            if len(nums) >= 3 and nums[2] is not None:
                temp_f = nums[2]
            else:
                # last non-None
                temp_f = next((x for x in reversed(nums) if x is not None), None)
            # timestamp likely first column if plausible (seconds since epoch)
            t_candidate = nums[0] if (len(nums) > 0 and nums[0] and nums[0] > 1e9) else None
            return (t_candidate if t_candidate is not None else now, temp_f)

    # Single numeric
    tf = _float_or_none(s)
    if tf is not None:
        return (now, tf)

    return (None, None)

# -------------------------- Acquisition & resampling --------------------------

def acquire_block(ser, seconds: float):
    """
    Collect (t, F) samples for ~seconds. t is host/device timestamp (float, seconds).
    Returns two numpy arrays (t, F) with strictly increasing t (duplicates dropped).
    """
    t_buf, f_buf = deque(), deque()
    t_end = time.time() + seconds
    last_t = -np.inf

    while time.time() < t_end:
        try:
            raw = ser.readline().decode("utf-8", errors="ignore")
        except Exception:
            continue
        t, tf = parse_line(raw)
        if t is None or tf is None:
            continue
        # enforce monotonic time (drop duplicates/out-of-order)
        if t > last_t:
            t_buf.append(t)
            f_buf.append(tf)
            last_t = t

    if not t_buf:
        return np.array([]), np.array([])

    t_arr = np.fromiter(t_buf, dtype=float)
    f_arr = np.fromiter(f_buf, dtype=float)
    return t_arr, f_arr

def resample_to_fixed_rate(t_arr, f_arr, fs):
    """
    Irregular → regular resampling via SciPy interpolation.
    Produces time vector starting at 0 with step 1/fs, and data_F aligned to it.
    """
    if t_arr.size < 2:
        return np.array([[]], dtype=float), np.array([[]], dtype=float)

    # Normalize time to start at 0
    t0 = t_arr[0]
    t_rel = t_arr - t0
    T = t_rel[-1]
    if T <= 0:
        return np.array([[]], dtype=float), np.array([[]], dtype=float)

    # Regular grid
    dt = 1.0 / float(fs)
    t_uniform = np.arange(0.0, T + 1e-12, dt)

    # Interpolator (linear; fill with nearest-edge values to avoid NaNs at ends)
    f = interp1d(t_rel, f_arr, kind="linear", bounds_error=False,
                 fill_value=(f_arr[0], f_arr[-1]))
    f_uniform = f(t_uniform)

    # Shape to 1xT
    return t_uniform[np.newaxis, :], f_uniform[np.newaxis, :]

# -------------------------- Main --------------------------

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=False)
    label_flag = bool(LABEL)  # currently unused in data (baseline → NaN labels)

    # Open serial
    ser = serial.Serial(PORT, BAUD_RATE, timeout=0.5)
    time.sleep(0.2)  # brief settle

    try:
        for idx in range(NUM_SAMPLES):
            print(f"[{idx+1}/{NUM_SAMPLES}] Recording sample {idx+1} of {NUM_SAMPLES}...")
            # Optional quick warmup to flush stale lines
            if WARMUP_SEC > 0:
                _ = acquire_block(ser, WARMUP_SEC)

            # Acquire one block
            block_seconds = float(DURATION_MINUTES) * 60.0
            t_raw, f_raw = acquire_block(ser, block_seconds)

            if t_raw.size < 2:
                print(f"[{idx+1}/{NUM_SAMPLES}] No usable data captured; skipping.")
                continue

            # Resample
            t_grid, f_grid = resample_to_fixed_rate(t_raw, f_raw, SAMPLE_RATE)
            if t_grid.size == 0:
                print(f"[{idx+1}/{NUM_SAMPLES}] Resampling failed; skipping.")
                continue

            # Labels: baseline → NaN
            labels = np.full_like(f_grid, np.nan, dtype=float)

            # File naming pattern
            # ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            fname = (
                f"tempF_dur-{int(DURATION_MINUTES)}min_"
                f"fs-{int(SAMPLE_RATE)}Hz_"
                f"idx-{idx:03d}_"
                f"label-{'T' if label_flag else 'F'}.npz"
            )
            path = os.path.join(OUTPUT_FOLDER, fname)

            # Save NPZ
            np.savez_compressed(path, time=t_grid, data_F=f_grid, labels=labels)
            T = t_grid.shape[1]
            print(f"[{idx+1}/{NUM_SAMPLES}] Saved {path}  (T={T} samples)")

    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
