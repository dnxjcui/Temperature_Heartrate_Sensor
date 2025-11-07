#!/usr/bin/env python3
"""
Compact temperature recorder - FIXED VERSION
- Reads variable-rate Fahrenheit data from a serial device
- Uses HOST-SIDE timestamps to avoid Arduino millis() overflow issues
- Resamples to fixed rate using SciPy interpolation
- Saves NPZ with fields: time (1xT), data_F (1xT), labels (1xT; NaN for baseline)
"""

import gc
import json
import os
import sys
import time
from datetime import datetime
from collections import deque
from typing import Optional, Tuple
import warnings

import numpy as np
from scipy.interpolate import interp1d
import serial  # pip install pyserial
import config

# ============== CONFIGURATION ==============
# Output settings
OUTPUT_FOLDER = "noise_baseline_1107_20251107"  # Output folder for NPZ files
DURATION_MINUTES = 1  # Per-sample duration (minutes)
NUM_SAMPLES = 20  # How many samples to record
SAMPLE_RATE = 200.0  # Fixed resample rate (Hz)
LABEL = False  # Label flag (currently unused in data, baseline → NaN labels)
WARMUP_SEC = 0.5  # Discard initial seconds each run

DEBUG = True
# ==========================================

# Serial communication settings
PORT = config.PORT  # Serial port
BAUD_RATE = config.BAUD_RATE  # Baud rate

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

def parse_line(line: str) -> Optional[float]:
    """
    Parse temperature from serial line. Returns temperature in Fahrenheit.
    Ignores timestamps from Arduino as we'll use host-side timing.
    
    Try JSON → CSV → single number.
    """
    s = line.strip()
    
    # Skip empty lines and header lines
    if not s or s.startswith('Board') or s.startswith('Time') or s.startswith('Adafruit'):
        return None
    
    # Try JSON first
    if s.startswith('{'):
        try:
            data = json.loads(s)
            temp_f = data.get('temp_f') or data.get('tempF') or data.get('fahrenheit')
            if temp_f is not None:
                return float(temp_f)
        except:
            pass
    
    # CSV format: millis(),objTempC,objTempF,avgTempF
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        nums = [_float_or_none(p) for p in parts]
        
        # Get temperature (prefer 3rd column = objTempF, else last non-None)
        if len(nums) >= 3 and nums[2] is not None:
            return nums[2]
        elif len(nums) >= 4 and nums[3] is not None:
            return nums[3]  # Use avgTempF if objTempF not available
        elif len(nums) >= 2:
            # Get last non-None value
            for val in reversed(nums[1:]):  # Skip timestamp
                if val is not None:
                    return val
    
    # Single numeric
    tf = _float_or_none(s)
    if tf is not None:
        return tf

    return None

# -------------------------- Fast Serial Reader --------------------------

class FastSerialReader:
    """Optimized serial reader with buffering"""
    
    def __init__(self, ser):
        self.ser = ser
        self.buffer = ""
        self.lines = deque()
        
    def read_available_lines(self):
        """Read all available data and return complete lines"""
        # Read all available bytes at once
        if self.ser.in_waiting > 0:
            try:
                new_data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                self.buffer += new_data
                
                # Extract complete lines
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    self.lines.append(line.strip())
                    
            except Exception as e:
                print(f"Read error: {e}")
                
        # Return all complete lines
        result = list(self.lines)
        self.lines.clear()
        return result

# -------------------------- Acquisition & resampling --------------------------

def acquire_block(ser, seconds: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Collect (t, F) samples for ~seconds using HOST-SIDE timestamps.
    This avoids Arduino millis() overflow issues.
    Returns two numpy arrays (t, F) with strictly increasing t.
    Timestamps start at 0 (relative to first sample).
    """
    t_buf = deque()
    f_buf = deque()
    
    # Use high-resolution timer for accurate timing
    t_start = time.perf_counter()
    t_end = t_start + seconds
    t0 = None  # First sample timestamp
    
    # Flush any stale data
    ser.reset_input_buffer()
    
    # Create fast reader
    reader = FastSerialReader(ser)
    
    sample_count = 0
    lines_processed = 0
    
    while time.perf_counter() < t_end:
        # Read all available lines at once (batch processing)
        lines = reader.read_available_lines()
        
        for line in lines:
            lines_processed += 1
            
            # Parse temperature
            tf = parse_line(line)
            if tf is None:
                continue
            
            # Validate temperature range
            if not (-4 <= tf <= 212):
                if DEBUG and sample_count < 5:  # Only print first few rejections
                    print(f"Rejected bad temp: {tf}°F")
                continue
            
            # Check for NaN or Inf
            if not np.isfinite(tf):
                continue
            
            # Record HOST timestamp for this sample
            t_now = time.perf_counter()
            
            # Initialize first timestamp
            if t0 is None:
                t0 = t_now
            
            # Calculate relative time
            t_rel = t_now - t0
            
            # Store sample
            t_buf.append(t_rel)
            f_buf.append(tf)
            sample_count += 1
        
        # Small sleep if no data to prevent CPU spinning
        if not lines:
            time.sleep(0.0001)  # 100 microseconds

    actual_duration = time.perf_counter() - t_start
    
    if not t_buf:
        warnings.warn("No samples collected!")
        return None, None
    
    if DEBUG: 
        print(f"Collected {sample_count} samples in {actual_duration:.2f}s")
    
    # Convert to numpy arrays
    t_arr = np.array(t_buf, dtype=np.float64)
    f_arr = np.array(f_buf, dtype=np.float64)
    
    # Verify monotonic timestamps (should always be true with host timing)
    if not np.all(np.diff(t_arr) > 0):
        warnings.warn("Non-monotonic timestamps detected!")
        # Fix by keeping only monotonic samples
        mask = np.concatenate([[True], np.diff(t_arr) > 0])
        t_arr = t_arr[mask]
        f_arr = f_arr[mask]
    
    return t_arr, f_arr

def resample_to_fixed_rate(t_arr, f_arr, fs, target_duration=None):
    """
    Irregular → regular resampling via SciPy interpolation.
    Produces time vector starting at 0 with step 1/fs, and data_F aligned to it.
    """
    if t_arr.size < 2:
        return np.array([[]], dtype=float), np.array([[]], dtype=float)

    # t_arr already starts at 0 from acquire_block
    T = t_arr[-1] - t_arr[0]
    if T <= 0:
        return np.array([[]], dtype=float), np.array([[]], dtype=float)
    
    # Use target duration if provided
    if target_duration is not None:
        T = target_duration

    # Regular grid
    dt = 1.0 / float(fs)
    t_uniform = np.arange(0.0, T + 1e-12, dt)

    # Interpolator
    f = interp1d(t_arr, f_arr, kind="linear", bounds_error=False,
                 fill_value=(f_arr[0], f_arr[-1]))
    f_uniform = f(t_uniform)

    # Shape to 1xT
    return t_uniform[np.newaxis, :], f_uniform[np.newaxis, :]

# -------------------------- Main --------------------------

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=False if not DEBUG else True)
    label_flag = bool(LABEL)

    # Save configuration
    config_dict = {
        "OUTPUT_FOLDER": OUTPUT_FOLDER,
        "DURATION_MINUTES": DURATION_MINUTES,
        "NUM_SAMPLES": NUM_SAMPLES,
        "SAMPLE_RATE": SAMPLE_RATE,
        "LABEL": LABEL,
        "PORT": PORT,
        "BAUD_RATE": BAUD_RATE,
        "TIMESTAMP_METHOD": "HOST_SIDE",
    }

    with open(os.path.join(OUTPUT_FOLDER, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    # Open serial with short timeout for non-blocking reads
    ser = serial.Serial(PORT, BAUD_RATE, timeout=0.001)  # Very short timeout
    
    print(f"Waiting for Arduino to initialize...")
    time.sleep(2)  # Wait for Arduino reset
    ser.reset_input_buffer()  # Clear startup messages
    
    # Test read to verify connection
    print("Testing connection...")
    test_start = time.time()
    test_lines = 0
    while time.time() - test_start < 0.5:
        if ser.in_waiting > 0:
            line = ser.readline()
            if line:
                test_lines += 1
    print(f"Connection test: received {test_lines} lines in 0.5s")

    try:
        for idx in range(NUM_SAMPLES):
            print(f"\n[{idx+1}/{NUM_SAMPLES}] Recording sample {idx+1} of {NUM_SAMPLES}...")
            
            # Optional warmup
            if WARMUP_SEC > 0:
                _ = acquire_block(ser, WARMUP_SEC)

            # Acquire one block
            block_seconds = float(DURATION_MINUTES) * 60.0
            print(f"  Recording {block_seconds:.1f}s of data...")
            
            t_raw, f_raw = acquire_block(ser, block_seconds)
            
            if t_raw is None or f_raw is None:
                print(f"  ERROR: No data captured, retrying...")
                retries = 0
                while t_raw is None or f_raw is None:
                    time.sleep(0.5)  # Brief pause
                    t_raw, f_raw = acquire_block(ser, block_seconds)
                    retries += 1
                    if retries > 3:
                        print(f"  FATAL: Failed after {retries} retries")
                        sys.exit(1)

            # Resample to fixed rate
            target_duration = float(DURATION_MINUTES) * 60.0
            t_grid, f_grid = resample_to_fixed_rate(t_raw, f_raw, SAMPLE_RATE, target_duration)
            
            if t_grid.size == 0:
                print(f"  ERROR: Resampling failed; skipping.")
                continue

            # Labels: baseline → NaN
            labels = np.full_like(f_grid, np.nan, dtype=float)

            # File naming
            fname = (
                f"tempF_dur-{DURATION_MINUTES:.2f}min_"
                f"fs-{int(SAMPLE_RATE)}Hz_"
                f"idx-{idx:03d}_"
                f"label-{'T' if label_flag else 'F'}.npz"
            )
            path = os.path.join(OUTPUT_FOLDER, fname)

            # Save NPZ
            np.savez_compressed(path, time=t_grid, data_F=f_grid, labels=labels)
            T = t_grid.shape[1]
            print(f" Saved {path}, {T} samples, Temp range: {f_grid.min():.1f}°F to {f_grid.max():.1f}°F")
            
            # Cleanup
            del t_raw, f_raw, t_grid, f_grid, labels
            gc.collect()
            
            # Brief pause between samples
            if idx < NUM_SAMPLES - 1:
                time.sleep(0.5)

    finally:
        try:
            ser.close()
            print("\nSerial port closed.")
        except Exception:
            pass
        print(f"\nData collection complete. Files saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()