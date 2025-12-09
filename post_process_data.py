#!/usr/bin/env python3
"""
Post-process NPZ data files to split samples by heart rate changes.
Finds contiguous blocks with the same heart rate and reports block size statistics.
"""

import numpy as np
import os
from glob import glob
from typing import List, Tuple

# ============== CONFIGURATION ==============
# FOLDER = "noise_baseline_20251209_DM-1_NS-2"
# FOLDER = "hr_20251209_DM-1_NS-2"
FOLDER = "hr-2_20251209_DM-1_NS-2"
DATA_DIR = "data"  # Base data directory

# Tolerance for considering heart rates as "the same" (bpm)
HR_TOLERANCE = 0.1 # Consider heart rates within x bpm as the same
# ==========================================

def find_heart_rate_blocks(labels: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Find contiguous blocks with the same heart rate.
    
    Args:
        labels: 1D array of heart rate labels (shape: (T,) or (1, T))
    
    Returns:
        List of tuples (start_idx, end_idx, heart_rate) for each block
    """
    # Flatten if needed (handle both 1D and 2D arrays)
    if labels.ndim > 1:
        labels = labels.flatten()
    
    if len(labels) == 0:
        return []
    
    blocks = []
    start_idx = 0
    current_hr = labels[0]
    
    # Handle NaN values - treat each NaN as its own block
    for i in range(1, len(labels)):
        hr = labels[i]
        
        # Check if heart rate changed (accounting for NaN and tolerance)
        hr_changed = False
        
        if np.isnan(current_hr) and not np.isnan(hr):
            hr_changed = True
        elif not np.isnan(current_hr) and np.isnan(hr):
            hr_changed = True
        elif not np.isnan(current_hr) and not np.isnan(hr):
            if abs(hr - current_hr) > HR_TOLERANCE:
                hr_changed = True
        
        if hr_changed:
            # Save current block
            blocks.append((start_idx, i, current_hr))
            start_idx = i
            current_hr = hr
    
    # Add final block
    blocks.append((start_idx, len(labels), current_hr))
    
    return blocks

def process_file(filepath: str) -> Tuple[List[Tuple[int, int, float]], int]:
    """
    Process a single NPZ file and return heart rate blocks.
    
    Args:
        filepath: Path to NPZ file
    
    Returns:
        Tuple of (blocks, total_samples) where blocks is list of (start, end, hr)
    """
    try:
        data = np.load(filepath, allow_pickle=True)
        
        if 'labels' not in data:
            print(f"  Warning: {filepath} missing 'labels' field")
            return [], 0
        
        labels = data['labels']
        total_samples = labels.size
        
        # Find blocks
        blocks = find_heart_rate_blocks(labels)
        
        return blocks, total_samples
        
    except Exception as e:
        print(f"  Error processing {filepath}: {e}")
        return [], 0

def get_block_sizes(blocks: List[Tuple[int, int, float]]) -> List[int]:
    """Extract block sizes from blocks list."""
    return [end - start for start, end, _ in blocks]

def format_hr(hr: float) -> str:
    """Format heart rate for display."""
    if np.isnan(hr):
        return "NaN"
    return f"{hr:.2f}"

def main():
    # Construct full folder path
    folder_path = os.path.join(DATA_DIR, FOLDER)
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    # Find all NPZ files in folder
    pattern = os.path.join(folder_path, "*.npz")
    files = sorted(glob(pattern))
    
    if len(files) == 0:
        print(f"Error: No NPZ files found in '{folder_path}'")
        return
    
    print(f"Processing {len(files)} files from '{folder_path}'")
    print(f"Heart rate tolerance: {HR_TOLERANCE} bpm")
    print("-" * 80)
    
    all_block_sizes = []
    file_stats = []
    
    # Process each file
    for filepath in files:
        filename = os.path.basename(filepath)
        blocks, total_samples = process_file(filepath)
        
        if len(blocks) == 0:
            continue
        
        block_sizes = get_block_sizes(blocks)
        all_block_sizes.extend(block_sizes)
        
        # Store file statistics
        file_stats.append({
            'filename': filename,
            'total_samples': total_samples,
            'num_blocks': len(blocks),
            'block_sizes': block_sizes,
            'blocks': blocks
        })
        
        # Print file summary
        print(f"\nFile: {filename}")
        print(f"  Total samples: {total_samples}")
        print(f"  Number of blocks: {len(blocks)}")
        
        if len(blocks) > 1:
            print(f"  Blocks:")
            for start, end, hr in blocks:
                size = end - start
                print(f"    [{start:6d}:{end:6d}] HR={format_hr(hr):>8s} bpm, size={size:6d} samples")
        else:
            start, end, hr = blocks[0]
            print(f"  Single block: HR={format_hr(hr)} bpm")
        
        if len(block_sizes) > 0:
            print(f"  Block sizes: min={min(block_sizes)}, max={max(block_sizes)}, mean={np.mean(block_sizes):.1f}")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    if len(all_block_sizes) == 0:
        print("No blocks found!")
        return
    
    min_block_size = min(all_block_sizes)
    max_block_size = max(all_block_sizes)
    mean_block_size = np.mean(all_block_sizes)
    median_block_size = np.median(all_block_sizes)
    
    print(f"Total blocks found: {len(all_block_sizes)}")
    print(f"Minimum block size: {min_block_size} samples")
    print(f"Maximum block size: {max_block_size} samples")
    print(f"Mean block size: {mean_block_size:.1f} samples")
    print(f"Median block size: {median_block_size:.1f} samples")
    
    # Show distribution
    if len(all_block_sizes) > 1:
        print(f"\nBlock size distribution:")
        print(f"  Std deviation: {np.std(all_block_sizes):.1f} samples")
        print(f"  Percentiles:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(all_block_sizes, p)
            print(f"    {p:2d}th: {val:.1f} samples")

if __name__ == "__main__":
    main()
