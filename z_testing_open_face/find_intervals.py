import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import your ground truth dictionaries/lists safely
try:
    from ground_truth import GROUND_TRUTH_INTERVALS_249, GROUND_TRUTH_INTERVALS_236
except ImportError:
    pass

# =========================================================
# 1. DATA PROCESSING & DETECTION FUNCTIONS
# =========================================================
def parse_frame_string(val):
    """Extracts integer frame numbers from strings like '(np.int64(82), np.int64(223))'"""
    matches = re.findall(r'int64\((\d+)\)', str(val))
    if len(matches) == 2:
        return int(matches[0]), int(matches[1])
    return None, None

def extract_intervals_from_csv(csv_path, column_name, fps=30.0):
    """Loads a CSV and converts the frame string column into time intervals (seconds)."""
    if not os.path.exists(csv_path):
        print(f"Warning: File not found -> {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return []

    intervals_sec = []
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in {csv_path}")
        return intervals_sec

    for index, row in df.iterrows():
        start_frame, end_frame = parse_frame_string(row[column_name])
        if start_frame is not None and end_frame is not None:
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            intervals_sec.append((start_sec, end_sec))
            
    return intervals_sec

def find_smart_intervals(timestamps, signal, peak_thresh=2.0, base_thresh=0.5):
    """
    Finds local maxima > peak_thresh, then expands to base_thresh to find start/end boundaries.
    """
    intervals = []
    n = len(signal)
    deriv = np.gradient(signal)
    
    # 1. Find indices of all local maxima (Zero-crossing of derivative)
    peaks = []
    for i in range(1, n - 1):
        if deriv[i-1] > 0 and deriv[i] <= 0:
            if signal[i] >= peak_thresh:
                peaks.append(i)
                
    # 2. For each peak, expand outward to find the start and end of the smile
    used_indices = np.zeros(n, dtype=bool)
    for p_idx in peaks:
        if used_indices[p_idx]:
            continue
            
        # Walk backward to find start
        start_idx = p_idx
        while start_idx > 0 and signal[start_idx] > base_thresh:
            start_idx -= 1
            
        # Walk forward to find end
        end_idx = p_idx
        while end_idx < n - 1 and signal[end_idx] > base_thresh:
            end_idx += 1
            
        intervals.append((timestamps[start_idx], timestamps[end_idx]))
        used_indices[start_idx:end_idx+1] = True
            
    return intervals

# =========================================================
# 2. MAIN VISUALIZATION FUNCTION
# =========================================================
# --- CHANGED: ground_truth defaults to None ---
def visualize_smile_events(video_id, ground_truth=None, fps=30.0, 
                           window_size=10, 
                           start_time=None,  
                           end_time=None,
                           peak_thresh =2.0,
                           base_thresh=0.5,    
                           raw_base_dir='/home/amir/Ados_lab/videos/openface_marisa',
                           out_base_dir='/home/amir/Ados_lab/out'):
    """
    Dynamically loads data, applies signal processing, extracts smart intervals,
    and plots the entire pipeline.
    """
    # --- CHANGED: Safely handle missing ground truth ---
    ground_truth = ground_truth or []
    
    print(f"\n--- Processing Video {video_id} ---")
    
    # Construct dynamic file paths based on the video ID
    raw_csv = f"{raw_base_dir}/{video_id}/{video_id}_Emotions_marissa.csv"
    csv_simple = f"{out_base_dir}/{video_id}/FeaturesAU/emo/SimpleSmiling/{video_id}_emo_I.csv"
    csv_combo = f"{out_base_dir}/{video_id}/FeaturesAU/emo/ComboSmiling/{video_id}_emo_I.csv"

    # 1. Load Extracted Intervals
    print("Loading Interval Data...")
    simple_extracted = extract_intervals_from_csv(csv_simple, column_name='peak_lp_simple', fps=fps)
    combo_extracted = extract_intervals_from_csv(csv_combo, column_name='peak_lp_combo', fps=fps)
    
    # 2. Load Raw OpenFace Signal
    print("Loading Raw OpenFace Signal...")
    if not os.path.exists(raw_csv):
        print(f"Error: Could not find raw signal file at {raw_csv}")
        return 
        
    df_raw = pd.read_csv(raw_csv)
    df_raw.columns = df_raw.columns.str.strip()  
    timestamps = df_raw['timestamp']

    # --- MATH STEP 1: MULTIPLY THE RAW SIGNALS ---
    if 'AU12_r' in df_raw.columns and 'AU06_r' in df_raw.columns:
        df_raw['Raw_Multiplied'] = df_raw['AU12_r'] * df_raw['AU06_r']

    # --- MATH STEP 2: APPLY LOW PASS FILTER TO ALL SIGNALS ---
    print(f"Applying Low Pass Filter (Window={window_size})...")
    if 'AU12_r' in df_raw.columns:
        df_raw['AU12_r_smooth'] = df_raw['AU12_r'].rolling(window=window_size, center=True, min_periods=1).mean()
    if 'AU06_r' in df_raw.columns:
        df_raw['AU06_r_smooth'] = df_raw['AU06_r'].rolling(window=window_size, center=True, min_periods=1).mean()
    if 'Raw_Multiplied' in df_raw.columns:
        df_raw['Multiply_Then_Smooth'] = df_raw['Raw_Multiplied'].rolling(window=window_size, center=True, min_periods=1).mean()

    # --- MATH STEP 3: MULTIPLY THE ALREADY-SMOOTHED SIGNALS ---
    if 'AU12_r_smooth' in df_raw.columns and 'AU06_r_smooth' in df_raw.columns:
        df_raw['Smooth_Then_Multiply'] = df_raw['AU12_r_smooth'] * df_raw['AU06_r_smooth']

    # --- MATH STEP 4: SMART INTERVAL DETECTION ---
    print("Detecting Smart Intervals...")
    smart_intervals = []
    if 'Smooth_Then_Multiply' in df_raw.columns:
        smart_intervals = find_smart_intervals(
            timestamps.values, 
            df_raw['Smooth_Then_Multiply'].values, 
            peak_thresh, 
            base_thresh
        )

    # 5. Setup the 6-Panel Plot
    print("Generating Plot...")
    fig, axs = plt.subplots(6, 1, figsize=(14, 18), sharex=True, gridspec_kw={'height_ratios': [1.5, 1.5, 1.5, 1.5, 1.5, 1]})
    
    title_suffix = f" (Zoomed: {start_time}s to {end_time}s)" if (start_time or end_time) else " (Full Video)"
    gt_suffix = " (No Ground Truth)" if not ground_truth else ""
    fig.suptitle(f'Smart Smile Detection Analysis - Video {video_id}{title_suffix}{gt_suffix}', fontsize=16, fontweight='bold', y=0.98)

    # --- Subplot 1: COMBINED RAW SIGNALS ---
    if 'AU12_r' in df_raw.columns and 'AU06_r' in df_raw.columns:
        axs[0].plot(timestamps, df_raw['AU12_r'], color='blue', alpha=0.5, label='Raw AU12_r')
        axs[0].plot(timestamps, df_raw['AU06_r'], color='red', alpha=0.5, label='Raw AU06_r')
        axs[0].set_ylabel('Intensity (0-5)')
        axs[0].set_title('Raw Signals Combined: AU12_r & AU06_r')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='upper right')
        for gt_start, gt_end in ground_truth:
            axs[0].axvspan(gt_start, gt_end, color='lime', alpha=0.2)

    # --- Subplot 2: COMBINED SMOOTHED SIGNALS ---
    if 'AU12_r_smooth' in df_raw.columns and 'AU06_r_smooth' in df_raw.columns:
        axs[1].plot(timestamps, df_raw['AU12_r_smooth'], color='darkblue', linewidth=2, label='Smoothed AU12_r')
        axs[1].plot(timestamps, df_raw['AU06_r_smooth'], color='darkred', linewidth=2, label='Smoothed AU06_r')
        axs[1].set_ylabel('Intensity (0-5)')
        axs[1].set_title(f'Smoothed Individual Signals (Window={window_size})')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc='upper right')
        for gt_start, gt_end in ground_truth:
            axs[1].axvspan(gt_start, gt_end, color='lime', alpha=0.2)

    # --- Subplot 3: RAW MULTIPLIED SIGNAL ---
    if 'Raw_Multiplied' in df_raw.columns:
        axs[2].plot(timestamps, df_raw['Raw_Multiplied'], color='magenta', alpha=0.6, label='Raw AU12 × Raw AU06')
        axs[2].set_ylabel('Intensity (0-25)')
        axs[2].set_title('Multiplied Raw Signals (Very Noisy)')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc='upper right')

    # --- Subplot 4: MULTIPLY THEN SMOOTH ---
    if 'Multiply_Then_Smooth' in df_raw.columns:
        axs[3].plot(timestamps, df_raw['Multiply_Then_Smooth'], color='olive', linewidth=2, label='Smoothed (Raw_Multiplied)')
        axs[3].set_ylabel('Intensity (0-25)')
        axs[3].set_title(f'Method 1: MULTIPLY First, THEN Smooth')
        axs[3].grid(True, alpha=0.3)
        axs[3].legend(loc='upper right')

    # --- Subplot 5: SMOOTH THEN MULTIPLY + SMART INTERVALS ---
    if 'Smooth_Then_Multiply' in df_raw.columns:
        axs[4].plot(timestamps, df_raw['Smooth_Then_Multiply'], color='purple', linewidth=2, label='AU12_smooth × AU06_smooth')
        axs[4].axhline(y=peak_thresh, color='black', linestyle='--', alpha=0.5, label=f'Peak Threshold ({peak_thresh})')
        axs[4].axhline(y=base_thresh, color='gray', linestyle='--', alpha=0.5, label=f'Base Threshold ({base_thresh})')
        
        # Draw the detected intervals right on the signal
        for s, e in smart_intervals:
            axs[4].axvspan(s, e, color='purple', alpha=0.15)
            axs[4].axvline(x=s, color='green', linestyle=':', alpha=0.7) # Start line
            axs[4].axvline(x=e, color='red', linestyle=':', alpha=0.7)   # End line

        axs[4].set_ylabel('Intensity (0-25)')
        axs[4].set_title('Method 2: SMOOTH First, THEN Multiply (with Smart Interval Bounds)')
        axs[4].grid(True, alpha=0.3)
        axs[4].legend(loc='upper right')

    # --- Subplot 6: Event Timeline ---
    for i, (start, end) in enumerate(ground_truth):
        axs[5].hlines(y=3, xmin=start, xmax=end, color='limegreen', linewidth=15)
        
    for i, (start, end) in enumerate(smart_intervals):
        axs[5].hlines(y=2, xmin=start, xmax=end, color='darkviolet', linewidth=15)

    for i, (start, end) in enumerate(simple_extracted):
        axs[5].hlines(y=1, xmin=start, xmax=end, color='royalblue', linewidth=15)

    for i, (start, end) in enumerate(combo_extracted):
        axs[5].hlines(y=0, xmin=start, xmax=end, color='darkorange', linewidth=15)

    axs[5].set_title('Event Detection Timeline')
    axs[5].set_xlabel('Timestamp (seconds)', fontsize=12)
    axs[5].set_yticks([0, 1, 2, 3])
    axs[5].set_yticklabels(['Legacy Combo', 'Legacy Simple', 'Smart Peak', 'Ground Truth'])
    axs[5].set_ylim(-0.8, 3.8)
    axs[5].grid(axis='x', linestyle='--', alpha=0.7)

    # --- APPLY X-AXIS ZOOM LIMITS ---
    min_x = start_time if start_time is not None else timestamps.min()
    max_x = end_time if end_time is not None else timestamps.max()
    plt.xlim(min_x, max_x)
    
    # Clean up layout and display
    plt.tight_layout()
    plt.show()

# =========================================================
# 3. EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    
    # Example: Zoom in on a specific section to see the peaks clearly
    visualize_smile_events(
        video_id="236", 
        ground_truth=GROUND_TRUTH_INTERVALS_236, # Works perfectly with None now!
        fps=30.0,
        window_size=10,
        start_time=0,  
        end_time=100.0,
        peak_thresh=2.0,
        base_thresh=0.5
    )

# it's possible to teach Fully Connected Network to teach what is a smile and what is not.
# later we can adjust those hyper parameters according to the prediction of the network. 
# For example, we can use the network to predict the probability of smiling at each frame,
# and then use those probabilities to set dynamic thresholds for peak detection and for base thresholds.
