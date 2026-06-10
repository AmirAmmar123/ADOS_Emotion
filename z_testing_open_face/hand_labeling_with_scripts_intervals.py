import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Import your ground truth dictionaries/lists
# (Assuming ground_truth.py exists in your directory)
try:
    from ground_truth import GROUND_TRUTH_INTERVALS_249, GROUND_TRUTH_INTERVALS_236
except ImportError:
    pass # If the file doesn't exist yet, we just pass

# =========================================================
# 1. DATA PROCESSING FUNCTIONS
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

# =========================================================
# 2. MAIN VISUALIZATION FUNCTION
# =========================================================
def visualize_smile_events(video_id, ground_truth=None, fps=30.0, 
                           window_size=10, 
                           start_time=None,  
                           end_time=None,    
                           raw_base_dir='/home/amir/Ados_lab/videos/openface_marisa',
                           out_base_dir='/home/amir/Ados_lab/out'):
    """
    Dynamically loads data for a specific video ID and plots the raw AU signals, 
    smoothed signals, multiplied signals, extracted intervals, and human ground truth.
    Allows zooming into a specific time window.
    """
    # --- NEW: Safely handle missing ground truth ---
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
    
    # Smooth the individual raw signals
    if 'AU12_r' in df_raw.columns:
        df_raw['AU12_r_smooth'] = df_raw['AU12_r'].rolling(window=window_size, center=True, min_periods=1).mean()
    if 'AU06_r' in df_raw.columns:
        df_raw['AU06_r_smooth'] = df_raw['AU06_r'].rolling(window=window_size, center=True, min_periods=1).mean()
        
    # Smooth the already-multiplied signal (Multiply -> Smooth)
    if 'Raw_Multiplied' in df_raw.columns:
        df_raw['Multiply_Then_Smooth'] = df_raw['Raw_Multiplied'].rolling(window=window_size, center=True, min_periods=1).mean()

    # --- MATH STEP 3: MULTIPLY THE ALREADY-SMOOTHED SIGNALS (Smooth -> Multiply) ---
    if 'AU12_r_smooth' in df_raw.columns and 'AU06_r_smooth' in df_raw.columns:
        df_raw['Smooth_Then_Multiply'] = df_raw['AU12_r_smooth'] * df_raw['AU06_r_smooth']

    # 5. Setup the 7-Panel Plot
    print("Generating Plot...")
    fig, axs = plt.subplots(7, 1, figsize=(14, 21), sharex=True, gridspec_kw={'height_ratios': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1]})
    
    # Update title to reflect if we are zoomed in
    title_suffix = f" (Zoomed: {start_time}s to {end_time}s)" if (start_time or end_time) else " (Full Video)"
    gt_suffix = " (No Ground Truth Provided)" if not ground_truth else ""
    fig.suptitle(f'Smile Detection Analysis - Video {video_id}{title_suffix}{gt_suffix}', fontsize=16, fontweight='bold', y=0.98)

    # --- Subplot 1: Raw AU12_r ---
    if 'AU12_r' in df_raw.columns:
        axs[0].plot(timestamps, df_raw['AU12_r'], color='blue', alpha=0.6, label='Raw AU12_r (Lip Corner)')
        axs[0].set_ylabel('Intensity (0-5)')
        axs[0].set_title('Raw Signal: AU12_r overlaid with Ground Truth')
        axs[0].grid(True, alpha=0.3)
        for gt_start, gt_end in ground_truth:
            axs[0].axvspan(gt_start, gt_end, color='lime', alpha=0.2)
    else:
        axs[0].text(0.5, 0.5, 'AU12_r column not found', ha='center', va='center')

    # --- Subplot 2: Raw AU06_r ---
    if 'AU06_r' in df_raw.columns:
        axs[1].plot(timestamps, df_raw['AU06_r'], color='red', alpha=0.6, label='Raw AU06_r (Cheek Raiser)')
        axs[1].set_ylabel('Intensity (0-5)')
        axs[1].set_title('Raw Signal: AU06_r overlaid with Ground Truth')
        axs[1].grid(True, alpha=0.3)
        for gt_start, gt_end in ground_truth:
            axs[1].axvspan(gt_start, gt_end, color='lime', alpha=0.2)
    else:
        axs[1].text(0.5, 0.5, 'AU06_r column not found', ha='center', va='center')

    # --- Subplot 3: RAW MULTIPLIED SIGNAL ---
    if 'Raw_Multiplied' in df_raw.columns:
        axs[2].plot(timestamps, df_raw['Raw_Multiplied'], color='magenta', alpha=0.6, label='Raw AU12_r × Raw AU06_r')
        axs[2].set_ylabel('Intensity (0-25)')
        axs[2].set_title('Multiplied Raw Signals (Very Noisy)')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc='upper right')
        for gt_start, gt_end in ground_truth:
            axs[2].axvspan(gt_start, gt_end, color='lime', alpha=0.2)
    else:
        axs[2].text(0.5, 0.5, 'Could not calculate raw multiplied signal', ha='center', va='center')

    # --- Subplot 4: MULTIPLY THEN SMOOTH ---
    if 'Multiply_Then_Smooth' in df_raw.columns:
        axs[3].plot(timestamps, df_raw['Multiply_Then_Smooth'], color='olive', linewidth=2, label='Smoothed (Raw_Multiplied)')
        axs[3].set_ylabel('Intensity (0-25)')
        axs[3].set_title(f'Method 1: MULTIPLY First, THEN Smooth (Window={window_size})')
        axs[3].grid(True, alpha=0.3)
        axs[3].legend(loc='upper right')
        for gt_start, gt_end in ground_truth:
            axs[3].axvspan(gt_start, gt_end, color='lime', alpha=0.2)
    else:
        axs[3].text(0.5, 0.5, 'Could not calculate Multiply-Then-Smooth', ha='center', va='center')

    # --- Subplot 5: SMOOTHED INDIVIDUAL SIGNALS ---
    if 'AU12_r_smooth' in df_raw.columns:
        axs[4].plot(timestamps, df_raw['AU12_r_smooth'], color='darkblue', linewidth=2, label='Smoothed AU12_r')
    if 'AU06_r_smooth' in df_raw.columns:
        axs[4].plot(timestamps, df_raw['AU06_r_smooth'], color='darkred', linewidth=2, label='Smoothed AU06_r')
    
    axs[4].set_ylabel('Intensity (0-5)')
    axs[4].set_title(f'Low Pass Filtered Individual Signals (Window={window_size})')
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc='upper right')
    for gt_start, gt_end in ground_truth:
        axs[4].axvspan(gt_start, gt_end, color='lime', alpha=0.2)

    # --- Subplot 6: SMOOTH THEN MULTIPLY ---
    if 'Smooth_Then_Multiply' in df_raw.columns:
        axs[5].plot(timestamps, df_raw['Smooth_Then_Multiply'], color='purple', linewidth=2, label='AU12_smooth × AU06_smooth')
        axs[5].set_ylabel('Intensity (0-25)')
        axs[5].set_title('Method 2: SMOOTH First, THEN Multiply (The Correct Way)')
        axs[5].grid(True, alpha=0.3)
        axs[5].legend(loc='upper right')
        for gt_start, gt_end in ground_truth:
            axs[5].axvspan(gt_start, gt_end, color='lime', alpha=0.2)
    else:
        axs[5].text(0.5, 0.5, 'Could not calculate Smooth-Then-Multiply', ha='center', va='center')

    # --- Subplot 7: Event Timeline ---
    for i, (start, end) in enumerate(ground_truth):
        axs[6].hlines(y=2, xmin=start, xmax=end, color='limegreen', linewidth=15)
        
    for i, (start, end) in enumerate(simple_extracted):
        axs[6].hlines(y=1, xmin=start, xmax=end, color='royalblue', linewidth=15)

    for i, (start, end) in enumerate(combo_extracted):
        axs[6].hlines(y=0, xmin=start, xmax=end, color='darkorange', linewidth=15)

    axs[6].set_title('Event Detection Timeline')
    axs[6].set_xlabel('Timestamp (seconds)', fontsize=12)
    axs[6].set_yticks([0, 1, 2])
    axs[6].set_yticklabels(['Combo\n(Orange)', 'Simple\n(Blue)', 'Ground Truth\n(Green)'])
    axs[6].set_ylim(-0.8, 2.8)
    axs[6].grid(axis='x', linestyle='--', alpha=0.7)

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
    
    # Example 2: Zoom in on a specific section where we DON'T have ground truth
    visualize_smile_events(
        video_id="237", 
        ground_truth=None,  # Safe to leave as None now!
        fps=30.0,
        window_size=30,
        start_time=0,  
        end_time=400.0   
    )