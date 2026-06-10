import os
import re
import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your ground truth dictionaries/lists safely
try:
    from ground_truth import GROUND_TRUTH_INTERVALS_249, GROUND_TRUTH_INTERVALS_236
except ImportError:
    pass # Safely ignore if the file doesn't exist yet

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
def visualize_smile_events(video_id, ground_truth=None, fps=30.0, 
                           window_size=10, 
                           start_time=None,  
                           end_time=None,    
                           raw_base_dir='/home/amir/Ados_lab/videos/openface_marisa',
                           out_base_dir='/home/amir/Ados_lab/out'):
    """
    Dynamically loads data, applies signal processing, extracts smart intervals,
    and plots the entire pipeline.
    """
    ground_truth = ground_truth or []
    
    print(f"\n--- Processing Video {video_id} ---")
    
    raw_csv = f"{raw_base_dir}/{video_id}/{video_id}_Emotions_marissa.csv"
    csv_simple = f"{out_base_dir}/{video_id}/FeaturesAU/emo/SimpleSmiling/{video_id}_emo_I.csv"
    csv_combo = f"{out_base_dir}/{video_id}/FeaturesAU/emo/ComboSmiling/{video_id}_emo_I.csv"

    print("Loading Interval Data...")
    simple_extracted = extract_intervals_from_csv(csv_simple, column_name='peak_lp_simple', fps=fps)
    combo_extracted = extract_intervals_from_csv(csv_combo, column_name='peak_lp_combo', fps=fps)
    
    print("Loading Raw OpenFace Signal...")
    if not os.path.exists(raw_csv):
        print(f"Error: Could not find raw signal file at {raw_csv}")
        return []
        
    df_raw = pd.read_csv(raw_csv)
    df_raw.columns = df_raw.columns.str.strip()  
    timestamps = df_raw['timestamp']

    # -------------------------------------------------------------------------
    # NEW: SAVE UNFILTERED DATA & APPLY TRACKING FAILURE FILTER
    # -------------------------------------------------------------------------
    # Save a copy of the raw signals so we can plot the "Before"
    if 'AU12_r' in df_raw.columns and 'AU06_r' in df_raw.columns:
        df_raw['AU12_r_unfiltered'] = df_raw['AU12_r'].copy()
        df_raw['AU06_r_unfiltered'] = df_raw['AU06_r'].copy()

    print("Filtering out tracking failures and low-confidence frames...")
    bad_frames = pd.Series(False, index=df_raw.index)
    
    if 'success' in df_raw.columns:
        bad_frames = bad_frames | (df_raw['success'] == 0)
    if 'confidence' in df_raw.columns:
        bad_frames = bad_frames | (df_raw['confidence'] < 0.80)
        
    df_raw['bad_tracking_flag'] = bad_frames # Save mask to plot the grey boxes
    df_raw.loc[bad_frames, ['AU12_r', 'AU06_r']] = 0.0 # Zero out the active math columns
    # -------------------------------------------------------------------------

    # --- MATH PIPELINE ---
    if 'AU12_r' in df_raw.columns and 'AU06_r' in df_raw.columns:
        df_raw['Raw_Multiplied'] = df_raw['AU12_r'] * df_raw['AU06_r']

    print(f"Applying Low Pass Filter (Window={window_size})...")
    if 'AU12_r' in df_raw.columns:
        df_raw['AU12_r_smooth'] = df_raw['AU12_r'].rolling(window=window_size, center=True, min_periods=1).mean()
    if 'AU06_r' in df_raw.columns:
        df_raw['AU06_r_smooth'] = df_raw['AU06_r'].rolling(window=window_size, center=True, min_periods=1).mean()
    if 'Raw_Multiplied' in df_raw.columns:
        df_raw['Multiply_Then_Smooth'] = df_raw['Raw_Multiplied'].rolling(window=window_size, center=True, min_periods=1).mean()

    if 'AU12_r_smooth' in df_raw.columns and 'AU06_r_smooth' in df_raw.columns:
        df_raw['Smooth_Then_Multiply'] = df_raw['AU12_r_smooth'] * df_raw['AU06_r_smooth']

    print("Detecting Smart Intervals...")
    smart_intervals = []
    if 'Smooth_Then_Multiply' in df_raw.columns:
        smart_intervals = find_smart_intervals(
            timestamps.values, 
            df_raw['Smooth_Then_Multiply'].values, 
            peak_thresh=2.0, 
            base_thresh=0.5
        )

    # --- PLOTTING ---
    print("Generating Plot...")
    # Changed to 7 subplots to accommodate the "Before & After"
    fig, axs = plt.subplots(7, 1, figsize=(14, 21), sharex=True, gridspec_kw={'height_ratios': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1]})
    
    title_suffix = f" (Zoomed: {start_time}s to {end_time}s)" if (start_time or end_time) else " (Full Video)"
    gt_suffix = " (No Ground Truth Provided)" if not ground_truth else ""
    
    fig.suptitle(
        f'Smart Smile Detection Analysis - Video {video_id}{title_suffix}{gt_suffix}', 
        fontsize=16, 
        fontweight='bold', 
        y=0.98, 
        x=0.02, 
        ha='left'
    )

    # --- SUBPLOT 0: BEFORE (Unfiltered) ---
    if 'AU12_r_unfiltered' in df_raw.columns and 'AU06_r_unfiltered' in df_raw.columns:
        axs[0].plot(timestamps, df_raw['AU12_r_unfiltered'], color='blue', alpha=0.5, label='Raw AU12_r')
        axs[0].plot(timestamps, df_raw['AU06_r_unfiltered'], color='red', alpha=0.5, label='Raw AU06_r')
        axs[0].set_ylabel('Intensity (0-5)')
        axs[0].set_title('1. BEFORE: Raw Signals (Vulnerable to Tracking Artifacts)')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='upper right')
        for gt_start, gt_end in ground_truth: axs[0].axvspan(gt_start, gt_end, color='lime', alpha=0.2)

    # --- SUBPLOT 1: AFTER (Filtered) ---
    if 'AU12_r' in df_raw.columns and 'AU06_r' in df_raw.columns:
        axs[1].plot(timestamps, df_raw['AU12_r'], color='blue', alpha=0.8, label='Cleaned AU12_r')
        axs[1].plot(timestamps, df_raw['AU06_r'], color='red', alpha=0.8, label='Cleaned AU06_r')
        
        # Shade the exact regions where the tracker failed
        if 'bad_tracking_flag' in df_raw.columns:
            axs[1].fill_between(timestamps, 0, 5, where=df_raw['bad_tracking_flag'], color='gray', alpha=0.3, label='Tracker Failure Zone (Zeroed)')
            
        axs[1].set_ylabel('Intensity (0-5)')
        axs[1].set_title('2. AFTER: Cleaned Signals (Tracking Failures Neutralized)')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc='upper right')
        for gt_start, gt_end in ground_truth: axs[1].axvspan(gt_start, gt_end, color='lime', alpha=0.2)

    # --- SUBPLOT 2: SMOOTHED ---
    if 'AU12_r_smooth' in df_raw.columns and 'AU06_r_smooth' in df_raw.columns:
        axs[2].plot(timestamps, df_raw['AU12_r_smooth'], color='darkblue', linewidth=2, label='Smoothed AU12_r')
        axs[2].plot(timestamps, df_raw['AU06_r_smooth'], color='darkred', linewidth=2, label='Smoothed AU06_r')
        axs[2].set_ylabel('Intensity (0-5)')
        axs[2].set_title(f'3. Smoothed Individual Signals (Window={window_size})')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc='upper right')
        for gt_start, gt_end in ground_truth: axs[2].axvspan(gt_start, gt_end, color='lime', alpha=0.2)

    # --- SUBPLOT 3: MULTIPLIED RAW ---
    if 'Raw_Multiplied' in df_raw.columns:
        axs[3].plot(timestamps, df_raw['Raw_Multiplied'], color='magenta', alpha=0.6, label='Raw AU12 × Raw AU06')
        axs[3].set_ylabel('Intensity (0-25)')
        axs[3].set_title('4. Multiplied Raw Signals')
        axs[3].grid(True, alpha=0.3)
        axs[3].legend(loc='upper right')

    # --- SUBPLOT 4: MULTIPLY THEN SMOOTH ---
    if 'Multiply_Then_Smooth' in df_raw.columns:
        axs[4].plot(timestamps, df_raw['Multiply_Then_Smooth'], color='olive', linewidth=2, label='Smoothed (Raw_Multiplied)')
        axs[4].set_ylabel('Intensity (0-25)')
        axs[4].set_title(f'5. Method 1: MULTIPLY First, THEN Smooth')
        axs[4].grid(True, alpha=0.3)
        axs[4].legend(loc='upper right')

    # --- SUBPLOT 5: SMOOTH THEN MULTIPLY ---
    if 'Smooth_Then_Multiply' in df_raw.columns:
        axs[5].plot(timestamps, df_raw['Smooth_Then_Multiply'], color='purple', linewidth=2, label='AU12_smooth × AU06_smooth')
        axs[5].axhline(y=2.0, color='black', linestyle='--', alpha=0.5, label='Peak Threshold (2.0)')
        axs[5].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Base Threshold (0.5)')
        for s, e in smart_intervals:
            axs[5].axvspan(s, e, color='purple', alpha=0.15)
            axs[5].axvline(x=s, color='green', linestyle=':', alpha=0.7) 
            axs[5].axvline(x=e, color='red', linestyle=':', alpha=0.7)   
        axs[5].set_ylabel('Intensity (0-25)')
        axs[5].set_title('6. Method 2: SMOOTH First, THEN Multiply (with Smart Interval Bounds)')
        axs[5].grid(True, alpha=0.3)
        axs[5].legend(loc='upper right')

    # --- SUBPLOT 6: TIMELINE ---
    for i, (start, end) in enumerate(ground_truth): axs[6].hlines(y=3, xmin=start, xmax=end, color='limegreen', linewidth=15)
    for i, (start, end) in enumerate(smart_intervals): axs[6].hlines(y=2, xmin=start, xmax=end, color='darkviolet', linewidth=15)
    for i, (start, end) in enumerate(simple_extracted): axs[6].hlines(y=1, xmin=start, xmax=end, color='royalblue', linewidth=15)
    for i, (start, end) in enumerate(combo_extracted): axs[6].hlines(y=0, xmin=start, xmax=end, color='darkorange', linewidth=15)

    axs[6].set_title('7. Event Detection Timeline')
    axs[6].set_xlabel('Timestamp (seconds)', fontsize=12)
    axs[6].set_yticks([0, 1, 2, 3])
    axs[6].set_yticklabels(['Legacy Combo', 'Legacy Simple', 'Smart Peak', 'Ground Truth'])
    axs[6].set_ylim(-0.8, 3.8)
    axs[6].grid(axis='x', linestyle='--', alpha=0.7)

    min_x = start_time if start_time is not None else timestamps.min()
    max_x = end_time if end_time is not None else timestamps.max()
    plt.xlim(min_x, max_x)
    plt.tight_layout()
    plt.show()

    return smart_intervals

# =========================================================
# 3. VIDEO RENDERING FUNCTION
# =========================================================
def render_smile_light(input_video_path, output_video_path, detected_intervals):
    """
    Reads a video, overlays a green light when a smile is detected, 
    and saves the result to an .avi file using the XVID codec.
    """
    print(f"\n--- Rendering Video Output ---")
    print(f"Opening original video: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open {input_video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0 # Fallback 
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Saving rendered video to: {output_video_path} at {fps} FPS")
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time_sec = frame_count / fps
        
        # Check if current time falls inside ANY of the detected intervals
        is_smiling = any(start <= current_time_sec <= end for start, end in detected_intervals)
        
        if is_smiling:
            # Calculate the exact center of the frame
            center_x = width // 2
            center_y = height // 2
            
            # 1. Draw the Circle perfectly in the middle
            radius = 30
            color = (0, 255, 0) # Green in BGR
            thickness = -1 # Solid fill
            
            cv2.circle(frame, (center_x, center_y), radius, color, thickness)
            
            # 2. Draw the Text centered slightly below the circle
            text_position = (center_x - 140, center_y + 60)
            
            cv2.putText(frame, "SMILE DETECTED", text_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        out.write(frame)
        frame_count += 1
        
        if frame_count % 500 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print("\n=======================================================")
    print("✅ RENDERING COMPLETE!")
    print(f"Video saved successfully to: {output_video_path}")
    print("⚠️  IMPORTANT: Navigate to this folder using Windows Explorer to open the .avi file.")
    print("=======================================================\n")

# =========================================================
# 4. EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    
    # Define your paths
    video_to_process = "249" # Change this to "236" to process the other video
    original_vid_path = f'/home/amir/Ados_lab/videos/emotions/marisa_videos/{video_to_process}_Emotions_marissa.mp4'
    output_vid_path = f'/home/amir/Ados_lab/out/{video_to_process}_smile_rendered.avi'

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_vid_path), exist_ok=True)

    # 1. Run the analysis and CAPTURE the returned intervals
    print("Step 1: Running mathematical analysis and plotting...")
    
    detected_smart_intervals = visualize_smile_events(
        video_id=video_to_process, 
        ground_truth=GROUND_TRUTH_INTERVALS_249, # Can safely be changed to None
        fps=30.0,
        window_size=30,
        start_time=0,  
        end_time=405.0    
    )

    # 2. Pass those captured intervals into the video renderer
    if detected_smart_intervals:
        print("\nStep 2: Rendering green light onto video...")
        render_smile_light(original_vid_path, output_vid_path, detected_smart_intervals)
    else:
        print("\nNo intervals were detected, or the data could not be loaded. Skipping video rendering.")