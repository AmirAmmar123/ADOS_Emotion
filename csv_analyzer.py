import os 
import numpy as np
import pandas as pd
import json
import glob
from mylogger import Logger
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

OPENFACE_MARISA_DIR = '/home/amir/Ados_lab/videos/openface_marisa'
OPENFACE_SUBJECTS_DIR = '/home/amir/Ados_lab/videos/openface_subjects'
ALL_INTERVALS_JSON_FILE = 'all_intervals.json'
TEST_NAME = 'zscore'
BASE_DIR = '/home/amir/Ados_lab/videos'


def get_dyad_paths(base_dir, video_id):
    """
    Dynamically finds the CSV and JSON files for both the subject and interviewer.
    """
    paths = {
        'subj_csv': None, 'subj_json': None,
        'intv_csv': None, 'intv_json': None
    }
    
    intv_dir = os.path.join(base_dir, f'openface_marisa/{video_id}')
    subj_dir = os.path.join(base_dir, f'openface_subjects/{video_id}')
    
    def find_files(target_dir):
        if not os.path.exists(target_dir):
            return None, None
            
        csv_files = glob.glob(os.path.join(target_dir, '*.csv'))
        json_files = glob.glob(os.path.join(target_dir, '*all_intervals.json'))
        
        csv_path = csv_files[0] if csv_files else None
        json_path = json_files[0] if json_files else None
        
        return csv_path, json_path

    paths['subj_csv'], paths['subj_json'] = find_files(subj_dir)
    paths['intv_csv'], paths['intv_json'] = find_files(intv_dir)
    
    return paths
    

def get_dyad_intersection(marisa_dir=OPENFACE_MARISA_DIR, subjects_dir=OPENFACE_SUBJECTS_DIR):
    """
    Returns a set of Video IDs that exist in BOTH directories.
    """
    marisa_ids = {entry.name for entry in os.scandir(marisa_dir) if entry.is_dir()}
    subject_ids = {entry.name for entry in os.scandir(subjects_dir) if entry.is_dir()}
    
    return marisa_ids & subject_ids


class DYADAnalyzer:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.fps = 30  
        self.subj_df = None
        self.intv_df = None
        self.subj_intervals = {}
        self.intv_intervals = {}
        
        os.makedirs('logs', exist_ok=True)
        self.logger = Logger('DYADAnalyzer', 'logs/dyad_analyzer.log')

    def _process_signal(self, csv_path):
        if not os.path.exists(csv_path):
            self.logger.error(f"Could not find CSV at {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df = df.copy() 
        
        df['bad_tracking'] = (df.get('success', 1) == 0) | (df.get('confidence', 1) < 0.8)
        
        au_cols = ['AU12_r', 'AU06_r']
        for c in au_cols:
            if c in df.columns:
                df.loc[df['bad_tracking'], c] = 0.0
                df[f'{c}_smooth'] = df[c].rolling(self.window_size, center=True, min_periods=1).mean()
            else:
                self.logger.warning(f"Missing column {c} in {csv_path}")
                return df

        df['Smooth_Smile_Product'] = df['AU12_r_smooth'] * df['AU06_r_smooth']
        return df

    def _load_intervals(self, json_path):
        if not os.path.exists(json_path):
            self.logger.warning(f"Could not find JSON at {json_path}. Intervals will be empty.")
            return {}
        with open(json_path, 'r') as f:
            return json.load(f)

    def load_dyad_data(self, subj_csv, intv_csv, subj_json, intv_json):
        """Loads both CSV signals and JSON intervals into the class instance."""
        self.logger.info("Processing Subject data...")
        self.subj_df = self._process_signal(subj_csv)
        self.subj_intervals = self._load_intervals(subj_json)
                
        self.logger.info("Processing Interviewer data...")
        self.intv_df = self._process_signal(intv_csv)
        self.intv_intervals = self._load_intervals(intv_json)

    def count_marisa_subject_smiles(self, test_name=TEST_NAME):
        """
        Counts the smiles instantly using the pre-loaded JSON dictionaries 
        stored within the class instance.
        """
        # .get() safely returns an empty list if the test_name key doesn't exist
        marisa_count = len(self.intv_intervals.get(test_name, []))
        subject_count = len(self.subj_intervals.get(test_name, []))
        
        return marisa_count, subject_count

    def calculate_global_cosine_shifts(self, max_shift_sec=5.0):
        """
        Calculates the overall Cosine Similarity between the full interviewer 
        signal and the full subject signal across different time shifts.
        
        Returns:
            delay_axis (numpy array): The shift amounts in seconds (X-axis).
            cosine_scores (numpy array): The overall similarity score for each shift (Y-axis).
        """
        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return None, None

        self.logger.info(f"Calculating Global Cosine Similarity for shifts up to {max_shift_sec} seconds...")

        # 1. Align to minimum length using the FULL uncut signals
        min_len = min(len(self.subj_df), len(self.intv_df))
        
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].reset_index(drop=True)
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].reset_index(drop=True)

        # 2. Setup the exact frame-by-frame resolution
        max_shift_frames = int(max_shift_sec * self.fps)
        range_resolution_frames = range(0, max_shift_frames + 1)
        
        cosine_scores = []
        delay_axis = []
        
        # Calculate the magnitude (norm) of the Interviewer signal once, as it doesn't shift
        norm_i = np.linalg.norm(i_sig)

        # 3. Calculate Cosine Similarity for each frame shift
        for shift_frames in range_resolution_frames:
            # Shift subject backward in time to simulate delay
            shifted_subject = s_sig.shift(-shift_frames, fill_value=0)
            
            # --- Cosine Similarity Math ---
            # 1. Dot Product (Sum of A * B)
            dot_product = np.dot(i_sig, shifted_subject)
            
            # 2. Magnitude of Shifted Subject
            norm_s = np.linalg.norm(shifted_subject)
            
            # 3. Final calculation safely avoiding division by zero
            if norm_i > 0 and norm_s > 0:
                cos_sim = dot_product / (norm_i * norm_s)
            else:
                cos_sim = 0.0
                
            cosine_scores.append(cos_sim)
            delay_axis.append(shift_frames / self.fps)
            
        return np.array(delay_axis), np.array(cosine_scores)
    
    def analyze_interval_cosine_stats(self, test_name='zscore', max_shift_sec=5.0):
        """
        Calculates Cosine Similarity for each individual smile interval across all shifts.
        Returns a Pandas DataFrame containing max, min, mean, std, and optimal delays per interval.
        """
        # STEP 1: Validation
        # Ensure that the data for both the subject and interviewer has been loaded first.
        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return None

        self.logger.info(f"Calculating Interval-based Cosine Statistics (Max Shift: {max_shift_sec}s)...")

        # STEP 2: Data Alignment and Extraction
        # Videos might be slightly different lengths. We find the shortest one so we don't 
        # get "out of bounds" errors when comparing them frame-by-frame.
        min_len = min(len(self.subj_df), len(self.intv_df))
        
        # Extract the timestamps and both smoothed signals as pure NumPy arrays. 
        # NumPy arrays are much faster for mathematical loops than Pandas DataFrames.
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values

        # STEP 3: Load Intervals and Setup Delay Axis
        # Fetch the start and end times where the Interviewer smiled (the "triggers")
        intervals = self.intv_intervals.get(test_name, [])
        
        # Convert the maximum shift time (e.g., 5 seconds) into an exact number of frames (e.g., 150)
        max_shift_frames = int(max_shift_sec * self.fps)
        
        # Create an array representing the X-axis of delays in seconds (0.0, 0.033, 0.066 ... 5.0)
        delay_axis = np.array([f / self.fps for f in range(max_shift_frames + 1)])

        results = []

        # STEP 4: Iterate Over Each Smile Interval
        for start_t, end_t in intervals:
            
            # Find the exact row numbers (indices) where the timestamp falls between start_t and end_t
            base_idx = np.where((ts >= start_t) & (ts <= end_t))[0]
            if len(base_idx) == 0: continue # Skip if the interval is empty/invalid

            # Extract just the interviewer's signal for this specific smile
            i_segment = i_sig[base_idx]
            
            # Calculate the magnitude (norm) of the interviewer's smile.
            norm_i = np.linalg.norm(i_segment)
            
            # If the interviewer's signal is completely flat (0), skip it to avoid dividing by zero later.
            if norm_i == 0: continue

            interval_scores = []

            # STEP 5: Test Every Possible Time Delay (Shift) for the Current Smile
            for shift_frames in range(max_shift_frames + 1):
                
                # THE CORE LOGIC: We want to see if the subject smiled *after* the interviewer.
                # By adding 'shift_frames' to the base_idx, we are looking into the subject's FUTURE.
                # Example: If interviewer smiled at frames 100-120, and shift is 30, we look at subject's frames 130-150.
                shifted_idx = base_idx + shift_frames
                
                # STEP 6: Boundary Protection
                # If the shift pushes us past the very end of the video, we must cut off the excess.
                valid_mask = shifted_idx < min_len # True for valid frames, False for out-of-bounds frames
                valid_shifted_idx = shifted_idx[valid_mask] # Subject's future frames (truncated if necessary)
                # Apply Mask to Base Index as well to keep them aligned (only the valid frames) 
                valid_base_idx = base_idx[valid_mask]       # Interviewer's current frames (truncated to match)
                
                # Extract the segments using the safe indices
                s_segment = s_sig[valid_shifted_idx]
                i_seg_trunc = i_sig[valid_base_idx]
                
                # Re-calculate the magnitudes (norms) just in case they were truncated
                norm_s = np.linalg.norm(s_segment)
                n_i_trunc = np.linalg.norm(i_seg_trunc)

                # STEP 7: Calculate Cosine Similarity
                # Formula: (A dot B) / (|A| * |B|)
                # We only divide if both magnitudes are greater than 0 to prevent crashes.
                if n_i_trunc > 0 and norm_s > 0:
                    cos_sim = np.dot(i_seg_trunc, s_segment) / (n_i_trunc * norm_s)
                else:
                    cos_sim = 0.0

                interval_scores.append(cos_sim)

            # Convert the list of scores for all shifts into a NumPy array
            interval_scores = np.array(interval_scores)

            # STEP 8: Extract Statistics for this Specific Interval
            # Find the position (index) of the highest and lowest similarity scores
            max_idx = np.argmax(interval_scores)
            min_idx = np.argmin(interval_scores)

            # Save a dictionary summarizing how well the subject matched this specific smile
            results.append({
                'Start_Time': start_t,
                'End_Time': end_t,
                'Duration': round(end_t - start_t, 2),
                'Max_Cos_Sim': interval_scores[max_idx],           # The highest synchrony score achieved
                'Optimal_Delay_Sec': delay_axis[max_idx],          # How many seconds it took the subject to hit that peak
                'Min_Cos_Sim': interval_scores[min_idx],           # The lowest synchrony score
                'Worst_Delay_Sec': delay_axis[min_idx],            # The delay that produced the lowest score
                'Mean_Cos_Sim': np.mean(interval_scores),          # Average similarity across all tested delays
                'Std_Cos_Sim': np.std(interval_scores)             # How much the similarity fluctuated
            })

        # STEP 9: Return the Final Table
        # Convert the list of dictionaries into a clean Pandas DataFrame for easy viewing and saving.
        return pd.DataFrame(results)
    
    def classify_smile_responses(self, test_name=TEST_NAME, max_response_sec=3.0):
        """
        Classifies subject smile responses to interviewer smiles based on overlap or temporal delay.
        Returns a DataFrame with English-standardized status labels.
        """
        intv_intervals = self.intv_intervals.get(test_name, [])
        subj_intervals = self.subj_intervals.get(test_name, [])
        
        results = []

        for i_start, i_end in intv_intervals:
            # Find all subject intervals that started within the response window [i_start, i_start + max_response_sec]
            responses = [s for s in subj_intervals if i_start < s[0] <= (i_start + max_response_sec)]
            
            if not responses:
                status = "NO_RESPONSE"
                details = None
            else:
                # Select the first occurring response within the window
                s_start, s_end = responses[0]
                
                # Calculate overlap duration using interval intersection formula
                overlap = max(0.0, min(i_end, s_end) - max(i_start, s_start))
                
                if overlap > 0:
                    status = "OVERLAP" # Indicates synchronous or immediate mimicry
                else:
                    status = "DELAYED_RESPONSE" # Indicates a response within the valid window, but no overlap
                
                details = {
                    'subject_start': s_start,
                    'delay_from_trigger': round(s_start - i_start, 2),
                    'overlap_duration': round(overlap, 2)
                }

            results.append({
                'intv_start': i_start,
                'intv_end': i_end,
                'status': status,
                'details': details
            })

        # Return the results as a clean DataFrame
        return pd.DataFrame(results)
    

    def detect_reactive_smiles(self, test_name=TEST_NAME, merge_gap_frames=10, reaction_window_frames=30):
        """
        1. Merges interviewer intervals that are separated by a gap smaller than 'merge_gap_frames'.
        2. Checks if the subject initiated a smile within 'reaction_window_frames' after the merged interval began.
        """
        intv_intervals = self.intv_intervals.get(test_name, [])
        subj_intervals = self.subj_intervals.get(test_name, [])

        if not intv_intervals:
            self.logger.warning("No interviewer intervals found to process.")
            return pd.DataFrame()

        # Convert frame constraints to seconds
        merge_gap_sec = merge_gap_frames / self.fps
        reaction_window_sec = reaction_window_frames / self.fps

        # ==========================================
        # STEP 1: Merge close Interviewer intervals
        # ==========================================
        # Ensure intervals are sorted chronologically by start time
        sorted_intv = sorted(intv_intervals, key=lambda x: x[0])
        
        merged_intv = []
        current_start, current_end = sorted_intv[0]

        for start, end in sorted_intv[1:]:
            # Calculate the time gap between the end of the previous smile and the start of the current one
            time_gap = start - current_end
            
            if time_gap <= merge_gap_sec:
                # The gap is small (<= 10 frames). Merge them by extending the current_end.
                current_end = max(current_end, end)
            else:
                # The gap is too large. Save the current merged block and start tracking a new one.
                merged_intv.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Append the final interval after the loop finishes
        merged_intv.append((current_start, current_end))
        
        self.logger.info(f"Merged {len(intv_intervals)} raw intervals down to {len(merged_intv)} continuous intervals.")


        # ==========================================
        # STEP 2: Detect Reactivity
        # ==========================================
        results = []

        for i_start, i_end in merged_intv:
            
            # Find any subject interval where the START time falls exactly between:
            # [Interviewer Start Time] and [Interviewer Start Time + Reaction Window]
            valid_responses = [
                s for s in subj_intervals 
                if i_start <= s[0] <= (i_start + reaction_window_sec)
            ]

            if valid_responses:
                # Take the first smile that matched the criteria
                s_start, s_end = valid_responses[0] 
                status = "REACTIVE_SMILE"
                delay_sec = round(s_start - i_start, 3)
                delay_frames = int(delay_sec * self.fps)
            else:
                s_start, s_end = None, None
                status = "NO_REACTION"
                delay_sec = None
                delay_frames = None

            results.append({
                'merged_intv_start': round(i_start, 3),
                'merged_intv_end': round(i_end, 3),
                'merged_duration_sec': round(i_end - i_start, 3),
                'status': status,
                'subj_response_start': round(s_start, 3) if s_start else None,
                'delay_sec': delay_sec,
                'delay_frames': delay_frames
            })

        return pd.DataFrame(results)

    def visualize_reactivity_pipeline(self, reactivity_df, test_name=TEST_NAME, merge_gap_frames=15, reaction_window_frames=30, s_time=0.0, e_time=60.0, video_id="Unknown"):
        """
        Generates a 3-panel figure to visually audit the interval merging and reactivity.
        Includes visual markers and exact timestamp numbers directly on the time axis for reactions.
        """
        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return

        intv_intervals = self.intv_intervals.get(test_name, [])
        subj_intervals = self.subj_intervals.get(test_name, [])
        
        if not intv_intervals:
            self.logger.warning("No interviewer intervals found to plot.")
            return

        merge_gap_sec = merge_gap_frames / self.fps
        reaction_window_sec = reaction_window_frames / self.fps
        sorted_intv = sorted(intv_intervals, key=lambda x: x[0])
        
        merged_intv = []
        current_start, current_end = sorted_intv[0]

        for start, end in sorted_intv[1:]:
            if (start - current_end) <= merge_gap_sec:
                current_end = max(current_end, end)
            else:
                merged_intv.append((current_start, current_end))
                current_start, current_end = start, end
        merged_intv.append((current_start, current_end))

        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        
        e_time = e_time or ts.max()
        valid_idx = (ts >= s_time) & (ts <= e_time)
        ts_zoom = ts[valid_idx]
        
        i_sig_zoom = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values[valid_idx]
        s_sig_zoom = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values[valid_idx]

        def format_intervals_for_plot(intervals, t_start, t_end):
            return [(s, max(0.1, e - s)) for s, e in intervals if (e >= t_start) and (s <= t_end)]
            
        raw_intv_rects = format_intervals_for_plot(intv_intervals, s_time, e_time)
        merged_intv_rects = format_intervals_for_plot(merged_intv, s_time, e_time)
        raw_subj_rects = format_intervals_for_plot(subj_intervals, s_time, e_time)

        fig, axs = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
        fig.suptitle(f"Reactivity Pipeline - Video {video_id} ({s_time}s to {e_time}s)", fontsize=16, fontweight='bold')

        # --- Panel 1: Before and After Merging ---
        axs[0].broken_barh(raw_intv_rects, (10, 8), facecolors='lightcoral', edgecolor='black', alpha=0.8)
        axs[0].broken_barh(merged_intv_rects, (25, 8), facecolors='firebrick', edgecolor='black')
        axs[0].set_yticks([14, 29])
        axs[0].set_yticklabels(['Raw Intervals', 'Merged Intervals'])
        axs[0].set_title(f"1. Interviewer Intervals: Debouncing (Gap <= {merge_gap_frames} frames)")
        axs[0].grid(True, alpha=0.4)

        # --- Panel 2: Continuous Signals ---
        axs[1].plot(ts_zoom, i_sig_zoom, color='firebrick', label='Interviewer Signal', linewidth=2)
        axs[1].plot(ts_zoom, s_sig_zoom, color='royalblue', label='Subject Signal', linewidth=2)
        axs[1].set_title("2. Smoothed Smile Signals")
        axs[1].legend(loc="upper right")
        axs[1].grid(True, alpha=0.4)

        # --- Panel 3: Merged Intv vs Raw Subject WITH AXIS INDICATORS ---
        search_windows = []
        for _, row in reactivity_df.iterrows():
            i_start = row['merged_intv_start']
            if s_time <= i_start <= e_time:
                search_windows.append((i_start, reaction_window_sec))
        
        if search_windows:
            axs[2].broken_barh(search_windows, (10, 8), facecolors='yellow', alpha=0.3, edgecolor='orange', hatch='////')

        axs[2].broken_barh(merged_intv_rects, (25, 8), facecolors='firebrick', edgecolor='black')
        axs[2].broken_barh(raw_subj_rects, (10, 8), facecolors='royalblue', edgecolor='black', alpha=0.8)
        
        valid_reactions = reactivity_df[reactivity_df['status'] == 'REACTIVE_SMILE']
        for _, row in valid_reactions.iterrows():
            i_start = row['merged_intv_start']
            s_start = row['subj_response_start']
            delay = row['delay_sec']
            
            if s_time <= i_start <= e_time or s_time <= s_start <= e_time:
                # 1. Gold star at the top of the subject's block
                axs[2].plot(s_start, 18, marker='*', color='gold', markersize=14, markeredgecolor='green', zorder=5)
                # 2. Reaction time text above the block
                axs[2].text(s_start, 20.5, f'{delay}s delay', fontsize=10, color='green', fontweight='bold', ha='center', va='bottom')
                
                # 3. Vertical dashed line dropping down to the X-axis
                axs[2].axvline(x=s_start, color='green', linestyle='--', alpha=0.6, zorder=2)
                
                # 4. Pin a green triangle directly ON the physical X-axis line using transformations
                axs[2].plot(s_start, 0, marker='^', color='green', markersize=12, transform=axs[2].get_xaxis_transform(), clip_on=False, zorder=10)
                
                # 5. Add the exact Timestamp NUMBER directly below the X-axis line
                axs[2].text(s_start, -0.06, f'{s_start:.2f}s', color='green', fontweight='bold', ha='center', va='top', transform=axs[2].get_xaxis_transform(), fontsize=10)

        axs[2].set_yticks([14, 29])
        axs[2].set_yticklabels(['Subject (Raw)', 'Interviewer (Merged)'])
        axs[2].set_title(f"3. Reaction Timing: Yellow area = {reaction_window_sec}s Search Window")
        axs[2].set_xlabel("Time (seconds)", fontsize=12)
        
        # Increase bottom padding so the new custom axis text fits cleanly
        axs[2].tick_params(axis='x', pad=15)
        axs[2].grid(True, alpha=0.4)
        
        window_patch = mpatches.Patch(facecolor='yellow', alpha=0.3, edgecolor='orange', hatch='////', label=f'Valid Search Window')
        star_marker = mlines.Line2D([], [], color='w', marker='*', markerfacecolor='gold', markeredgecolor='green', markersize=12, label='Reaction Point')
        axis_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='--', markersize=10, label='Timestamp on X-Axis')
        axs[2].legend(handles=[window_patch, star_marker, axis_marker], loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()



    def simulate_cosine_shift_for_interval(self, test_name=TEST_NAME, interval_index=0, max_shift_sec=3.0):
        """
        Creates a 3-panel Interactive HTML Simulation.
        Top: Static Full Video Context (Macro view).
        Middle: Raw sliding signals (Animated without normalization).
        Bottom: Live Cosine Similarity curve (Animated).
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return

        intervals = self.intv_intervals.get(test_name, [])
        if not intervals or interval_index >= len(intervals):
            self.logger.warning(f"Invalid interval index. Only {len(intervals)} intervals available.")
            return

        start_t, end_t = intervals[interval_index]
        self.logger.info(f"Building simulation for Interval {interval_index} ({start_t}s - {end_t}s)...")
        
        # 1. Data Alignment
        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values

        base_idx = np.where((ts >= start_t) & (ts <= end_t))[0]
        if len(base_idx) == 0: return

        # 2. Extract Raw Segments (No Normalization for plotting)
        i_segment_raw = i_sig[base_idx]
        norm_i = np.linalg.norm(i_segment_raw)
        if norm_i == 0:
            self.logger.warning("Interviewer signal is completely flat here. Choose another interval.")
            return

        max_shift_frames = int(max_shift_sec * self.fps)
        delay_axis = [shift / self.fps for shift in range(max_shift_frames + 1)]
        
        # Pre-calculate all shifts and scores
        all_s_segments_raw = []
        cosine_scores = []
        best_score, best_delay = -1.0, 0.0

        for shift in range(max_shift_frames + 1):
            shifted_idx = base_idx + shift
            
            valid_mask = shifted_idx < min_len
            valid_shifted_idx = shifted_idx[valid_mask]
            valid_base_idx = base_idx[valid_mask]
            
            s_seg_raw = s_sig[valid_shifted_idx]
            i_seg_trunc = i_sig[valid_base_idx]
            
            if len(s_seg_raw) < len(base_idx):
                pad_width = len(base_idx) - len(s_seg_raw)
                s_seg_raw = np.pad(s_seg_raw, (0, pad_width), 'constant', constant_values=0)
                i_seg_trunc = np.pad(i_seg_trunc, (0, pad_width), 'constant', constant_values=0)

            # Mathematical Cosine Similarity still requires the norms
            norm_s = np.linalg.norm(s_seg_raw)
            n_i_trunc = np.linalg.norm(i_seg_trunc)
            
            if n_i_trunc > 0 and norm_s > 0:
                cos_sim = np.dot(i_seg_trunc, s_seg_raw) / (n_i_trunc * norm_s)
            else:
                cos_sim = 0.0

            cosine_scores.append(cos_sim)
            all_s_segments_raw.append(s_seg_raw) # Saving the raw segment for plotting

            if cos_sim > best_score:
                best_score, best_delay = cos_sim, delay_axis[shift]

        # --- BUILD PLOTLY SUBPLOTS ---
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=False,
            row_heights=[0.25, 0.45, 0.3], 
            vertical_spacing=0.1,
            subplot_titles=(
                "Full Video Context (Static Reference)", 
                "Raw Signals (Animated Sliding Window)", 
                "Cosine Similarity Profile"
            )
        )

        # Trace 0 & 1: Full Context (Static)
        fig.add_trace(go.Scatter(x=ts, y=i_sig, mode='lines', name='Intv (Full)', line=dict(color='firebrick', width=1), opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=s_sig, mode='lines', name='Subj (Full)', line=dict(color='royalblue', width=1), opacity=0.5), row=1, col=1)
        
        # Add a gold highlight box to show where we are in the video
        fig.add_vrect(x0=start_t, x1=end_t, fillcolor="gold", opacity=0.4, line_width=1, line_color="orange", row=1, col=1)

        # Trace 2 & 3: Local Interval Signals (Animated) using RAW arrays
        fig.add_trace(go.Scatter(x=ts[base_idx], y=i_segment_raw, mode='lines', name='Intv (Base)', line=dict(color='firebrick', width=4)), row=2, col=1)
        fig.add_trace(go.Scatter(x=ts[base_idx], y=all_s_segments_raw[0], mode='lines', name='Subj (Sliding)', line=dict(color='royalblue', width=4, dash='dash'), fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.2)'), row=2, col=1)

        # Trace 4 & 5: Cosine Curve and Moving Dot (Animated)
        fig.add_trace(go.Scatter(x=delay_axis, y=cosine_scores, mode='lines', name='Similarity Curve', line=dict(color='purple', width=3)), row=3, col=1)
        fig.add_trace(go.Scatter(x=[delay_axis[0]], y=[cosine_scores[0]], mode='markers', name='Current Shift', marker=dict(color='red', size=14, line=dict(color='black', width=2))), row=3, col=1)

        # --- CREATE SLIDER STEPS ---
        steps = []
        for i, delay_sec in enumerate(delay_axis):
            step = dict(
                method="update",
                args=[
                    {"y": [
                        i_sig,                   # Trace 0 (Static)
                        s_sig,                   # Trace 1 (Static)
                        i_segment_raw,           # Trace 2 (Static Base)
                        all_s_segments_raw[i],   # Trace 3 (Sliding Subject)
                        cosine_scores,           # Trace 4 (Static Curve)
                        [cosine_scores[i]]       # Trace 5 (Moving Dot)
                    ],
                    "x": [
                        ts,                      # Trace 0
                        ts,                      # Trace 1
                        ts[base_idx],            # Trace 2
                        ts[base_idx],            # Trace 3
                        delay_axis,              # Trace 4
                        [delay_sec]              # Trace 5
                    ]},
                    {"title": f"<b>Interval:</b> {start_t:.2f}s - {end_t:.2f}s | <b>Shift:</b> {delay_sec:.2f}s | <b>Score:</b> {cosine_scores[i]:.3f}"}
                ],
                label=f"{delay_sec:.2f}s"
            )
            steps.append(step)

        # Add Slider Layout
        sliders = [dict(
            active=0, currentvalue={"prefix": "Time Shift: "}, pad={"t": 50}, steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            title=f"<b>Interval:</b> {start_t:.2f}s - {end_t:.2f}s | <b>Shift:</b> 0.00s | <b>Score:</b> {cosine_scores[0]:.3f}",
            height=800, width=1000, template="plotly_white", showlegend=False
        )

        # Updated Y-Axis title since it is no longer normalized
        fig.update_yaxes(title_text="Smooth Smile Intensity", row=2, col=1)
        fig.update_xaxes(title_text="Base Video Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Cosine Score", range=[0, 1.05], row=3, col=1)
        fig.update_xaxes(title_text="Subject Shift Delay (seconds)", row=3, col=1)

        html_filename = f"Cosine_Simulation_Interval_{interval_index}.html"
        fig.write_html(html_filename)
        print(f"\n---> Peak Synchrony for interval {interval_index} was {best_score:.3f} at a delay of {best_delay:.2f}s.")
        print(f"---> Simulation Saved! Open '{html_filename}' to view.\n")




if __name__ == "__main__":
    
    intersection = get_dyad_intersection()
    
    if not intersection:
        print("Error: No overlapping video IDs found between Marisa and Subjects directories.")
    else:
        # Pop a video ID to test with
        test_video_id = 1
        print(f"Testing pipeline with Video ID: {test_video_id}")
        
        paths = get_dyad_paths(BASE_DIR, test_video_id)

        # 1. Instantiate the class!
        dyad_analyzer = DYADAnalyzer()

        # 2. Load the data into memory
        dyad_analyzer.load_dyad_data(
            subj_csv=paths['subj_csv'], 
            intv_csv=paths['intv_csv'], 
            subj_json=paths['subj_json'], 
            intv_json=paths['intv_json']
        )
        
        # 3. Call the counting method (it automatically reads self.subj_intervals)
        marisa_count, subject_count = dyad_analyzer.count_marisa_subject_smiles()
        
        print(f"Results for Video {test_video_id} -> Marisa smiles: {marisa_count}, Subject smiles: {subject_count}")

        delay_axis, cosine_scores = dyad_analyzer.calculate_global_cosine_shifts(1)
        print(cosine_scores)


        stats_df = dyad_analyzer.analyze_interval_cosine_stats(test_name='zscore', max_shift_sec=1)
        
        if stats_df is not None and not stats_df.empty:
            print(f"\n--- Interval Statistics for Video {test_video_id} ---")
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            
            print(stats_df.round(3))
            
            csv_filename = f"Smile_Synchrony_Stats_Video_{test_video_id}.csv"
            stats_df.to_csv(csv_filename, index=False)
            print(f"\nSaved statistics to {csv_filename}")
        else:
            print("No valid intervals found for analysis.")


        # Check for reactive smiles using your custom algorithm
        # merge_gap_frames=10 (Close fragmented smiles)
        # reaction_window_frames=30 (Look 1 second ahead for a reaction)
        reactivity_df = dyad_analyzer.detect_reactive_smiles(
            test_name='zscore', 
            merge_gap_frames=15, 
            reaction_window_frames=30
        )
        
        print(f"\n--- Reactive Smile Analysis for Video {test_video_id} ---")
        print(reactivity_df)
        
        # Calculate exactly how many merged smiles got a reaction
        total_merged_smiles = len(reactivity_df)
        reactions = len(reactivity_df[reactivity_df['status'] == 'REACTIVE_SMILE'])
        
        if total_merged_smiles > 0:
            print(f"\nThe subject reacted to {reactions} out of {total_merged_smiles} interviewer smiles.")
            print(f"Reaction Rate: {(reactions/total_merged_smiles)*100:.1f}%")

        # VISUALIZE THE PIPELINE (Zoomed in on the first 100 seconds)
        dyad_analyzer.visualize_reactivity_pipeline(
            reactivity_df=reactivity_df,  # <--- Pass the table here!
            test_name='zscore',
            merge_gap_frames=30,
            s_time=0.0,      # Zoom start time
            e_time=600.0     # Zoom end time
        )


        valid_reactions = reactivity_df[reactivity_df['status'] == 'REACTIVE_SMILE']

        if not valid_reactions.empty:
            best_interval_index = valid_reactions.index[0]
            print(f"\nGenerating Simulation for successful interval index: {best_interval_index}")
            
            dyad_analyzer.simulate_cosine_shift_for_interval(
                test_name='zscore',
                interval_index=best_interval_index, 
                max_shift_sec=2.0 
            )
        else:
            print("No reactive smiles found to simulate.")