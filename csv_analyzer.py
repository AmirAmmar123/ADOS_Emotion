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
from plotly.subplots import make_subplots
from scipy.stats import entropy

OPENFACE_MARISA_DIR = '/home/amir/Ados_lab/videos/openface_marisa'
OPENFACE_SUBJECTS_DIR = '/home/amir/Ados_lab/videos/openface_subjects'
ALL_INTERVALS_JSON_FILE = 'all_intervals.json'
TEST_NAME = 'zscore'
BASE_DIR = '/home/amir/Ados_lab/videos'

# --- GLOBAL AESTHETIC PALETTE ---
C_INTV = '#7209B7'       # Deep Purple
C_INTV_LIGHT = '#B56576' # Muted Rose/Purple (Pre-merging)
C_SUBJ = '#00AFB9'       # Vibrant Emerald/Teal (Subject)
C_WINDOW = '#F77F00'     # Bright Amber (Search Window)
C_REACTION = '#F07167'   # Coral Pink (Reaction Star/Line)
C_TEXT = '#212529'       # Dark Slate (Text)
C_BG = '#FDFDFD'         # Off-white canvas
C_ACCENT = '#2A9D8F'     
FONT_FAMILY = "Segoe UI, Roboto, Helvetica, Arial, sans-serif"
# --------------------------------

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
        self.logger.info("Processing Subject data...")
        self.subj_df = self._process_signal(subj_csv)
        self.subj_intervals = self._load_intervals(subj_json)
                
        self.logger.info("Processing Interviewer data...")
        self.intv_df = self._process_signal(intv_csv)
        self.intv_intervals = self._load_intervals(intv_json)

    def count_marisa_subject_smiles(self, test_name=TEST_NAME):
        marisa_count = len(self.intv_intervals.get(test_name, []))
        subject_count = len(self.subj_intervals.get(test_name, []))
        return marisa_count, subject_count

    def calculate_global_cosine_shifts(self, max_shift_sec=5.0):
        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return None, None

        self.logger.info(f"Calculating Global Cosine Similarity for shifts up to {max_shift_sec} seconds...")

        min_len = min(len(self.subj_df), len(self.intv_df))
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].reset_index(drop=True)
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].reset_index(drop=True)

        max_shift_frames = int(max_shift_sec * self.fps)
        range_resolution_frames = range(0, max_shift_frames + 1)
        
        cosine_scores = []
        delay_axis = []
        norm_i = np.linalg.norm(i_sig)

        for shift_frames in range_resolution_frames:
            shifted_subject = s_sig.shift(-shift_frames, fill_value=0)
            dot_product = np.dot(i_sig, shifted_subject)
            norm_s = np.linalg.norm(shifted_subject)
            
            if norm_i > 0 and norm_s > 0:
                cos_sim = dot_product / (norm_i * norm_s)
            else:
                cos_sim = 0.0
                
            cosine_scores.append(cos_sim)
            delay_axis.append(shift_frames / self.fps)
            
        return np.array(delay_axis), np.array(cosine_scores)
    
    def analyze_interval_cosine_stats(self, test_name='zscore', max_shift_sec=5.0):
        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return None

        self.logger.info(f"Calculating Interval-based Cosine Statistics (Max Shift: {max_shift_sec}s)...")

        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values

        intervals = self.intv_intervals.get(test_name, [])
        max_shift_frames = int(max_shift_sec * self.fps)
        delay_axis = np.array([f / self.fps for f in range(max_shift_frames + 1)])
        results = []

        for start_t, end_t in intervals:
            base_idx = np.where((ts >= start_t) & (ts <= end_t))[0]
            if len(base_idx) == 0: continue 

            i_segment = i_sig[base_idx]
            norm_i = np.linalg.norm(i_segment)
            if norm_i == 0: continue

            interval_scores = []

            for shift_frames in range(max_shift_frames + 1):
                shifted_idx = base_idx + shift_frames
                valid_mask = shifted_idx < min_len 
                valid_shifted_idx = shifted_idx[valid_mask] 
                valid_base_idx = base_idx[valid_mask]       
                
                s_segment = s_sig[valid_shifted_idx]
                i_seg_trunc = i_sig[valid_base_idx]
                
                norm_s = np.linalg.norm(s_segment)
                n_i_trunc = np.linalg.norm(i_seg_trunc)

                if n_i_trunc > 0 and norm_s > 0:
                    cos_sim = np.dot(i_seg_trunc, s_segment) / (n_i_trunc * norm_s)
                else:
                    cos_sim = 0.0

                interval_scores.append(cos_sim)

            interval_scores = np.array(interval_scores)
            max_idx = np.argmax(interval_scores)
            min_idx = np.argmin(interval_scores)

            results.append({
                'Start_Time': start_t,
                'End_Time': end_t,
                'Duration': round(end_t - start_t, 2),
                'Max_Cos_Sim': interval_scores[max_idx],           
                'Optimal_Delay_Sec': delay_axis[max_idx],          
                'Min_Cos_Sim': interval_scores[min_idx],           
                'Worst_Delay_Sec': delay_axis[min_idx],            
                'Mean_Cos_Sim': np.mean(interval_scores),          
                'Std_Cos_Sim': np.std(interval_scores)             
            })

        return pd.DataFrame(results)
    
    def classify_smile_responses(self, test_name=TEST_NAME, max_response_sec=3.0):
        intv_intervals = self.intv_intervals.get(test_name, [])
        subj_intervals = self.subj_intervals.get(test_name, [])
        results = []

        for i_start, i_end in intv_intervals:
            responses = [s for s in subj_intervals if i_start < s[0] <= (i_start + max_response_sec)]
            if not responses:
                status = "NO_RESPONSE"
                details = None
            else:
                s_start, s_end = responses[0]
                overlap = max(0.0, min(i_end, s_end) - max(i_start, s_start))
                
                status = "OVERLAP" if overlap > 0 else "DELAYED_RESPONSE" 
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
        return pd.DataFrame(results)
    

    def detect_reactive_smiles(self, test_name=TEST_NAME, merge_gap_frames=10, reaction_window_frames=30):
        intv_intervals = self.intv_intervals.get(test_name, [])
        subj_intervals = self.subj_intervals.get(test_name, [])

        if not intv_intervals:
            self.logger.warning("No interviewer intervals found to process.")
            return pd.DataFrame()

        merge_gap_sec = merge_gap_frames / self.fps
        reaction_window_sec = reaction_window_frames / self.fps
        sorted_intv = sorted(intv_intervals, key=lambda x: x[0])
        
        merged_intv = []
        current_start, current_end = sorted_intv[0]

        for start, end in sorted_intv[1:]:
            time_gap = start - current_end
            if time_gap <= merge_gap_sec:
                current_end = max(current_end, end)
            else:
                merged_intv.append((current_start, current_end))
                current_start, current_end = start, end
        merged_intv.append((current_start, current_end))
        
        self.logger.info(f"Merged {len(intv_intervals)} raw intervals down to {len(merged_intv)} continuous intervals.")

        results = []
        for i_start, i_end in merged_intv:
            valid_responses = [s for s in subj_intervals if i_start <= s[0] <= (i_start + reaction_window_sec)]

            if valid_responses:
                s_start, s_end = valid_responses[0] 
                status = "REACTIVE_SMILE"
                delay_sec = round(s_start - i_start, 3)
                delay_frames = int(delay_sec * self.fps)
            else:
                s_start, s_end, delay_sec, delay_frames = None, None, None, None
                status = "NO_REACTION"

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

        # Setup Figure
        fig, axs = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
        fig.patch.set_facecolor(C_BG)
        fig.suptitle(f"Smile Synchrony & Reactivity Pipeline | Video {video_id}", 
                     fontsize=18, fontweight='bold', color=C_TEXT, y=0.97)
        fig.text(0.5, 0.94, f"Time Window: {s_time}s to {e_time}s", ha='center', fontsize=12, color='gray')

        for ax in axs:
            ax.set_facecolor('white')
            ax.grid(color='#E9ECEF', linestyle='-', linewidth=1.2, alpha=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CED4DA')
            ax.spines['bottom'].set_color('#CED4DA')
            ax.tick_params(colors=C_TEXT, length=0) 

        # --- Panel 1: Before and After Merging ---
        axs[0].broken_barh(raw_intv_rects, (10, 6), facecolors=C_INTV_LIGHT, edgecolor='white', linewidth=0.5, alpha=0.9)
        axs[0].broken_barh(merged_intv_rects, (22, 6), facecolors=C_INTV, edgecolor='white', linewidth=0.5)
        axs[0].set_yticks([13, 25])
        axs[0].set_yticklabels(['Raw Intervals', 'Merged'], fontweight='bold', color=C_TEXT)
        axs[0].set_title(f"Interviewer Signal Debouncing (Max Gap: {merge_gap_frames} frames)", fontsize=13, fontweight='bold', color=C_TEXT, loc='left')

        # --- Panel 2: Continuous Signals ---
        axs[1].plot(ts_zoom, i_sig_zoom, color=C_INTV, label='Interviewer Signal', linewidth=2.5, alpha=0.9)
        axs[1].plot(ts_zoom, s_sig_zoom, color=C_SUBJ, label='Subject Signal', linewidth=2.5, alpha=0.9)
        axs[1].fill_between(ts_zoom, i_sig_zoom, color=C_INTV, alpha=0.08)
        axs[1].fill_between(ts_zoom, s_sig_zoom, color=C_SUBJ, alpha=0.08)
        axs[1].set_title("Smoothed Signal Traces", fontsize=13, fontweight='bold', color=C_TEXT, loc='left')
        axs[1].legend(loc="upper right", frameon=False, fontsize=10, labelcolor=C_TEXT)

        # --- Panel 3: Merged Intv vs Raw Subject WITH AXIS INDICATORS ---
        search_windows = []
        for _, row in reactivity_df.iterrows():
            i_start = row['merged_intv_start']
            if s_time <= i_start <= e_time:
                search_windows.append((i_start, reaction_window_sec))
        
        if search_windows:
            axs[2].broken_barh(search_windows, (10, 18), facecolors=C_WINDOW, alpha=0.2, edgecolor='none')

        axs[2].broken_barh(merged_intv_rects, (22, 6), facecolors=C_INTV, edgecolor='white', linewidth=0.5)
        axs[2].broken_barh(raw_subj_rects, (10, 6), facecolors=C_SUBJ, edgecolor='white', linewidth=0.5, alpha=0.9)
        
        valid_reactions = reactivity_df[reactivity_df['status'] == 'REACTIVE_SMILE']
        for _, row in valid_reactions.iterrows():
            i_start = row['merged_intv_start']
            s_start = row['subj_response_start']
            delay = row['delay_sec']
            
            if s_time <= i_start <= e_time or s_time <= s_start <= e_time:
                axs[2].plot(s_start, 17, marker='*', color='#FFD166', markersize=16, markeredgecolor=C_REACTION, markeredgewidth=1.5, zorder=5)
                axs[2].text(s_start, 19, f'+{delay}s', fontsize=10, color=C_REACTION, fontweight='bold', ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor=C_REACTION, boxstyle='round,pad=0.2', alpha=0.8))
                axs[2].axvline(x=s_start, color=C_REACTION, linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
                axs[2].plot(s_start, 0, marker='v', color=C_REACTION, markersize=10, transform=axs[2].get_xaxis_transform(), clip_on=False, zorder=10)
                axs[2].text(s_start, -0.08, f'{s_start:.2f}s', color=C_REACTION, fontweight='bold', ha='center', va='top', transform=axs[2].get_xaxis_transform(), fontsize=10)

        axs[2].set_yticks([13, 25])
        axs[2].set_yticklabels(['Subject\nResponses', 'Interviewer\nTriggers'], fontweight='bold', color=C_TEXT)
        axs[2].set_title(f"Interaction Timeline (Yellow: {reaction_window_sec}s Search Window)", fontsize=13, fontweight='bold', color=C_TEXT, loc='left')
        axs[2].set_xlabel("Video Timestamp (seconds)", fontsize=12, fontweight='bold', color=C_TEXT)
        axs[2].tick_params(axis='x', pad=18)
        
        window_patch = mpatches.Patch(facecolor=C_WINDOW, alpha=0.3, label='Valid Search Window')
        star_marker = mlines.Line2D([], [], color='none', marker='*', markerfacecolor='#FFD166', markeredgecolor=C_REACTION, markersize=12, label='Reaction Matched')
        axs[2].legend(handles=[window_patch, star_marker], loc='upper right', frameon=True, facecolor='white', edgecolor='#E0E0E0', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.93], h_pad=3.0)
        plt.show()

    def simulate_cosine_shift_for_interval(self, test_name=TEST_NAME, interval_index=0, max_shift_sec=3.0):
        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return

        intervals = self.intv_intervals.get(test_name, [])
        if not intervals or interval_index >= len(intervals):
            self.logger.warning(f"Invalid interval index. Only {len(intervals)} intervals available.")
            return

        start_t, end_t = intervals[interval_index]
        self.logger.info(f"Building simulation for Interval {interval_index} ({start_t}s - {end_t}s)...")
        
        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values

        base_idx = np.where((ts >= start_t) & (ts <= end_t))[0]
        if len(base_idx) == 0: return

        i_segment_raw = i_sig[base_idx]
        norm_i = np.linalg.norm(i_segment_raw)
        if norm_i == 0:
            self.logger.warning("Interviewer signal is completely flat here. Choose another interval.")
            return

        max_shift_frames = int(max_shift_sec * self.fps)
        delay_axis = [shift / self.fps for shift in range(max_shift_frames + 1)]
        
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

            norm_s = np.linalg.norm(s_seg_raw)
            n_i_trunc = np.linalg.norm(i_seg_trunc)
            
            if n_i_trunc > 0 and norm_s > 0:
                cos_sim = np.dot(i_seg_trunc, s_seg_raw) / (n_i_trunc * norm_s)
            else:
                cos_sim = 0.0

            cosine_scores.append(cos_sim)
            all_s_segments_raw.append(s_seg_raw) 

            if cos_sim > best_score:
                best_score, best_delay = cos_sim, delay_axis[shift]

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=False,
            row_heights=[0.20, 0.50, 0.30], vertical_spacing=0.1,
            subplot_titles=("Macro Context (Full Video Timeline)", "Micro Context (Sliding Window Animation)", "Time-Lagged Cross-Correlation (Synchrony Profile)")
        )

        fig.add_trace(go.Scatter(x=ts, y=i_sig, mode='lines', name='Intv (Full)', line=dict(color=C_INTV, width=1.5), opacity=0.4, hoverinfo='none'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=s_sig, mode='lines', name='Subj (Full)', line=dict(color=C_SUBJ, width=1.5), opacity=0.4, hoverinfo='none'), row=1, col=1)
        fig.add_vrect(x0=start_t, x1=end_t, fillcolor="#F4A261", opacity=0.4, line_width=0, row=1, col=1)

        fig.add_trace(go.Scatter(x=ts[base_idx], y=i_segment_raw, mode='lines', name='Interviewer Trigger', line=dict(color=C_INTV, width=4, shape='spline'), fill='tozeroy', fillcolor=f'rgba(114, 9, 183, 0.1)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=ts[base_idx], y=all_s_segments_raw[0], mode='lines', name='Subject Response', line=dict(color=C_SUBJ, width=4, shape='spline'), fill='tozeroy', fillcolor=f'rgba(0, 175, 185, 0.2)'), row=2, col=1)

        fig.add_trace(go.Scatter(x=delay_axis, y=cosine_scores, mode='lines', name='Similarity Curve', line=dict(color=C_ACCENT, width=3, shape='spline'), fill='tozeroy', fillcolor=f'rgba(42, 157, 143, 0.1)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=[delay_axis[0]], y=[cosine_scores[0]], mode='markers', name='Current Shift', marker=dict(color='white', size=14, line=dict(color=C_ACCENT, width=3))), row=3, col=1)

        steps = []
        for i, delay_sec in enumerate(delay_axis):
            step = dict(
                method="update",
                args=[
                    {"y": [
                        i_sig, s_sig, i_segment_raw, all_s_segments_raw[i], cosine_scores, [cosine_scores[i]]       
                    ],
                    "x": [
                        ts, ts, ts[base_idx], ts[base_idx], delay_axis, [delay_sec]              
                    ]},
                    {"title": f"<span style='font-size:24px; color:{C_TEXT}'><b>Synchrony Simulator</b></span><br><span style='font-size:16px; color:gray'>Interval: {start_t:.2f}s to {end_t:.2f}s | Current Delay: <b style='color:{C_SUBJ}'>{delay_sec:.2f}s</b> | Synchrony Score: <b style='color:{C_ACCENT}'>{cosine_scores[i]:.3f}</b></span>"}
                ],
                label=f"{delay_sec:.2f}s"
            )
            steps.append(step)

        sliders = [dict(
            active=0, currentvalue={"visible": False}, pad={"t": 30}, steps=steps,
            x=0.05, len=0.90, font=dict(color=C_TEXT), tickcolor=C_TEXT
        )]

        fig.update_layout(
            font=dict(family=FONT_FAMILY, color=C_TEXT),
            paper_bgcolor='#F8F9FA', plot_bgcolor='white', sliders=sliders,
            title=f"<span style='font-size:24px; color:{C_TEXT}'><b>Synchrony Simulator</b></span><br><span style='font-size:16px; color:gray'>Interval: {start_t:.2f}s to {end_t:.2f}s | Current Delay: <b style='color:{C_SUBJ}'>0.00s</b> | Synchrony Score: <b style='color:{C_ACCENT}'>{cosine_scores[0]:.3f}</b></span>",
            height=900, width=1100, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        for row in range(1, 4):
            fig.update_yaxes(gridcolor='#F1F3F5', zerolinecolor='#DEE2E6', row=row, col=1)
            fig.update_xaxes(gridcolor='#F1F3F5', zerolinecolor='#DEE2E6', row=row, col=1)

        fig.update_yaxes(title_text="Intensity", row=2, col=1, title_font=dict(size=12, color='gray'))
        fig.update_xaxes(title_text="Base Video Time (seconds)", row=2, col=1, title_font=dict(size=12, color='gray'))
        fig.update_yaxes(title_text="Cosine Score", range=[0, 1.05], row=3, col=1, title_font=dict(size=12, color='gray'))
        
        standard_ticks = list(np.arange(0, max_shift_sec + 0.1, 0.5))
        if best_delay not in standard_ticks:
            all_ticks = sorted(standard_ticks + [best_delay])
        else:
            all_ticks = standard_ticks

        tick_texts = []
        for tick in all_ticks:
            if round(tick, 2) == round(best_delay, 2):
                tick_texts.append(f"<b style='color:{C_ACCENT}; font-size:14px'>{tick:.2f}s<br>Peak</b>")
            else:
                tick_texts.append(f"{tick:.1f}s")

        fig.update_xaxes(title_text="Subject Shift Delay (seconds)", title_font=dict(size=12, color='gray'), tickvals=all_ticks, ticktext=tick_texts, row=3, col=1)

        html_filename = f"Cosine_Simulation_Interval_{interval_index}.html"
        fig.write_html(html_filename)
        print(f"\n---> Peak Synchrony for interval {interval_index} was {best_score:.3f} at a delay of {best_delay:.2f}s.")
        print(f"---> Simulation Saved! Open '{html_filename}' to view.\n")

    def analyze_and_plot_pdf_kl_divergence(self, test_name=TEST_NAME, interval_index=0, bin_size="auto"):
        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return

        intervals = self.intv_intervals.get(test_name, [])
        if not intervals or interval_index >= len(intervals):
            self.logger.warning(f"Invalid interval index. Only {len(intervals)} intervals available.")
            return

        start_t, end_t = intervals[interval_index]
        self.logger.info(f"Analyzing PDF & KL Divergence for Interval {interval_index} ({start_t}s - {end_t}s)...")
        
        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values

        base_idx = np.where((ts >= start_t) & (ts <= end_t))[0]
        if len(base_idx) == 0: return

        i_raw = i_sig[base_idx]
        s_raw = s_sig[base_idx]

        epsilon = 1e-10
        i_safe = i_raw + epsilon
        s_safe = s_raw + epsilon
        
        i_pdf = i_safe / np.sum(i_safe)
        s_pdf = s_safe / np.sum(s_safe)

        kl_intv_subj = entropy(pk=i_pdf, qk=s_pdf) 
        kl_subj_intv = entropy(pk=s_pdf, qk=i_pdf) 

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False,
            row_heights=[0.55, 0.45], vertical_spacing=0.15,
            subplot_titles=(
                f"<b>Time-Series PDF</b><br><span style='font-size:14px; color:gray'>KL(Intv || Subj): <b style='color:{C_INTV}'>{kl_intv_subj:.4f}</b> &nbsp; | &nbsp; KL(Subj || Intv): <b style='color:{C_SUBJ}'>{kl_subj_intv:.4f}</b></span>", 
                "<b>Intensity Distribution (Histogram)</b><br><span style='font-size:14px; color:gray'>Frequency of Smile Intensities ignoring time</span>"
            )
        )

        fig.add_trace(go.Scatter(x=ts[base_idx], y=i_pdf, mode='lines', name='Interviewer PDF', line=dict(color=C_INTV, width=3, shape='spline'), fill='tozeroy', fillcolor=f'rgba(114, 9, 183, 0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts[base_idx], y=s_pdf, mode='lines', name='Subject PDF', line=dict(color=C_SUBJ, width=3, shape='spline', dash='dash'), fill='tozeroy', fillcolor=f'rgba(0, 175, 185, 0.2)'), row=1, col=1)

        max_val = max(np.max(i_raw), np.max(s_raw))
        final_bin_size = max(0.5, max_val / 20.0) if bin_size == "auto" else bin_size

        fig.add_trace(go.Histogram(x=i_raw, name='Interviewer', marker_color=C_INTV, opacity=0.75, xbins=dict(start=0, end=max_val, size=final_bin_size)), row=2, col=1)
        fig.add_trace(go.Histogram(x=s_raw, name='Subject', marker_color=C_SUBJ, opacity=0.75, xbins=dict(start=0, end=max_val, size=final_bin_size)), row=2, col=1)

        fig.update_layout(barmode='overlay')
        fig.update_layout(
            font=dict(family=FONT_FAMILY, color=C_TEXT),
            paper_bgcolor='#FDFDFD', plot_bgcolor='white',     
            title=f"<span style='font-size:24px; color:{C_TEXT}'><b>Distribution & Divergence Analysis</b></span><br><span style='font-size:16px; color:gray'>Interval: {start_t:.2f}s to {end_t:.2f}s</span>",
            height=800, width=1000, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        for row in range(1, 3):
            fig.update_yaxes(gridcolor='#F1F3F5', zerolinecolor='#DEE2E6', row=row, col=1)
            fig.update_xaxes(gridcolor='#F1F3F5', zerolinecolor='#DEE2E6', row=row, col=1)

        fig.update_yaxes(title_text="Probability P(x)", row=1, col=1, title_font=dict(size=12, color='gray'))
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1, title_font=dict(size=12, color='gray'))
        fig.update_yaxes(title_text="Frequency (Frame Count)", row=2, col=1, title_font=dict(size=12, color='gray'))
        fig.update_xaxes(title_text="Raw Smile Intensity (Smooth_Smile_Product)", row=2, col=1, title_font=dict(size=12, color='gray'))

        html_filename = f"PDF_KL_Divergence_Interval_{interval_index}.html"
        fig.write_html(html_filename)
        print(f"\n---> SUCCESS: KL Divergence Calculated!")
        print(f"---> KL(Intv||Subj): {kl_intv_subj:.4f} | KL(Subj||Intv): {kl_subj_intv:.4f}")
        print(f"---> Analysis Saved! Open '{html_filename}' to view.\n")
    
    def analyze_and_plot_full_signal_pdf_kl_divergence(self, bin_size="auto"):
        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return

        self.logger.info("Analyzing PDF & KL Divergence for the FULL SIGNAL...")
        
        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        s_raw = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values
        i_raw = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values

        epsilon = 1e-10
        i_safe = i_raw + epsilon
        s_safe = s_raw + epsilon
        
        i_pdf = i_safe / np.sum(i_safe)
        s_pdf = s_safe / np.sum(s_safe)

        kl_intv_subj = entropy(pk=i_pdf, qk=s_pdf) 
        kl_subj_intv = entropy(pk=s_pdf, qk=i_pdf) 

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False,
            row_heights=[0.55, 0.45], vertical_spacing=0.15,
            subplot_titles=(
                f"<b>Full Video Time-Series PDF</b><br><span style='font-size:14px; color:gray'>KL(Intv || Subj): <b style='color:{C_INTV}'>{kl_intv_subj:.4f}</b> &nbsp; | &nbsp; KL(Subj || Intv): <b style='color:{C_SUBJ}'>{kl_subj_intv:.4f}</b></span>", 
                "<b>Global Intensity Distribution (Log-Scaled Histogram)</b><br><span style='font-size:14px; color:gray'>Frequency of all Smile Intensities across the entire video</span>"
            )
        )

        fig.add_trace(go.Scatter(x=ts, y=i_pdf, mode='lines', name='Interviewer PDF', line=dict(color=C_INTV, width=1.5), fill='tozeroy', fillcolor=f'rgba(114, 9, 183, 0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=s_pdf, mode='lines', name='Subject PDF', line=dict(color=C_SUBJ, width=1.5), fill='tozeroy', fillcolor=f'rgba(0, 175, 185, 0.2)'), row=1, col=1)

        max_val = max(np.max(i_raw), np.max(s_raw))
        final_bin_size = max(0.5, max_val / 40.0) if bin_size == "auto" else bin_size

        fig.add_trace(go.Histogram(x=i_raw, name='Interviewer', marker_color=C_INTV, opacity=0.75, xbins=dict(start=0, end=max_val, size=final_bin_size)), row=2, col=1)
        fig.add_trace(go.Histogram(x=s_raw, name='Subject', marker_color=C_SUBJ, opacity=0.75, xbins=dict(start=0, end=max_val, size=final_bin_size)), row=2, col=1)

        fig.update_layout(barmode='overlay')
        fig.update_layout(
            font=dict(family=FONT_FAMILY, color=C_TEXT),
            paper_bgcolor='#FDFDFD', plot_bgcolor='white',     
            title=f"<span style='font-size:24px; color:{C_TEXT}'><b>Global Distribution & Divergence Analysis (Full Video)</b></span>",
            height=850, width=1100, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        for row in range(1, 3):
            fig.update_yaxes(gridcolor='#F1F3F5', zerolinecolor='#DEE2E6', row=row, col=1)
            fig.update_xaxes(gridcolor='#F1F3F5', zerolinecolor='#DEE2E6', row=row, col=1)

        fig.update_yaxes(title_text="Probability P(x)", row=1, col=1, title_font=dict(size=12, color='gray'))
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1, title_font=dict(size=12, color='gray'))
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08), row=1, col=1) 
        
        fig.update_yaxes(title_text="Frequency (Log Scale)", type="log", row=2, col=1, title_font=dict(size=12, color='gray'))
        fig.update_xaxes(title_text="Raw Smile Intensity", row=2, col=1, title_font=dict(size=12, color='gray'))

        html_filename = "PDF_KL_Divergence_Full_Video.html"
        fig.write_html(html_filename)
        print(f"\n---> SUCCESS: FULL VIDEO KL Divergence Calculated!")
        print(f"---> KL(Intv||Subj): {kl_intv_subj:.4f} | KL(Subj||Intv): {kl_subj_intv:.4f}")
        print(f"---> Analysis Saved! Open '{html_filename}' to view.\n")

if __name__ == "__main__":
    
    intersection = get_dyad_intersection()
    
    if not intersection:
        print("Error: No overlapping video IDs found between Marisa and Subjects directories.")
    else:
        # Pop a video ID to test with
        test_video_id = 27
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
        
        # 3. Call the counting method
        marisa_count, subject_count = dyad_analyzer.count_marisa_subject_smiles()
        
        print(f"Results for Video {test_video_id} -> Marisa smiles: {marisa_count}, Subject smiles: {subject_count}")

        delay_axis, cosine_scores = dyad_analyzer.calculate_global_cosine_shifts(1)

        stats_df = dyad_analyzer.analyze_interval_cosine_stats(test_name='zscore', max_shift_sec=1)
        
        if stats_df is not None and not stats_df.empty:
            print(f"\n--- Interval Statistics for Video {test_video_id} ---")
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            
            print(stats_df.round(3))
            
            csv_filename = f"Smile_Synchrony_Stats_Video_{test_video_id}.csv"
            stats_df.to_csv(csv_filename, index=False)
            
            # --- APPENDING SMILE COUNTS AND RAW INTERVALS ---
            with open(csv_filename, 'a') as f:
                f.write("\n" + "#" * 51 + "\n")
                f.write("SMILE COUNTS\n")
                f.write(f"Interviewer Total Smiles,{marisa_count}\n")
                f.write(f"Subject Total Smiles,{subject_count}\n")
                
                f.write("\n" + "#" * 51 + "\n")
                f.write("INTERVIEWER INTERVALS (Start, End)\n")
                intv_ints = dyad_analyzer.intv_intervals.get('zscore', [])
                for start, end in intv_ints:
                    f.write(f"{start},{end}\n")
                    
                f.write("\n" + "#" * 51 + "\n")
                f.write("SUBJECT INTERVALS (Start, End)\n")
                subj_ints = dyad_analyzer.subj_intervals.get('zscore', [])
                for start, end in subj_ints:
                    f.write(f"{start},{end}\n")
                    
            print(f"\nSaved statistics and raw intervals to {csv_filename}")
        else:
            print("No valid intervals found for analysis.")

        # Check for reactive smiles using your custom algorithm
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

        # VISUALIZE THE PIPELINE (Zoomed in on the first 600 seconds)
        dyad_analyzer.visualize_reactivity_pipeline(
            reactivity_df=reactivity_df, 
            test_name='zscore',
            merge_gap_frames=30,
            reaction_window_frames=30,
            s_time=0.0,      
            e_time=600.0,    
            video_id=test_video_id
        )

        valid_reactions = reactivity_df[reactivity_df['status'] == 'REACTIVE_SMILE']

        if not valid_reactions.empty:
            best_interval_index = valid_reactions.index[0]
            print(f"\nGenerating Simulation for successful interval index: {best_interval_index}")
            
            # The Cosine Similarity Simulation
            dyad_analyzer.simulate_cosine_shift_for_interval(
                test_name='zscore',
                interval_index=best_interval_index, 
                max_shift_sec=2.0 
            )

            # Execution of PDF / KL Divergence Analysis for the interval
            dyad_analyzer.analyze_and_plot_pdf_kl_divergence(
                test_name='zscore',
                interval_index=best_interval_index,
                bin_size=0.25  # Or leave empty for 'auto'
            )
            
        else:
            print("No reactive smiles found to simulate.")

        # Execution of PDF / KL Divergence Analysis for the FULL video
        dyad_analyzer.analyze_and_plot_full_signal_pdf_kl_divergence(bin_size=0.25) # Or leave empty for 'auto'
