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
import matplotlib.gridspec as gridspec

OPENFACE_MARISA_DIR = '/home/amir/Ados_lab/videos/openface_marisa'
OPENFACE_SUBJECTS_DIR = '/home/amir/Ados_lab/videos/openface_subjects'
ALL_INTERVALS_JSON_FILE = 'all_intervals.json'
TEST_NAME = 'zscore'
BASE_DIR = '/home/amir/Ados_lab/videos'

# --- GLOBAL AESTHETIC PALETTE ---
C_INTV = '#7209B7'       # Deep Purple
C_INTV_LIGHT = '#B56576' # Muted Rose/Purple 
C_SUBJ = '#00AFB9'       # Vibrant Emerald/Teal
C_SUBJ_LIGHT = '#6CB4B8' # Muted Emerald
C_WINDOW = '#F77F00'     # Bright Amber (Search Window)
C_REACTION = '#F07167'   # Coral Pink (Reaction Star/Line)
C_SYNC = '#FFD166'       # Gold (Simultaneous Smiles)
C_TEXT = '#212529'       # Dark Slate (Text)
C_BG = '#FDFDFD'         # Off-white canvas
C_ACCENT = '#2A9D8F'     
FONT_FAMILY = "Segoe UI, Roboto, Helvetica, Arial, sans-serif"
# --------------------------------

def get_dyad_paths(base_dir, video_id):
    paths = {
        'subj_csv': None, 'subj_json': None,
        'intv_csv': None, 'intv_json': None
    }
    intv_dir = os.path.join(base_dir, f'openface_marisa/{video_id}')
    subj_dir = os.path.join(base_dir, f'openface_subjects/{video_id}')
    
    def find_files(target_dir):
        if not os.path.exists(target_dir): return None, None
        csv_files = glob.glob(os.path.join(target_dir, '*.csv'))
        json_files = glob.glob(os.path.join(target_dir, '*all_intervals.json'))
        return csv_files[0] if csv_files else None, json_files[0] if json_files else None

    paths['subj_csv'], paths['subj_json'] = find_files(subj_dir)
    paths['intv_csv'], paths['intv_json'] = find_files(intv_dir)
    return paths
    

def get_dyad_intersection(marisa_dir=OPENFACE_MARISA_DIR, subjects_dir=OPENFACE_SUBJECTS_DIR):
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
        if not os.path.exists(csv_path): return None
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
                return df
        df['Smooth_Smile_Product'] = df['AU12_r_smooth'] * df['AU06_r_smooth']
        return df

    def _load_intervals(self, json_path):
        if not os.path.exists(json_path): return {}
        with open(json_path, 'r') as f: return json.load(f)

    def load_dyad_data(self, subj_csv, intv_csv, subj_json, intv_json):
        self.subj_df = self._process_signal(subj_csv)
        self.subj_intervals = self._load_intervals(subj_json)
        self.intv_df = self._process_signal(intv_csv)
        self.intv_intervals = self._load_intervals(intv_json)

    def count_marisa_subject_smiles(self, test_name=TEST_NAME):
        marisa_count = len(self.intv_intervals.get(test_name, []))
        subject_count = len(self.subj_intervals.get(test_name, []))
        return marisa_count, subject_count

    def get_smile_intersections(self, test_name=TEST_NAME):
        """Calculates the exact overlapping intervals where BOTH subjects are smiling."""
        intv_ints = self.intv_intervals.get(test_name, [])
        subj_ints = self.subj_intervals.get(test_name, [])

        sorted_i = sorted(intv_ints, key=lambda x: x[0])
        sorted_s = sorted(subj_ints, key=lambda x: x[0])

        intersections = []
        i, j = 0, 0

        while i < len(sorted_i) and j < len(sorted_s):
            low = max(sorted_i[i][0], sorted_s[j][0])
            high = min(sorted_i[i][1], sorted_s[j][1])

            if low < high:  
                intersections.append([round(low, 3), round(high, 3)])

            if sorted_i[i][1] < sorted_s[j][1]:
                i += 1
            else:
                j += 1

        return intersections

    def analyze_and_plot_smile_durations(self, test_name=TEST_NAME, bin_size=0.5, out_dir="."):
        intv_ints = self.intv_intervals.get(test_name, [])
        subj_ints = self.subj_intervals.get(test_name, [])
        
        i_durations = [e - s for s, e in intv_ints]
        s_durations = [e - s for s, e in subj_ints]
        
        i_mean = np.mean(i_durations) if i_durations else 0.0
        i_std = np.std(i_durations) if i_durations else 0.0
        s_mean = np.mean(s_durations) if s_durations else 0.0
        s_std = np.std(s_durations) if s_durations else 0.0

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=i_durations, name='Interviewer', marker_color=C_INTV, opacity=0.75, xbins=dict(start=0, size=bin_size)))
        fig.add_trace(go.Histogram(x=s_durations, name='Subject', marker_color=C_SUBJ, opacity=0.75, xbins=dict(start=0, size=bin_size)))
        
        fig.update_layout(
            barmode='overlay', 
            title=f"<span style='font-size:24px; color:{C_TEXT}'><b>Smile Duration Distribution</b></span><br><span style='font-size:16px; color:gray'>Intv: Mean={i_mean:.2f}s (±{i_std:.2f}s) | Subj: Mean={s_mean:.2f}s (±{s_std:.2f}s)</span>",
            xaxis_title="Smile Length (seconds)", yaxis_title="Frequency (Count)",
            font=dict(family=FONT_FAMILY, color=C_TEXT), paper_bgcolor=C_BG, plot_bgcolor='white', height=600, width=900
        )
        fig.update_yaxes(gridcolor='#F1F3F5', zerolinecolor='#DEE2E6')
        fig.update_xaxes(gridcolor='#F1F3F5', zerolinecolor='#DEE2E6')
        
        output_path = os.path.join(out_dir, "Smile_Duration_Histogram.html")
        fig.write_html(output_path)
        self.logger.info(f"Saved {output_path}")

        return {'i_mean': i_mean, 'i_std': i_std, 's_mean': s_mean, 's_std': s_std}

    def get_smile_intensities(self, test_name=TEST_NAME):
        """Calculates the mean and std of smile intensity (Smooth_Smile_Product) during valid intervals."""
        if self.subj_df is None or self.intv_df is None: return 0.0, 0.0, 0.0, 0.0
        
        i_ints = self.intv_intervals.get(test_name, [])
        s_ints = self.subj_intervals.get(test_name, [])
        
        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values
        
        def get_intensities(intervals, sig):
            intensities = []
            for start, end in intervals:
                idx = np.where((ts >= start) & (ts <= end))[0]
                if len(idx) > 0:
                    intensities.extend(sig[idx])
            return np.mean(intensities) if intensities else 0.0, np.std(intensities) if intensities else 0.0

        i_mean_int, i_std_int = get_intensities(i_ints, i_sig)
        s_mean_int, s_std_int = get_intensities(s_ints, s_sig)
        
        return i_mean_int, i_std_int, s_mean_int, s_std_int

    def calculate_global_cosine_shifts(self, max_shift_sec=5.0):
        if self.subj_df is None or self.intv_df is None: return None, None
        min_len = min(len(self.subj_df), len(self.intv_df))
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].reset_index(drop=True)
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].reset_index(drop=True)

        max_shift_frames = int(max_shift_sec * self.fps)
        range_resolution_frames = range(0, max_shift_frames + 1)
        cosine_scores, delay_axis = [], []
        norm_i = np.linalg.norm(i_sig)

        for shift_frames in range_resolution_frames:
            shifted_subject = s_sig.shift(-shift_frames, fill_value=0)
            dot_product = np.dot(i_sig, shifted_subject)
            norm_s = np.linalg.norm(shifted_subject)
            cos_sim = (dot_product / (norm_i * norm_s)) if norm_i > 0 and norm_s > 0 else 0.0
            cosine_scores.append(cos_sim)
            delay_axis.append(shift_frames / self.fps)
            
        return np.array(delay_axis), np.array(cosine_scores)
    
    def analyze_interval_cosine_stats(self, test_name='zscore', max_shift_sec=5.0):
        if self.subj_df is None or self.intv_df is None: return None
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
                s_segment = s_sig[shifted_idx[valid_mask]]
                i_seg_trunc = i_sig[base_idx[valid_mask]]
                
                norm_s, n_i_trunc = np.linalg.norm(s_segment), np.linalg.norm(i_seg_trunc)
                cos_sim = np.dot(i_seg_trunc, s_segment) / (n_i_trunc * norm_s) if n_i_trunc > 0 and norm_s > 0 else 0.0
                interval_scores.append(cos_sim)

            interval_scores = np.array(interval_scores)
            max_idx, min_idx = np.argmax(interval_scores), np.argmin(interval_scores)

            results.append({
                'Start_Time': start_t, 'End_Time': end_t, 'Duration': round(end_t - start_t, 2),
                'Max_Cos_Sim': interval_scores[max_idx], 'Optimal_Delay_Sec': delay_axis[max_idx],          
                'Min_Cos_Sim': interval_scores[min_idx], 'Worst_Delay_Sec': delay_axis[min_idx],            
                'Mean_Cos_Sim': np.mean(interval_scores), 'Std_Cos_Sim': np.std(interval_scores)             
            })
        return pd.DataFrame(results)

    def detect_reactive_smiles(self, test_name=TEST_NAME, merge_gap_frames=10, reaction_window_frames=30, trigger='intv'):
        if trigger == 'intv':
            trigger_intervals = self.intv_intervals.get(test_name, [])
            response_intervals = self.subj_intervals.get(test_name, [])
            t_name, r_name = 'intv', 'subj'
        else:
            trigger_intervals = self.subj_intervals.get(test_name, [])
            response_intervals = self.intv_intervals.get(test_name, [])
            t_name, r_name = 'subj', 'intv'

        if not trigger_intervals:
            return pd.DataFrame()

        merge_gap_sec = merge_gap_frames / self.fps
        reaction_window_sec = reaction_window_frames / self.fps
        sorted_trig = sorted(trigger_intervals, key=lambda x: x[0])
        
        merged_trig = []
        current_start, current_end = sorted_trig[0]
        for start, end in sorted_trig[1:]:
            if (start - current_end) <= merge_gap_sec:
                current_end = max(current_end, end)
            else:
                merged_trig.append((current_start, current_end))
                current_start, current_end = start, end
        merged_trig.append((current_start, current_end))

        results = []
        for t_start, t_end in merged_trig:
            valid_responses = [s for s in response_intervals if t_start <= s[0] <= (t_start + reaction_window_sec)]

            if valid_responses:
                r_start, r_end = valid_responses[0] 
                status = "REACTIVE_SMILE"
                delay_sec = round(r_start - t_start, 3)
                delay_frames = int(delay_sec * self.fps)
            else:
                r_start, r_end, delay_sec, delay_frames = None, None, None, None
                status = "NO_REACTION"

            results.append({
                f'merged_{t_name}_start': round(t_start, 3),
                f'merged_{t_name}_end': round(t_end, 3),
                'merged_duration_sec': round(t_end - t_start, 3),
                'status': status,
                f'{r_name}_response_start': round(r_start, 3) if r_start else None,
                'delay_sec': delay_sec,
                'delay_frames': delay_frames
            })
        return pd.DataFrame(results)

    def detect_lookback_smiles(self, test_name=TEST_NAME, merge_gap_frames=15, lookback_window_frames=30, reference='subj'):
        if reference == 'subj':
            ref_intervals = self.subj_intervals.get(test_name, [])
            prec_intervals = self.intv_intervals.get(test_name, [])
            ref_name, prec_name = 'subj', 'intv'
        else:
            ref_intervals = self.intv_intervals.get(test_name, [])
            prec_intervals = self.subj_intervals.get(test_name, [])
            ref_name, prec_name = 'intv', 'subj'
        
        merge_gap_sec = merge_gap_frames / self.fps
        lookback_window_sec = lookback_window_frames / self.fps
        
        sorted_ref = sorted(ref_intervals, key=lambda x: x[0])
        if not sorted_ref: return pd.DataFrame()
        
        merged_ref = []
        current_start, current_end = sorted_ref[0]
        for start, end in sorted_ref[1:]:
            if (start - current_end) <= merge_gap_sec:
                current_end = max(current_end, end)
            else:
                merged_ref.append((current_start, current_end))
                current_start, current_end = start, end
        merged_ref.append((current_start, current_end))
        
        results = []
        for ref_start, ref_end in merged_ref:
            preceding = [i for i in prec_intervals if (ref_start - lookback_window_sec) <= i[0] <= ref_start]
            
            if preceding:
                p_start, p_end = preceding[-1] 
                status = f"PRECEDED_BY_{prec_name.upper()}"
                delay_sec = round(ref_start - p_start, 3)
            else:
                p_start, p_end, delay_sec = None, None, None
                status = f"SPONTANEOUS_{ref_name.upper()}"
                
            results.append({
                f'merged_{ref_name}_start': round(ref_start, 3),
                f'merged_{ref_name}_end': round(ref_end, 3),
                'status': status,
                f'preceding_{prec_name}_start': round(p_start, 3) if p_start else None,
                'lookback_delay_sec': delay_sec
            })
        return pd.DataFrame(results)

    def visualize_intersections(self, test_name=TEST_NAME, s_time=0.0, e_time=60.0, video_id="Unknown", out_dir=".", show_plot=False):
        if self.subj_df is None or self.intv_df is None: return
        
        intersections = self.get_smile_intersections(test_name)
        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        
        e_time = e_time or ts.max()
        valid_idx = (ts >= s_time) & (ts <= e_time)
        ts_zoom = ts[valid_idx]
        i_sig_zoom = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values[valid_idx]
        s_sig_zoom = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values[valid_idx]

        def format_intervals(intervals):
            return [(s, max(0.1, e - s)) for s, e in intervals if (e >= s_time) and (s <= e_time)]
            
        f_intersections = format_intervals(intersections)

        fig = plt.figure(figsize=(16, 7))
        fig.patch.set_facecolor(C_BG)
        fig.suptitle(f"Simultaneous Smiles (Intersection) | Video {video_id}", fontsize=18, fontweight='bold', color=C_TEXT, y=0.96)
        fig.text(0.5, 0.90, f"Time Window: {s_time}s to {e_time}s", ha='center', fontsize=12, color='gray')

        gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 0.5], hspace=0.3)
        ax_sig = fig.add_subplot(gs[0, 0])
        ax_sync = fig.add_subplot(gs[1, 0], sharex=ax_sig)

        for ax in [ax_sig, ax_sync]:
            ax.set_facecolor('white')
            ax.grid(color='#E9ECEF', linestyle='-', linewidth=1.2, alpha=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CED4DA')
            ax.spines['bottom'].set_color('#CED4DA')
            ax.tick_params(colors=C_TEXT, length=0) 

        # --- Continuous Signals ---
        ax_sig.plot(ts_zoom, i_sig_zoom, color=C_INTV, label='Interviewer Signal', linewidth=2.5, alpha=0.9)
        ax_sig.plot(ts_zoom, s_sig_zoom, color=C_SUBJ, label='Subject Signal', linewidth=2.5, alpha=0.9)
        ax_sig.fill_between(ts_zoom, i_sig_zoom, color=C_INTV, alpha=0.08)
        ax_sig.fill_between(ts_zoom, s_sig_zoom, color=C_SUBJ, alpha=0.08)
        ax_sig.set_title("Smoothed Signal Traces", fontsize=14, fontweight='bold', color=C_TEXT, loc='left')
        ax_sig.legend(loc="upper right", frameon=False, fontsize=11, labelcolor=C_TEXT)

        # --- Synchronous Smiles ---
        if f_intersections:
            ax_sync.broken_barh(f_intersections, (10, 8), facecolors=C_SYNC, edgecolor='white', linewidth=0.5)
        ax_sync.set_yticks([14])
        ax_sync.set_yticklabels(['Intersection\n(Both)'], fontweight='bold', color=C_TEXT)
        ax_sync.set_title("Simultaneous Smiles", fontsize=12, fontweight='bold', color=C_TEXT, loc='left')
        ax_sync.set_xlabel("Video Timestamp (seconds)", fontsize=11, fontweight='bold', color=C_TEXT)
        ax_sync.tick_params(axis='x', pad=15)

        sync_marker = mlines.Line2D([], [], color='none', marker='s', markerfacecolor=C_SYNC, markersize=12, label='Simultaneous Smile')
        fig.legend(handles=[sync_marker], loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=True, facecolor='white', edgecolor='#E0E0E0', fontsize=11)

        plt.subplots_adjust(top=0.83, bottom=0.15, left=0.08, right=0.95)
        
        output_img = os.path.join(out_dir, f"Simultaneous_Smiles_Video_{video_id}.png")
        plt.savefig(output_img, dpi=300, facecolor=C_BG)
        self.logger.info(f"Saved {output_img}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def visualize_reactivity_pipeline(self, df_i2s, df_s2i, df_lookback_s, df_lookback_i, test_name=TEST_NAME, merge_gap_frames=15, reaction_window_frames=30, s_time=0.0, e_time=60.0, video_id="Unknown", out_dir=".", show_plot=False):
        if self.subj_df is None or self.intv_df is None: return

        intv_intervals = self.intv_intervals.get(test_name, [])
        subj_intervals = self.subj_intervals.get(test_name, [])

        merge_gap_sec = merge_gap_frames / self.fps
        reaction_window_sec = reaction_window_frames / self.fps
        
        def merge_intervals(intervals):
            if not intervals: return []
            sorted_ints = sorted(intervals, key=lambda x: x[0])
            merged, c_start, c_end = [], sorted_ints[0][0], sorted_ints[0][1]
            for start, end in sorted_ints[1:]:
                if (start - c_end) <= merge_gap_sec: c_end = max(c_end, end)
                else: merged.append((c_start, c_end)); c_start, c_end = start, end
            merged.append((c_start, c_end))
            return merged

        merged_intv = merge_intervals(intv_intervals)
        merged_subj = merge_intervals(subj_intervals)

        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        
        e_time = e_time or ts.max()
        valid_idx = (ts >= s_time) & (ts <= e_time)
        ts_zoom, i_sig_zoom, s_sig_zoom = ts[valid_idx], self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values[valid_idx], self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values[valid_idx]

        def format_intervals(intervals):
            return [(s, max(0.1, e - s)) for s, e in intervals if (e >= s_time) and (s <= e_time)]
            
        r_intv, m_intv = format_intervals(intv_intervals), format_intervals(merged_intv)
        r_subj, m_subj = format_intervals(subj_intervals), format_intervals(merged_subj)

        fig = plt.figure(figsize=(16, 15))
        fig.patch.set_facecolor(C_BG)
        fig.suptitle(f"Comprehensive Synchrony Pipeline | Video {video_id}", fontsize=20, fontweight='bold', color=C_TEXT, y=0.96)
        fig.text(0.5, 0.93, f"Time Window: {s_time}s to {e_time}s", ha='center', fontsize=12, color='gray')

        gs = gridspec.GridSpec(4, 2, height_ratios=[1.2, 0.8, 1.2, 1.2], hspace=0.5, wspace=0.15)
        
        ax_sig = fig.add_subplot(gs[0, :])
        ax_dl = fig.add_subplot(gs[1, 0], sharex=ax_sig)
        ax_dr = fig.add_subplot(gs[1, 1], sharex=ax_sig)
        ax_rl = fig.add_subplot(gs[2, 0], sharex=ax_sig)
        ax_rr = fig.add_subplot(gs[2, 1], sharex=ax_sig)
        ax_lb_l = fig.add_subplot(gs[3, 0], sharex=ax_sig) 
        ax_lb_r = fig.add_subplot(gs[3, 1], sharex=ax_sig) 

        for ax in [ax_sig, ax_dl, ax_dr, ax_rl, ax_rr, ax_lb_l, ax_lb_r]:
            ax.set_facecolor('white')
            ax.grid(color='#E9ECEF', linestyle='-', linewidth=1.2, alpha=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CED4DA')
            ax.spines['bottom'].set_color('#CED4DA')
            ax.tick_params(colors=C_TEXT, length=0) 

        # --- ROW 1: Continuous Signals ---
        ax_sig.plot(ts_zoom, i_sig_zoom, color=C_INTV, label='Interviewer Signal', linewidth=2.5, alpha=0.9)
        ax_sig.plot(ts_zoom, s_sig_zoom, color=C_SUBJ, label='Subject Signal', linewidth=2.5, alpha=0.9)
        ax_sig.fill_between(ts_zoom, i_sig_zoom, color=C_INTV, alpha=0.08)
        ax_sig.fill_between(ts_zoom, s_sig_zoom, color=C_SUBJ, alpha=0.08)
        ax_sig.set_title("Smoothed Signal Traces", fontsize=14, fontweight='bold', color=C_TEXT, loc='left')
        ax_sig.legend(loc="upper right", frameon=False, fontsize=11, labelcolor=C_TEXT)

        # --- ROW 2: Left (Intv Debounce) & Right (Subj Debounce) ---
        ax_dl.broken_barh(r_intv, (10, 6), facecolors=C_INTV_LIGHT, edgecolor='white', linewidth=0.5, alpha=0.9)
        ax_dl.broken_barh(m_intv, (22, 6), facecolors=C_INTV, edgecolor='white', linewidth=0.5)
        ax_dl.set_yticks([13, 25])
        ax_dl.set_yticklabels(['Raw', 'Merged'], fontweight='bold', color=C_TEXT)
        ax_dl.set_title("Interviewer Debouncing", fontsize=12, fontweight='bold', color=C_TEXT, loc='left')

        ax_dr.broken_barh(r_subj, (10, 6), facecolors=C_SUBJ_LIGHT, edgecolor='white', linewidth=0.5, alpha=0.9)
        ax_dr.broken_barh(m_subj, (22, 6), facecolors=C_SUBJ, edgecolor='white', linewidth=0.5)
        ax_dr.set_yticks([13, 25])
        ax_dr.set_yticklabels(['Raw', 'Merged'], fontweight='bold', color=C_TEXT)
        ax_dr.set_title("Subject Debouncing", fontsize=12, fontweight='bold', color=C_TEXT, loc='left')

        # --- ROW 3: Bidirectional Forward Reactivity ---
        def plot_reactivity(ax, df, trig_rects, resp_rects, t_col_start, r_col_start, c_trig, c_resp, l_trig, l_resp):
            windows = [(row[t_col_start], reaction_window_sec) for _, row in df.iterrows() if s_time <= row[t_col_start] <= e_time]
            if windows: ax.broken_barh(windows, (10, 18), facecolors=C_WINDOW, alpha=0.15, edgecolor='none')

            ax.broken_barh(trig_rects, (22, 6), facecolors=c_trig, edgecolor='white', linewidth=0.5)
            ax.broken_barh(resp_rects, (10, 6), facecolors=c_resp, edgecolor='white', linewidth=0.5, alpha=0.9)
            
            valid = df[df['status'] == 'REACTIVE_SMILE']
            for _, row in valid.iterrows():
                t_s, r_s, dly = row[t_col_start], row[r_col_start], row['delay_sec']
                if s_time <= t_s <= e_time or s_time <= r_s <= e_time:
                    ax.plot(r_s, 17, marker='*', color='#FFD166', markersize=14, markeredgecolor=C_REACTION, markeredgewidth=1.5, zorder=5)
                    ax.text(r_s, 19, f'+{dly}s', fontsize=9, color=C_REACTION, fontweight='bold', ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor=C_REACTION, boxstyle='round,pad=0.2', alpha=0.8))
                    ax.axvline(x=r_s, color=C_REACTION, linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
                    ax.plot(r_s, 0, marker='v', color=C_REACTION, markersize=10, transform=ax.get_xaxis_transform(), clip_on=False, zorder=10)
                    ax.text(r_s, -0.08, f'{r_s:.2f}s', color=C_REACTION, fontweight='bold', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=10)

            ax.set_yticks([13, 25])
            ax.set_yticklabels([f'{l_resp}\nResponses', f'{l_trig}\nTriggers'], fontweight='bold', color=C_TEXT)
            ax.set_title(f"Forward: {l_trig} triggers {l_resp}", fontsize=13, fontweight='bold', color=C_TEXT, loc='left')
            ax.set_xlabel("Video Timestamp (seconds)", fontsize=11, fontweight='bold', color=C_TEXT)
            ax.tick_params(axis='x', pad=15)

        plot_reactivity(ax_rl, df_i2s, m_intv, r_subj, 'merged_intv_start', 'subj_response_start', C_INTV, C_SUBJ, "Interviewer", "Subject")
        plot_reactivity(ax_rr, df_s2i, m_subj, r_intv, 'merged_subj_start', 'intv_response_start', C_SUBJ, C_INTV, "Subject", "Interviewer")
        
        # --- ROW 4: Bidirectional Backward (Lookback) Reactivity ---
        def plot_lookback(ax, df, ref_rects, prec_rects, ref_col_start, prec_col_start, c_ref, c_prec, l_ref, l_prec, status_target):
            lb_windows = [(row[ref_col_start] - reaction_window_sec, reaction_window_sec) for _, row in df.iterrows() if s_time <= row[ref_col_start] <= e_time]
            if lb_windows: ax.broken_barh(lb_windows, (10, 18), facecolors=C_WINDOW, alpha=0.15, edgecolor='none')

            ax.broken_barh(ref_rects, (22, 6), facecolors=c_ref, edgecolor='white', linewidth=0.5)
            ax.broken_barh(prec_rects, (10, 6), facecolors=c_prec, edgecolor='white', linewidth=0.5, alpha=0.9)

            valid_lb = df[df['status'] == status_target]
            for _, row in valid_lb.iterrows():
                r_s = row[ref_col_start]
                p_s = row[prec_col_start]
                dly = row['lookback_delay_sec']
                
                if s_time <= r_s <= e_time or s_time <= p_s <= e_time:
                    ax.plot(p_s, 17, marker='*', color='#FFD166', markersize=14, markeredgecolor=C_REACTION, markeredgewidth=1.5, zorder=5)
                    ax.text(p_s, 19, f'-{dly}s', fontsize=9, color=C_REACTION, fontweight='bold', ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor=C_REACTION, boxstyle='round,pad=0.2', alpha=0.8))
                    ax.axvline(x=p_s, color=C_REACTION, linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
                    ax.plot(p_s, 0, marker='v', color=C_REACTION, markersize=10, transform=ax.get_xaxis_transform(), clip_on=False, zorder=10)
                    ax.text(p_s, -0.08, f'{p_s:.2f}s', color=C_REACTION, fontweight='bold', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=10)

            ax.set_yticks([13, 25])
            ax.set_yticklabels([f'{l_prec}\n(Preceding)', f'{l_ref}\n(Reference)'], fontweight='bold', color=C_TEXT)
            ax.set_title(f"Backward: {l_prec} smiled before {l_ref}?", fontsize=13, fontweight='bold', color=C_TEXT, loc='left')
            ax.set_xlabel("Video Timestamp (seconds)", fontsize=11, fontweight='bold', color=C_TEXT)
            ax.tick_params(axis='x', pad=15)

        plot_lookback(ax_lb_l, df_lookback_s, m_subj, r_intv, 'merged_subj_start', 'preceding_intv_start', C_SUBJ, C_INTV, 'Subject', 'Interviewer', 'PRECEDED_BY_INTV')
        plot_lookback(ax_lb_r, df_lookback_i, m_intv, r_subj, 'merged_intv_start', 'preceding_subj_start', C_INTV, C_SUBJ, 'Interviewer', 'Subject', 'PRECEDED_BY_SUBJ')

        window_patch = mpatches.Patch(facecolor=C_WINDOW, alpha=0.2, label=f'{reaction_window_sec}s Search Window')
        star_marker = mlines.Line2D([], [], color='none', marker='*', markerfacecolor='#FFD166', markeredgecolor=C_REACTION, markersize=12, label='Reaction Matched')
        
        fig.legend(handles=[window_patch, star_marker], loc='upper right', bbox_to_anchor=(0.95, 0.96), frameon=True, facecolor='white', edgecolor='#E0E0E0', fontsize=11)

        plt.subplots_adjust(top=0.90, bottom=0.06, left=0.08, right=0.95)
        
        output_img = os.path.join(out_dir, f"Reactivity_Pipeline_Video_{video_id}.png")
        plt.savefig(output_img, dpi=300, facecolor=C_BG)
        self.logger.info(f"Saved {output_img}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def simulate_cosine_shift_for_interval(self, test_name=TEST_NAME, interval_index=0, max_shift_sec=3.0, out_dir="."):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if self.subj_df is None or self.intv_df is None: return

        intervals = self.intv_intervals.get(test_name, [])
        if not intervals or interval_index >= len(intervals): return

        start_t, end_t = intervals[interval_index]
        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values

        base_idx = np.where((ts >= start_t) & (ts <= end_t))[0]
        if len(base_idx) == 0: return

        i_segment_raw = i_sig[base_idx]
        norm_i = np.linalg.norm(i_segment_raw)
        if norm_i == 0: return

        max_shift_frames = int(max_shift_sec * self.fps)
        delay_axis = [shift / self.fps for shift in range(max_shift_frames + 1)]
        
        all_s_segments_raw, cosine_scores = [], []
        best_score, best_delay = -1.0, 0.0

        for shift in range(max_shift_frames + 1):
            shifted_idx = base_idx + shift
            valid_mask = shifted_idx < min_len
            s_seg_raw = s_sig[shifted_idx[valid_mask]]
            i_seg_trunc = i_sig[base_idx[valid_mask]]
            
            if len(s_seg_raw) < len(base_idx):
                pad_width = len(base_idx) - len(s_seg_raw)
                s_seg_raw = np.pad(s_seg_raw, (0, pad_width), 'constant', constant_values=0)
                i_seg_trunc = np.pad(i_seg_trunc, (0, pad_width), 'constant', constant_values=0)

            norm_s, n_i_trunc = np.linalg.norm(s_seg_raw), np.linalg.norm(i_seg_trunc)
            cos_sim = np.dot(i_seg_trunc, s_seg_raw) / (n_i_trunc * norm_s) if n_i_trunc > 0 and norm_s > 0 else 0.0

            cosine_scores.append(cos_sim)
            all_s_segments_raw.append(s_seg_raw) 
            if cos_sim > best_score: best_score, best_delay = cos_sim, delay_axis[shift]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=False, row_heights=[0.20, 0.50, 0.30], vertical_spacing=0.1, subplot_titles=("Macro Context", "Sliding Window Animation", "Synchrony Profile"))
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
                    {"y": [i_sig, s_sig, i_segment_raw, all_s_segments_raw[i], cosine_scores, [cosine_scores[i]]], "x": [ts, ts, ts[base_idx], ts[base_idx], delay_axis, [delay_sec]]},
                    {"title": f"<span style='font-size:24px; color:{C_TEXT}'><b>Synchrony Simulator</b></span><br><span style='font-size:16px; color:gray'>Interval: {start_t:.2f}s to {end_t:.2f}s | Current Delay: <b style='color:{C_SUBJ}'>{delay_sec:.2f}s</b> | Synchrony Score: <b style='color:{C_ACCENT}'>{cosine_scores[i]:.3f}</b></span>"}
                ], label=f"{delay_sec:.2f}s"
            )
            steps.append(step)

        sliders = [dict(active=0, currentvalue={"visible": False}, pad={"t": 30}, steps=steps, x=0.05, len=0.90, font=dict(color=C_TEXT), tickcolor=C_TEXT)]
        fig.update_layout(font=dict(family=FONT_FAMILY, color=C_TEXT), paper_bgcolor='#F8F9FA', plot_bgcolor='white', sliders=sliders, title=f"<span style='font-size:24px; color:{C_TEXT}'><b>Synchrony Simulator</b></span><br><span style='font-size:16px; color:gray'>Interval: {start_t:.2f}s to {end_t:.2f}s | Current Delay: <b style='color:{C_SUBJ}'>0.00s</b> | Synchrony Score: <b style='color:{C_ACCENT}'>{cosine_scores[0]:.3f}</b></span>", height=900, width=1100, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        standard_ticks = list(np.arange(0, max_shift_sec + 0.1, 0.5))
        all_ticks = sorted(standard_ticks + [best_delay]) if best_delay not in standard_ticks else standard_ticks
        tick_texts = [f"<b style='color:{C_ACCENT}; font-size:14px'>{t:.2f}s<br>Peak</b>" if round(t,2)==round(best_delay,2) else f"{t:.1f}s" for t in all_ticks]
        fig.update_xaxes(title_text="Subject Shift Delay (seconds)", title_font=dict(size=12, color='gray'), tickvals=all_ticks, ticktext=tick_texts, row=3, col=1)
        
        output_path = os.path.join(out_dir, f"Cosine_Simulation_Interval_{interval_index}.html")
        fig.write_html(output_path)

    def analyze_and_plot_pdf_kl_divergence(self, test_name=TEST_NAME, interval_index=0, bin_size="auto", out_dir="."):
        if self.subj_df is None or self.intv_df is None: return
        intervals = self.intv_intervals.get(test_name, [])
        if not intervals or interval_index >= len(intervals): return

        start_t, end_t = intervals[interval_index]
        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values
        base_idx = np.where((ts >= start_t) & (ts <= end_t))[0]
        if len(base_idx) == 0: return

        i_raw, s_raw = i_sig[base_idx], s_sig[base_idx]
        epsilon = 1e-10
        i_pdf = (i_raw + epsilon) / np.sum(i_raw + epsilon)
        s_pdf = (s_raw + epsilon) / np.sum(s_raw + epsilon)

        kl_intv_subj, kl_subj_intv = entropy(pk=i_pdf, qk=s_pdf), entropy(pk=s_pdf, qk=i_pdf)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[0.55, 0.45], vertical_spacing=0.15, subplot_titles=(f"<b>Time-Series PDF</b><br><span style='font-size:14px; color:gray'>KL(Intv || Subj): <b style='color:{C_INTV}'>{kl_intv_subj:.4f}</b> &nbsp; | &nbsp; KL(Subj || Intv): <b style='color:{C_SUBJ}'>{kl_subj_intv:.4f}</b></span>", "<b>Intensity Distribution (Histogram)</b>"))
        fig.add_trace(go.Scatter(x=ts[base_idx], y=i_pdf, mode='lines', name='Interviewer PDF', line=dict(color=C_INTV, width=3, shape='spline'), fill='tozeroy', fillcolor=f'rgba(114, 9, 183, 0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts[base_idx], y=s_pdf, mode='lines', name='Subject PDF', line=dict(color=C_SUBJ, width=3, shape='spline', dash='dash'), fill='tozeroy', fillcolor=f'rgba(0, 175, 185, 0.2)'), row=1, col=1)

        max_val = max(np.max(i_raw), np.max(s_raw))
        final_bin_size = max(0.5, max_val / 20.0) if bin_size == "auto" else bin_size
        fig.add_trace(go.Histogram(x=i_raw, name='Interviewer', marker_color=C_INTV, opacity=0.75, xbins=dict(start=0, end=max_val, size=final_bin_size)), row=2, col=1)
        fig.add_trace(go.Histogram(x=s_raw, name='Subject', marker_color=C_SUBJ, opacity=0.75, xbins=dict(start=0, end=max_val, size=final_bin_size)), row=2, col=1)

        fig.update_layout(barmode='overlay', font=dict(family=FONT_FAMILY, color=C_TEXT), paper_bgcolor='#FDFDFD', plot_bgcolor='white', title=f"<span style='font-size:24px; color:{C_TEXT}'><b>Distribution Analysis</b></span><br><span style='font-size:16px; color:gray'>Interval: {start_t:.2f}s to {end_t:.2f}s</span>", height=800, width=1000, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        output_path = os.path.join(out_dir, f"PDF_KL_Divergence_Interval_{interval_index}.html")
        fig.write_html(output_path)
    
    def analyze_and_plot_full_signal_pdf_kl_divergence(self, bin_size="auto", out_dir="."):
        if self.subj_df is None or self.intv_df is None: return
        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        s_raw = self.subj_df['Smooth_Smile_Product'].iloc[:min_len].values
        i_raw = self.intv_df['Smooth_Smile_Product'].iloc[:min_len].values

        epsilon = 1e-10
        i_pdf = (i_raw + epsilon) / np.sum(i_raw + epsilon)
        s_pdf = (s_raw + epsilon) / np.sum(s_raw + epsilon)
        kl_intv_subj, kl_subj_intv = entropy(pk=i_pdf, qk=s_pdf), entropy(pk=s_pdf, qk=i_pdf) 

        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[0.55, 0.45], vertical_spacing=0.15, subplot_titles=(f"<b>Full Video Time-Series PDF</b><br><span style='font-size:14px; color:gray'>KL(Intv || Subj): <b style='color:{C_INTV}'>{kl_intv_subj:.4f}</b> &nbsp; | &nbsp; KL(Subj || Intv): <b style='color:{C_SUBJ}'>{kl_subj_intv:.4f}</b></span>", "<b>Global Intensity Distribution (Log-Scaled Histogram)</b>"))
        fig.add_trace(go.Scatter(x=ts, y=i_pdf, mode='lines', name='Interviewer PDF', line=dict(color=C_INTV, width=1.5), fill='tozeroy', fillcolor=f'rgba(114, 9, 183, 0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=s_pdf, mode='lines', name='Subject PDF', line=dict(color=C_SUBJ, width=1.5), fill='tozeroy', fillcolor=f'rgba(0, 175, 185, 0.2)'), row=1, col=1)

        max_val = max(np.max(i_raw), np.max(s_raw))
        final_bin_size = max(0.5, max_val / 40.0) if bin_size == "auto" else bin_size
        fig.add_trace(go.Histogram(x=i_raw, name='Interviewer', marker_color=C_INTV, opacity=0.75, xbins=dict(start=0, end=max_val, size=final_bin_size)), row=2, col=1)
        fig.add_trace(go.Histogram(x=s_raw, name='Subject', marker_color=C_SUBJ, opacity=0.75, xbins=dict(start=0, end=max_val, size=final_bin_size)), row=2, col=1)

        fig.update_layout(barmode='overlay', font=dict(family=FONT_FAMILY, color=C_TEXT), paper_bgcolor='#FDFDFD', plot_bgcolor='white', title=f"<span style='font-size:24px; color:{C_TEXT}'><b>Global Distribution & Divergence Analysis</b></span>", height=850, width=1100, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(type="log", row=2, col=1)
        
        output_path = os.path.join(out_dir, "PDF_KL_Divergence_Full_Video.html")
        fig.write_html(output_path)


if __name__ == "__main__":
    
    SHOW_PLOTS = False # <--- TOGGLE THIS TO TRUE IF YOU WANT POP-UPS
    RUNONE = False # <--- TOGGLE THIS TO TRUE IF YOU WANT TO RUN ONLY ONE VIDEO (FOR DEBUGGING)
    for video_id in get_dyad_intersection():
            test_video_id = video_id
            print(f"\n[{test_video_id}] ===============================================")
            print(f"[{test_video_id}] Processing pipeline...")
            
            out_dir = os.path.join(BASE_DIR, f"results/Results_Video_{test_video_id}")
            os.makedirs(out_dir, exist_ok=True)
            
            paths = get_dyad_paths(BASE_DIR, test_video_id)
            dyad_analyzer = DYADAnalyzer()
            dyad_analyzer.load_dyad_data(subj_csv=paths['subj_csv'], intv_csv=paths['intv_csv'], subj_json=paths['subj_json'], intv_json=paths['intv_json'])
            
            # --- Core Counts and Duration ---
            marisa_count, subject_count = dyad_analyzer.count_marisa_subject_smiles()
            smile_intersections = dyad_analyzer.get_smile_intersections(test_name='zscore')
            num_intersections = len(smile_intersections)
            dur_stats = dyad_analyzer.analyze_and_plot_smile_durations(bin_size=0.5, out_dir=out_dir)

            # --- New Features: Overlap Lengths & Intensities ---
            overlap_lengths = [end - start for start, end in smile_intersections]
            avg_overlap_len = np.mean(overlap_lengths) if overlap_lengths else 0.0

            i_mean_int, i_std_int, s_mean_int, s_std_int = dyad_analyzer.get_smile_intensities(test_name='zscore')

            # --- Cosine Stats ---
            max_shift_sec = 1.0
            total_shifts = int(max_shift_sec * dyad_analyzer.fps) + 1
            stats_df = dyad_analyzer.analyze_interval_cosine_stats(test_name='zscore', max_shift_sec=max_shift_sec)
            
            mg_frames = 15
            rw_frames = 30
            
            # --- Reactivity DataFrames ---
            df_i2s = dyad_analyzer.detect_reactive_smiles(test_name='zscore', merge_gap_frames=mg_frames, reaction_window_frames=rw_frames, trigger='intv')
            df_s2i = dyad_analyzer.detect_reactive_smiles(test_name='zscore', merge_gap_frames=mg_frames, reaction_window_frames=rw_frames, trigger='subj')
            df_lb_s = dyad_analyzer.detect_lookback_smiles(test_name='zscore', merge_gap_frames=mg_frames, lookback_window_frames=rw_frames, reference='subj')
            df_lb_i = dyad_analyzer.detect_lookback_smiles(test_name='zscore', merge_gap_frames=mg_frames, lookback_window_frames=rw_frames, reference='intv')

            # --- New Features: Reactivity Summary Stats ---
            i2s_reacts = df_i2s[df_i2s['status'] == 'REACTIVE_SMILE'] if not df_i2s.empty else pd.DataFrame()
            num_i2s_reactions = len(i2s_reacts)
            
            avg_i2s_delay = i2s_reacts['delay_sec'].mean() if num_i2s_reactions > 0 else 0.0
            std_i2s_delay = i2s_reacts['delay_sec'].std()
            std_i2s_delay = std_i2s_delay if pd.notna(std_i2s_delay) else 0.0 # Catch NaN if only 1 reaction
            
            perc_subj_is_reaction = (num_i2s_reactions / subject_count * 100) if subject_count > 0 else 0.0
            perc_intv_got_reaction = (num_i2s_reactions / len(df_i2s) * 100) if not df_i2s.empty else 0.0

            # --- MASTER CSV WRITING ---
            if stats_df is not None and not stats_df.empty:
                csv_filename = os.path.join(out_dir, f"Smile_Synchrony_Stats_Video_{test_video_id}.csv")
                
                with open(csv_filename, 'w') as f:
                    f.write("=== DYAD SMILE SYNCHRONY REPORT ===\n\n")
                    f.write("1. GENERAL STATISTICS\n")
                    f.write("--- Counts & Durations ---\n")
                    f.write(f"Interviewer Total Smiles,{marisa_count}\n")
                    f.write(f"Subject Total Smiles,{subject_count}\n")
                    f.write(f"Interviewer Smile Duration Mean (s),{dur_stats['i_mean']:.3f}\n")
                    f.write(f"Interviewer Smile Duration Std (s),{dur_stats['i_std']:.3f}\n")
                    f.write(f"Subject Smile Duration Mean (s),{dur_stats['s_mean']:.3f}\n")
                    f.write(f"Subject Smile Duration Std (s),{dur_stats['s_std']:.3f}\n\n")

                    f.write("--- Intensities (Smooth_Smile_Product) ---\n")
                    f.write(f"Interviewer Smile Intensity Mean,{i_mean_int:.3f}\n")
                    f.write(f"Interviewer Smile Intensity Std,{i_std_int:.3f}\n")
                    f.write(f"Subject Smile Intensity Mean,{s_mean_int:.3f}\n")
                    f.write(f"Subject Smile Intensity Std,{s_std_int:.3f}\n\n")

                    f.write("--- Intersections (Overlaps) ---\n")
                    f.write(f"Simultaneous Smiles (Intersections) Count,{num_intersections}\n")
                    f.write(f"Average Intersection Duration (s),{avg_overlap_len:.3f}\n\n")

                    f.write("--- Reactivity Summary (Interviewer -> Subject) ---\n")
                    f.write(f"Total Reactions (Subject reacted to Intv),{num_i2s_reactions}\n")
                    f.write(f"Average Delay of Reaction (s),{avg_i2s_delay:.3f}\n")
                    f.write(f"Std Dev of Delay of Reaction (s),{std_i2s_delay:.3f}\n")
                    f.write(f"% of Subject's smiles that were reactions,{perc_subj_is_reaction:.1f}%\n")
                    f.write(f"% of Interviewer's smiles that got a reaction,{perc_intv_got_reaction:.1f}%\n\n")
                    
                    f.write("2. COSINE SHIFT ANALYSIS\n")
                    f.write(f"Total Shifts Evaluated,{total_shifts}\n")
                    f.write(f"Max Shift Seconds,{max_shift_sec}\n")
                    f.write(f"Frames Per Second,{dyad_analyzer.fps}\n\n")

                stats_df.to_csv(csv_filename, mode='a', index=False)

                with open(csv_filename, 'a') as f:
                    f.write("\n3. FORWARD REACTIVITY (Interviewer -> Subject)\n")
                    f.write(f"Merge Gap Frames,{mg_frames}\n")
                    f.write(f"Reaction Window Frames,{rw_frames}\n\n")
                df_i2s.to_csv(csv_filename, mode='a', index=False)

                with open(csv_filename, 'a') as f:
                    f.write("\n4. FORWARD REACTIVITY (Subject -> Interviewer)\n")
                    f.write(f"Merge Gap Frames,{mg_frames}\n")
                    f.write(f"Reaction Window Frames,{rw_frames}\n\n")
                df_s2i.to_csv(csv_filename, mode='a', index=False)

                with open(csv_filename, 'a') as f:
                    f.write("\n5. BACKWARD REACTIVITY (Subject Lookback -> Interviewer)\n")
                    f.write(f"Merge Gap Frames,{mg_frames}\n")
                    f.write(f"Lookback Window Frames,{rw_frames}\n\n")
                df_lb_s.to_csv(csv_filename, mode='a', index=False)

                with open(csv_filename, 'a') as f:
                    f.write("\n6. BACKWARD REACTIVITY (Interviewer Lookback -> Subject)\n")
                    f.write(f"Merge Gap Frames,{mg_frames}\n")
                    f.write(f"Lookback Window Frames,{rw_frames}\n\n")
                df_lb_i.to_csv(csv_filename, mode='a', index=False)

                with open(csv_filename, 'a') as f:
                    f.write("\n7. RAW INTERVALS (Start, End)\n")
                    f.write("--- Interviewer ---\n")
                    for start, end in dyad_analyzer.intv_intervals.get('zscore', []): f.write(f"{start},{end}\n")
                    f.write("--- Subject ---\n")
                    for start, end in dyad_analyzer.subj_intervals.get('zscore', []): f.write(f"{start},{end}\n")
                    
                    f.write("\n8. SIMULTANEOUS SMILES (INTERSECTIONS) [Start, End]\n")
                    for start, end in smile_intersections: f.write(f"{start},{end}\n")
                        
                print(f"[{test_video_id}] Saved comprehensive master report: {csv_filename}")
            else:
                print(f"[{test_video_id}] No valid intervals found for analysis.")

            dyad_analyzer.visualize_reactivity_pipeline(
                df_i2s=df_i2s, 
                df_s2i=df_s2i, 
                df_lookback_s=df_lb_s,  
                df_lookback_i=df_lb_i,  
                test_name='zscore',
                merge_gap_frames=mg_frames,
                reaction_window_frames=rw_frames,
                s_time=0.0,      
                e_time=600.0,    
                video_id=test_video_id,
                out_dir=out_dir,
                show_plot=SHOW_PLOTS
            )
            
            dyad_analyzer.visualize_intersections(
                test_name='zscore',
                s_time=0.0,
                e_time=600.0,
                video_id=test_video_id,
                out_dir=out_dir,
                show_plot=SHOW_PLOTS
            )

            valid_reactions = df_i2s[df_i2s['status'] == 'REACTIVE_SMILE']
            if not valid_reactions.empty:
                best_interval_index = valid_reactions.index[0]
                dyad_analyzer.simulate_cosine_shift_for_interval(test_name='zscore', interval_index=best_interval_index, max_shift_sec=2.0, out_dir=out_dir)
                dyad_analyzer.analyze_and_plot_pdf_kl_divergence(test_name='zscore', interval_index=best_interval_index, bin_size=1.0, out_dir=out_dir)
                
            dyad_analyzer.analyze_and_plot_full_signal_pdf_kl_divergence(bin_size=1.0, out_dir=out_dir)
            print(f"[{test_video_id}] Completed successfully!")
            if RUNONE: break
