import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylogger import Logger
import glob
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

def get_dyad_paths(base_dir, video_id):
    """
    Dynamically finds the CSV and JSON files for both the subject and interviewer
    by searching the directory, ignoring exact file naming variations.
    """
    paths = {
        'subj_csv': None, 'subj_json': None,
        'intv_csv': None, 'intv_json': None
    }
    
    intv_dir = os.path.join(base_dir, f'openface_marisa/{video_id}')
    subj_dir = os.path.join(base_dir, f'openface_subjects/{video_id}')
    
    def find_files(target_dir):
        if not os.path.exists(target_dir):
            print(f"Warning: Directory not found -> {target_dir}")
            return None, None
            
        csv_files = glob.glob(os.path.join(target_dir, '*.csv'))
        json_files = glob.glob(os.path.join(target_dir, '*all_intervals.json'))
        
        csv_path = csv_files[0] if csv_files else None
        json_path = json_files[0] if json_files else None
        
        return csv_path, json_path

    paths['subj_csv'], paths['subj_json'] = find_files(subj_dir)
    paths['intv_csv'], paths['intv_json'] = find_files(intv_dir)
    
    return paths

class DyadicSmileVisualizer:
    """
    Analyzes and compares genuine smiles between a Subject and an Interviewer.
    Plots smoothed signals, applies time-shifts, and calculates a Synchrony Product.
    """
    def __init__(self, fps=30.0, window_size=15):
        self.fps = fps
        self.window_size = window_size
        
        self.subj_df = None
        self.intv_df = None
        self.subj_intervals = {}
        self.intv_intervals = {}
        
        os.makedirs('logs', exist_ok=True)
        self.logger = Logger('DyadicSmileVisualizer', 'logs/dyadic_analysis.log')

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

    def visualize(self, method='baseline', s_time=None, e_time=None, subj_shift_sec=0.0, intv_shift_sec=0.0, video_id="Unknown"):
        """
        Original 4-panel comparison plot.
        """
        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return

        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len]
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len]

        s_frames = int(round(subj_shift_sec * self.fps))
        i_frames = int(round(intv_shift_sec * self.fps))

        shifted_subj = s_sig.shift(s_frames, fill_value=0)
        shifted_intv = i_sig.shift(i_frames, fill_value=0)

        synchrony_product = shifted_subj * shifted_intv

        fig, axs = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
        fig.suptitle(f"Dyadic Synchrony - video {video_id} (Method: {method.upper()} | Subj Shift: {subj_shift_sec}s | Intv Shift: {intv_shift_sec}s)", fontsize=16, fontweight='bold')

        s_intervals = self.subj_intervals.get(method, [])
        i_intervals = self.intv_intervals.get(method, [])

        axs[0].plot(ts, shifted_subj, color='blue', linewidth=2, label=f'Subject (Shifted {subj_shift_sec}s)')
        axs[0].fill_between(ts, 0, 1, where=self.subj_df['bad_tracking'].iloc[:min_len], transform=axs[0].get_xaxis_transform(), color='gray', alpha=0.3, label='Tracking Fail')
        for idx, (start, end) in enumerate(s_intervals):
            axs[0].axvspan(start + subj_shift_sec, end + subj_shift_sec, color='blue', alpha=0.2, label=f'{method.capitalize()}' if idx == 0 else "")
        axs[0].set_title("1. Subject Smoothed Signal (Shifted)")
        axs[0].legend(loc='upper right')
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(ts, shifted_intv, color='firebrick', linewidth=2, label=f'Interviewer (Shifted {intv_shift_sec}s)')
        axs[1].fill_between(ts, 0, 1, where=self.intv_df['bad_tracking'].iloc[:min_len], transform=axs[1].get_xaxis_transform(), color='gray', alpha=0.3, label='Tracking Fail')
        for idx, (start, end) in enumerate(i_intervals):
            axs[1].axvspan(start + intv_shift_sec, end + intv_shift_sec, color='firebrick', alpha=0.2, label=f'{method.capitalize()}' if idx == 0 else "")
        axs[1].set_title("2. Interviewer Smoothed Signal (Shifted)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(ts, shifted_subj, color='blue', linewidth=2, label='Subject')
        axs[2].plot(ts, shifted_intv, color='firebrick', linewidth=2, label='Interviewer')
        axs[2].set_title("3. Overlaid Signals (Check for visual alignment here)")
        axs[2].legend(loc='upper right')
        axs[2].grid(True, alpha=0.3)

        axs[3].plot(ts, synchrony_product, color='purple', linewidth=3, label='Synchrony Product (Subj * Intv)')
        axs[3].set_title("4. The Synchrony Product (Spikes indicate perfect mirrored smiling)")
        axs[3].set_xlabel("Time (seconds)")
        axs[3].legend(loc='upper right')
        axs[3].grid(True, alpha=0.3)

        plt.xlim(max(s_time or ts.min(), ts.min()), min(e_time or ts.max(), ts.max()))
        plt.tight_layout(h_pad=3.0, rect=[0, 0.04, 1, 0.96])
        
        plt.show()


    # =========================================================================
    # NEW FUNCTION: Dedicated 3D Surface Plot
    # =========================================================================
    def visualize_3d_surface(self, s_time=None, e_time=None, max_shift_sec=5.0, video_id="Unknown"):
        """
        Generates an interactive 3D Surface Plot using Plotly.
        """
        if self.subj_df is None or self.intv_df is None:
            self.logger.error("Please load data first.")
            return

        self.logger.info(f"Generating INTERACTIVE 3D Surface Plot for {max_shift_sec} seconds...")

        # 1. Truncate arrays to matching lengths and time bounds
        min_len = min(len(self.subj_df), len(self.intv_df))
        ts = self.subj_df['timestamp'].iloc[:min_len].values
        
        valid_idx = (ts >= (s_time or ts.min())) & (ts <= (e_time or ts.max()))
        X_time = ts[valid_idx]
        
        s_sig = self.subj_df['Smooth_Smile_Product'].iloc[:min_len][valid_idx].reset_index(drop=True)
        i_sig = self.intv_df['Smooth_Smile_Product'].iloc[:min_len][valid_idx].reset_index(drop=True)

        # 2. Setup the exact frame-by-frame resolution
        max_shift_frames = int(max_shift_sec * self.fps)
        range_resolution_frames = range(0, max_shift_frames + 1)
        
        array_result = []
        y_axis_delays = []
        
        # 3. Calculate the 2D function
        for shift_frames in range_resolution_frames:
            shifted_subject = s_sig.shift(-shift_frames, fill_value=0)
            signal = i_sig * shifted_subject
            
            array_result.append(signal.values)
            y_axis_delays.append(shift_frames / self.fps)
            
        # Convert to numpy array and APPLY GAUSSIAN FILTER FOR SMOOTHING
        array_result = np.array(array_result)
        smoothed_array_result = gaussian_filter(array_result, sigma=1.5)

        # --- PLOTLY 3D PLOTTING ---
        fig = go.Figure(data=[go.Surface(
            z=smoothed_array_result,  # Using the SMOOTHED matrix!
            x=X_time,
            y=y_axis_delays,
            colorscale='Viridis',
            hovertemplate='Time: %{x}s<br>Delay: %{y}s<br>Score: %{z:.2f}<extra></extra>'
        )])

        fig.update_layout(
            title=f'Interactive 3D Synchrony Surface - Video {video_id}',
            autosize=False,
            width=1200,
            height=800,
            scene=dict(
                xaxis_title='Time (seconds)',
                yaxis_title='Subject Delay (seconds)',
                zaxis_title='Synchrony Product Score',
                camera=dict(
                    eye=dict(x=-1.5, y=-1.5, z=1.0) # Sets the initial viewing angle
                )
            ),
            margin=dict(l=65, r=50, b=65, t=90)
        )

        # Save the interactive graph as an HTML file instead of trying to open a browser in the terminal
        html_filename = f"3D_Synchrony_Surface_Video_{video_id}.html"
        fig.write_html(html_filename)
        self.logger.info(f"Successfully saved interactive 3D plot to: {html_filename}")
        print(f"\n---> SUCCESS: Open '{html_filename}' in your Chrome/Edge browser to view the 3D plot!\n")


if __name__ == "__main__":
    
    BASE_DIR = '/home/amir/Ados_lab/videos'
    VIDEO_ID = "250" # <-- Just change this number for different videos
    
    paths = get_dyad_paths(BASE_DIR, VIDEO_ID)
    
    if not paths['subj_csv'] or not paths['intv_csv']:
        print(f"Error: Missing CSV files for Video ID {VIDEO_ID}. Aborting.")
    else:
        dyad_analyzer = DyadicSmileVisualizer(fps=30.0, window_size=15)
        
        dyad_analyzer.load_dyad_data(
            subj_csv=paths['subj_csv'], 
            intv_csv=paths['intv_csv'], 
            subj_json=paths['subj_json'], 
            intv_json=paths['intv_json']
        )
        
        # 1. Run the original 4-Panel Visualization
        dyad_analyzer.visualize(
            method='zscore', 
            s_time=0, 
            e_time=5000.0, 
            subj_shift_sec=0, 
            intv_shift_sec=0.0,
            video_id=VIDEO_ID 
        )
        
        # 2. Run the DEDICATED 3D Surface Plot!
        dyad_analyzer.visualize_3d_surface(
            s_time=0, 
            e_time=5000.0, 
            max_shift_sec=1.5, # Will shift up to 2 seconds (60 frames)
            video_id=VIDEO_ID
        )