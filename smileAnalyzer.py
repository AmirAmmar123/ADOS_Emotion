import os
import re
import cv2 
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from mylogger import Logger

class SmileAnalyzer:
    """
    A pipeline for detecting, visualizing, and rendering genuine Duchenne smiles 
    from OpenFace Action Unit data using temporal filtering and feature interaction.
    """
    
    def __init__(self, fps=30.0, window_size=30, peak_thresh=2.0, base_thresh=0.5):
        self.fps = fps
        self.window_size = window_size
        self.peak_thresh = peak_thresh
        self.base_thresh = base_thresh
        
        # State variables to store analysis results for downstream classes
        self.df = None
        self.smart_intervals = []
        self.bad_intervals = []
        self.simple_intervals = []
        self.combo_intervals = []
        
        os.makedirs('logs', exist_ok=True)
        self.logger = Logger('SmileAnalyzer', 'logs/smile_analysis.log')

    @staticmethod
    def _parse_frame_string(val):
        matches = re.findall(r'int64\((\d+)\)', str(val))
        if len(matches) == 2:
            return int(matches[0]), int(matches[1])
        return None, None

    def _extract_intervals_from_csv(self, csv_path, column_name):
        if not csv_path or not os.path.exists(csv_path):
            return []

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self.logger.error(f"Error reading {csv_path}: {e}")
            return []

        intervals_sec = []
        if column_name not in df.columns:
            self.logger.error(f"Column '{column_name}' not found in {csv_path}")
            return intervals_sec

        for _, row in df.iterrows():
            start_frame, end_frame = self._parse_frame_string(row[column_name])
            if start_frame is not None and end_frame is not None:
                start_sec = start_frame / self.fps
                end_sec = end_frame / self.fps
                intervals_sec.append((start_sec, end_sec))
                
        return intervals_sec

    def _find_smart_intervals(self, timestamps, signal):
        intervals = []
        n = len(signal)
        deriv = np.gradient(signal)
        
        peaks = []
        for i in range(1, n - 1):
            if deriv[i-1] > 0 and deriv[i] <= 0:
                if signal[i] >= self.peak_thresh:
                    peaks.append(i)
                    
        used_indices = np.zeros(n, dtype=bool)
        for p_idx in peaks:
            if used_indices[p_idx]:
                continue
                
            start_idx = p_idx
            while start_idx > 0 and signal[start_idx] > self.base_thresh:
                start_idx -= 1
                
            end_idx = p_idx
            while end_idx < n - 1 and signal[end_idx] > self.base_thresh:
                end_idx += 1
                
            intervals.append((timestamps[start_idx], timestamps[end_idx]))
            used_indices[start_idx:end_idx+1] = True
                
        return intervals

    def analyze(self, raw_csv_path, simple_csv_path=None, combo_csv_path=None):
        """
        Processes OpenFace AU data, applies temporal filtering, and extracts smile intervals.
        Saves the results as object attributes and returns them.
        """
        self.logger.info(f"Starting analysis for: {raw_csv_path}")
        
        self.logger.debug("Loading simple and combo interval data...")
        self.simple_intervals = self._extract_intervals_from_csv(simple_csv_path, column_name='peak_lp_simple')
        self.combo_intervals = self._extract_intervals_from_csv(combo_csv_path, column_name='peak_lp_combo')
        
        if not os.path.exists(raw_csv_path):
            self.logger.error(f"Raw signal file not found: {raw_csv_path}")
            raise FileNotFoundError(f"Raw signal file not found: {raw_csv_path}")
            
        df_raw = pd.read_csv(raw_csv_path)
        df_raw.columns = df_raw.columns.str.strip()  
        timestamps = df_raw['timestamp'].values

        if 'AU12_r' in df_raw.columns and 'AU06_r' in df_raw.columns:
            df_raw['AU12_r_unfiltered'] = df_raw['AU12_r'].copy()
            df_raw['AU06_r_unfiltered'] = df_raw['AU06_r'].copy()

        self.logger.debug("Filtering out tracking failures and low-confidence frames...")
        bad_frames = pd.Series(False, index=df_raw.index)
        
        if 'success' in df_raw.columns:
            bad_frames = bad_frames | (df_raw['success'] == 0)
        if 'confidence' in df_raw.columns:
            bad_frames = bad_frames | (df_raw['confidence'] < 0.80)
            
        df_raw['bad_tracking_flag'] = bad_frames
        df_raw.loc[bad_frames, ['AU12_r', 'AU06_r']] = 0.0

        # Extract bad intervals into standard (start_sec, end_sec) tuples using numpy diff
        self.bad_intervals = []
        bad_mask = bad_frames.values.astype(int)
        if bad_mask.any():
            diff = np.diff(np.concatenate(([0], bad_mask, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1
            self.bad_intervals = [(timestamps[s], timestamps[e]) for s, e in zip(starts, ends)]
            self.logger.info(f"Detected {len(self.bad_intervals)} intervals with tracking failure/low confidence.")

        self.logger.debug("Applying Mathematical Pipeline (Multiply -> Smooth -> Multiply)...")
        if 'AU12_r' in df_raw.columns and 'AU06_r' in df_raw.columns:
            df_raw['Raw_Multiplied'] = df_raw['AU12_r'] * df_raw['AU06_r']

        if 'AU12_r' in df_raw.columns:
            df_raw['AU12_r_smooth'] = df_raw['AU12_r'].rolling(window=self.window_size, center=True, min_periods=1).mean()
        if 'AU06_r' in df_raw.columns:
            df_raw['AU06_r_smooth'] = df_raw['AU06_r'].rolling(window=self.window_size, center=True, min_periods=1).mean()
        if 'Raw_Multiplied' in df_raw.columns:
            df_raw['Multiply_Then_Smooth'] = df_raw['Raw_Multiplied'].rolling(window=self.window_size, center=True, min_periods=1).mean()

        self.smart_intervals = []
        if 'AU12_r_smooth' in df_raw.columns and 'AU06_r_smooth' in df_raw.columns:
            df_raw['Smooth_Then_Multiply'] = df_raw['AU12_r_smooth'] * df_raw['AU06_r_smooth']
            self.logger.debug("Detecting Smart Intervals...")
            self.smart_intervals = self._find_smart_intervals(timestamps, df_raw['Smooth_Then_Multiply'].values)
            self.logger.info(f"Successfully detected {len(self.smart_intervals)} genuine smile intervals.")

        self.df = df_raw
        return self.df, self.smart_intervals, self.simple_intervals, self.combo_intervals, self.bad_intervals

    def export_data(self, output_dir, prefix=""):
        """
        Saves the processed DataFrame to CSV and the analyzed intervals to JSON.
        """
        if self.df is None:
            self.logger.error("Cannot export: No data available. Please run analyze() first.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Save the full processed dataframe
        csv_path = os.path.join(output_dir, f"{prefix}processed_signals.csv")
        self.df.to_csv(csv_path, index=False)

        # 2. Save the intervals dictionary as JSON
        json_path = os.path.join(output_dir, f"{prefix}intervals.json")
        data_to_save = {
            "smart_intervals": self.smart_intervals,
            "bad_intervals": self.bad_intervals,
            "legacy_simple_intervals": self.simple_intervals,
            "legacy_combo_intervals": self.combo_intervals
        }
        
        with open(json_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)

        self.logger.info(f"Exported processed DataFrame to: {csv_path}")
        self.logger.info(f"Exported interval data to: {json_path}")


    def visualize(self, df, smart_intervals, simple_extracted, combo_extracted, 
                  ground_truth=None, start_time=None, end_time=None, title="Smile Analysis", save_path=None):
        """
        Generates a 7-panel visualization of the signal processing pipeline.
        Includes boundary logic to constrain view, and can save the plot if save_path is provided.
        """
        self.logger.info("Generating 7-panel visualization plot...")
        ground_truth = ground_truth or []
        timestamps = df['timestamp'].values
        
        fig, axs = plt.subplots(7, 1, figsize=(14, 21), sharex=True, gridspec_kw={'height_ratios': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1]})
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98, x=0.02, ha='left')

        if 'AU12_r_unfiltered' in df.columns and 'AU06_r_unfiltered' in df.columns:
            axs[0].plot(timestamps, df['AU12_r_unfiltered'], color='blue', alpha=0.5, label='Raw AU12_r')
            axs[0].plot(timestamps, df['AU06_r_unfiltered'], color='red', alpha=0.5, label='Raw AU06_r')
            axs[0].set_ylabel('Intensity (0-5)')
            axs[0].set_title('1. BEFORE: Raw Signals (Vulnerable to Tracking Artifacts)')
            axs[0].grid(True, alpha=0.3)
            axs[0].legend(loc='upper right')
            for gt_start, gt_end in ground_truth: axs[0].axvspan(gt_start, gt_end, color='lime', alpha=0.2)

        if 'AU12_r' in df.columns and 'AU06_r' in df.columns:
            axs[1].plot(timestamps, df['AU12_r'], color='blue', alpha=0.8, label='Cleaned AU12_r')
            axs[1].plot(timestamps, df['AU06_r'], color='red', alpha=0.8, label='Cleaned AU06_r')
            if 'bad_tracking_flag' in df.columns:
                axs[1].fill_between(timestamps, 0, 5, where=df['bad_tracking_flag'], color='gray', alpha=0.3, label='Tracker Failure Zone')
            axs[1].set_ylabel('Intensity (0-5)')
            axs[1].set_title('2. AFTER: Cleaned Signals (Tracking Failures Neutralized)')
            axs[1].grid(True, alpha=0.3)
            axs[1].legend(loc='upper right')
            for gt_start, gt_end in ground_truth: axs[1].axvspan(gt_start, gt_end, color='lime', alpha=0.2)

        if 'AU12_r_smooth' in df.columns and 'AU06_r_smooth' in df.columns:
            axs[2].plot(timestamps, df['AU12_r_smooth'], color='darkblue', linewidth=2, label='Smoothed AU12_r')
            axs[2].plot(timestamps, df['AU06_r_smooth'], color='darkred', linewidth=2, label='Smoothed AU06_r')
            axs[2].set_ylabel('Intensity (0-5)')
            axs[2].set_title(f'3. Smoothed Individual Signals (Window={self.window_size})')
            axs[2].grid(True, alpha=0.3)
            axs[2].legend(loc='upper right')
            for gt_start, gt_end in ground_truth: axs[2].axvspan(gt_start, gt_end, color='lime', alpha=0.2)

        if 'Raw_Multiplied' in df.columns:
            axs[3].plot(timestamps, df['Raw_Multiplied'], color='magenta', alpha=0.6, label='Raw AU12 × Raw AU06')
            axs[3].set_ylabel('Intensity (0-25)')
            axs[3].set_title('4. Multiplied Raw Signals')
            axs[3].grid(True, alpha=0.3)
            axs[3].legend(loc='upper right')

        if 'Multiply_Then_Smooth' in df.columns:
            axs[4].plot(timestamps, df['Multiply_Then_Smooth'], color='olive', linewidth=2, label='Smoothed (Raw_Multiplied)')
            axs[4].set_ylabel('Intensity (0-25)')
            axs[4].set_title('5. Method 1: MULTIPLY First, THEN Smooth')
            axs[4].grid(True, alpha=0.3)
            axs[4].legend(loc='upper right')

        if 'Smooth_Then_Multiply' in df.columns:
            axs[5].plot(timestamps, df['Smooth_Then_Multiply'], color='purple', linewidth=2, label='AU12_smooth × AU06_smooth')
            axs[5].axhline(y=self.peak_thresh, color='black', linestyle='--', alpha=0.5, label=f'Peak Threshold ({self.peak_thresh})')
            axs[5].axhline(y=self.base_thresh, color='gray', linestyle='--', alpha=0.5, label=f'Base Threshold ({self.base_thresh})')
            for s, e in smart_intervals:
                axs[5].axvspan(s, e, color='purple', alpha=0.15)
                axs[5].axvline(x=s, color='green', linestyle=':', alpha=0.7) 
                axs[5].axvline(x=e, color='red', linestyle=':', alpha=0.7)   
            axs[5].set_ylabel('Intensity (0-25)')
            axs[5].set_title('6. Method 2: SMOOTH First, THEN Multiply (with Smart Interval Bounds)')
            axs[5].grid(True, alpha=0.3)
            axs[5].legend(loc='upper right')

        for _, (start, end) in enumerate(ground_truth): axs[6].hlines(y=3, xmin=start, xmax=end, color='limegreen', linewidth=15)
        for _, (start, end) in enumerate(smart_intervals): axs[6].hlines(y=2, xmin=start, xmax=end, color='darkviolet', linewidth=15)
        for _, (start, end) in enumerate(simple_extracted): axs[6].hlines(y=1, xmin=start, xmax=end, color='royalblue', linewidth=15)
        for _, (start, end) in enumerate(combo_extracted): axs[6].hlines(y=0, xmin=start, xmax=end, color='darkorange', linewidth=15)

        axs[6].set_title('7. Event Detection Timeline')
        axs[6].set_xlabel('Timestamp (seconds)', fontsize=12)
        axs[6].set_yticks([0, 1, 2, 3])
        axs[6].set_yticklabels(['Legacy Combo', 'Legacy Simple', 'Smart Peak', 'Ground Truth'])
        axs[6].set_ylim(-0.8, 3.8)
        axs[6].grid(axis='x', linestyle='--', alpha=0.7)

        real_start = timestamps.min()
        real_end = timestamps.max()
        
        plot_start = max(start_time, real_start) if start_time is not None else real_start
        plot_end = min(end_time, real_end) if end_time is not None else real_end
        
        plt.xlim(plot_start, plot_end)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot successfully saved to: {save_path}")
            
        plt.show()

    def render_video(self, input_video_path, output_video_path, detected_intervals, bad_intervals=None):
        """
        Reads a video, overlays a green light when a smile is detected, 
        and a gray light when tracking fails.
        """
        self.logger.info("Initializing video rendering engine...")
        bad_intervals = bad_intervals or []
        
        if not os.path.exists(input_video_path):
            self.logger.error(f"Input video not found at: {input_video_path}")
            raise FileNotFoundError(f"Input video not found at: {input_video_path}")
            
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            self.logger.error(f"Could not open video: {input_video_path}")
            raise IOError(f"Could not open video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if (fps > 0 and fps == fps) else self.fps
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        self.logger.info(f"Saving rendered video to: {output_video_path} at {fps} FPS")
        
        frame_count = 0
        
        with tqdm(total=total_frames, desc="Processing Video Frames", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_time_sec = frame_count / fps
                
                is_bad_tracking = any(start <= current_time_sec <= end for start, end in bad_intervals)
                is_smiling = any(start <= current_time_sec <= end for start, end in detected_intervals)
                
                center_x, center_y = width // 2, height // 2
                
                if is_bad_tracking:
                    cv2.circle(frame, (center_x, center_y), 30, (128, 128, 128), -1)
                    cv2.putText(frame, "TRACKING LOST", (center_x - 130, center_y + 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 3)
                elif is_smiling:
                    cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
                    cv2.putText(frame, "SMILE DETECTED", (center_x - 140, center_y + 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                out.write(frame)
                frame_count += 1
                pbar.update(1)

        cap.release()
        out.release()
        self.logger.info("Video rendering complete!")


if __name__ == "__main__":
    
    VIDEO_ID = "236"
    RAW_CSV_PATH = f'/home/amir/Ados_lab/videos/openface_marisa/{VIDEO_ID}/{VIDEO_ID}_Emotions_marissa.csv'
    SIMPLE_CSV_PATH = f'/home/amir/Ados_lab/out/{VIDEO_ID}/FeaturesAU/emo/SimpleSmiling/{VIDEO_ID}_emo_I.csv'
    COMBO_CSV_PATH = f'/home/amir/Ados_lab/out/{VIDEO_ID}/FeaturesAU/emo/ComboSmiling/{VIDEO_ID}_emo_I.csv'
    
    INPUT_VID_PATH = f'/home/amir/Ados_lab/videos/emotions/marisa_videos/{VIDEO_ID}_Emotions_marissa.mp4'
    
    # Setup Output Directories
    OUT_DIR = f'/home/amir/Ados_lab/out/{VIDEO_ID}/'
    OUTPUT_VID_PATH = os.path.join(OUT_DIR, f'{VIDEO_ID}_smile_rendered.avi')
    OUTPUT_PLOT_PATH = os.path.join(OUT_DIR, f'{VIDEO_ID}_analysis_plot.png')
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    try:
        from ground_truth import GROUND_TRUTH_INTERVALS_236
        gt_data = GROUND_TRUTH_INTERVALS_236
    except ImportError:
        gt_data = None
    
    analyzer = SmileAnalyzer(fps=30.0, window_size=15, peak_thresh=2.0, base_thresh=0.5)
    
    # Step 1: Perform the mathematical analysis
    processed_df, smart_intervals, simple_ext, combo_ext, bad_intervals = analyzer.analyze(
        raw_csv_path=RAW_CSV_PATH,
        simple_csv_path=SIMPLE_CSV_PATH,
        combo_csv_path=COMBO_CSV_PATH
    )

    # Step 2: Export Data for Future Downstream Use
    analyzer.export_data(output_dir=OUT_DIR, prefix=f"{VIDEO_ID}_")

    # Step 3: Visualizer
    analyzer.visualize(
        df=processed_df,
        smart_intervals=smart_intervals,
        simple_extracted=simple_ext,
        combo_extracted=combo_ext,
        ground_truth=None,  
        start_time=0,
        end_time=700.0,
        title=f"Smile Detection Analysis - {VIDEO_ID}",
        save_path=OUTPUT_PLOT_PATH  
    )

    # Step 4: Render Video
    if smart_intervals or bad_intervals:
        analyzer.render_video(
            input_video_path=INPUT_VID_PATH, 
            output_video_path=OUTPUT_VID_PATH, 
            detected_intervals=smart_intervals,
            bad_intervals=bad_intervals 
        )