import os
import re
import cv2 
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mylogger import Logger

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: 'hmmlearn' not found. Run 'pip install hmmlearn' to use Method 4.")


class SmileAnalyzer:
    """
    Modular pipeline for detecting, visualizing, and rendering genuine Duchenne smiles 
    using Baseline, Z-Score, Kinematic, Antagonist, and Hidden Markov Models.
    """
    
    def __init__(self, fps=30.0, window_size=30, peak_thresh=2.0, base_thresh=0.5):
        self.fps = fps
        self.window_size = window_size
        self.peak_thresh = peak_thresh
        self.base_thresh = base_thresh
        
        self.df = None
        self.bad_intervals = []
        self.ground_truth = []
        self.legacy_intervals = {"simple": [], "combo": []}
        self.intervals = {"baseline": [], "zscore": [], "penalty": [], "kinematic": [], "hmm": []}
        
        os.makedirs('logs', exist_ok=True)
        self.logger = Logger('SmileAnalyzer', 'logs/smile_analysis.log')

    def _extract_csv_intervals(self, path, col):
        """Extracts frame intervals from CSV using regex and converts to seconds."""
        if not path or not os.path.exists(path): return []
        try:
            ext = pd.read_csv(path)[col].astype(str).str.extract(r'int64\((\d+)\).*?int64\((\d+)\)').dropna().astype(int)
            return [(r[0]/self.fps, r[1]/self.fps) for _, r in ext.iterrows()]
        except Exception as e:
            self.logger.error(f"Failed to read {path}: {e}")
            return []

    def _mask_to_intervals(self, mask, ts):
        """Converts a boolean array into a list of (start_time, end_time) tuples."""
        if not mask.any(): return []
        idx = np.where(np.diff(np.concatenate(([0], mask.astype(int), [0]))))[0]
        return [(ts[idx[i]], ts[idx[i+1]-1]) for i in range(0, len(idx), 2)]

    def _find_smart_intervals(self, ts, sig, peak_t, base_t):
        """Region-growing algorithm to expand peaks into full intervals."""
        grad = np.gradient(sig)
        peaks = np.where((grad[:-1] > 0) & (grad[1:] <= 0) & (sig[1:] >= peak_t))[0] + 1
        intervals, used = [], np.zeros(len(sig), dtype=bool)
        
        for p in peaks:
            if used[p]: continue
            s, e = p, p
            while s > 0 and sig[s] > base_t: s -= 1
            while e < len(sig) - 1 and sig[e] > base_t: e += 1
            intervals.append((ts[s], ts[e]))
            used[s:e+1] = True
            
        return intervals

    def load_and_preprocess(self, raw_csv, simple_csv=None, combo_csv=None, ground_truth=None):
        """Loads data, nullifies tracking failures, and calculates the base multiplier signal."""
        self.logger.info(f"Loading data from: {raw_csv}")
        self.ground_truth = ground_truth or []
        self.legacy_intervals["simple"] = self._extract_csv_intervals(simple_csv, 'peak_lp_simple')
        self.legacy_intervals["combo"] = self._extract_csv_intervals(combo_csv, 'peak_lp_combo')
        
        if not os.path.exists(raw_csv): raise FileNotFoundError(f"Missing: {raw_csv}")
        df = pd.read_csv(raw_csv)
        df.columns, ts = df.columns.str.strip(), df['timestamp'].values

        df['bad_tracking'] = (df.get('success', 1) == 0) | (df.get('confidence', 1) < 0.8)
        self.bad_intervals = self._mask_to_intervals(df['bad_tracking'].values, ts)

        cols = [c for c in ['AU12_r', 'AU06_r', 'AU15_r', 'AU20_r'] if c in df]
        for c in cols:
            df[f'{c}_unfiltered'] = df[c].copy()
            df.loc[df['bad_tracking'], c] = 0.0
            df[f'{c}_smooth'] = df[c].rolling(self.window_size, center=True, min_periods=1).mean()

        if {'AU12_r_smooth', 'AU06_r_smooth'}.issubset(df.columns):
            df['Base_Signal'] = df['AU12_r_smooth'] * df['AU06_r_smooth']

        self.df = df
        self.logger.info("Preprocessing complete.")

    def _handle_plot(self, fig, ts, s_time, e_time, save_dir, filename, plot, save):
        """Applies limits, conditionally saves, displays, and clears the figure."""
        if not plot and not save: 
            plt.close(fig)
            return
            
        plt.xlim(max(s_time or ts.min(), ts.min()), min(e_time or ts.max(), ts.max()))
        plt.tight_layout()
        if save and save_dir:
            fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved figure: {filename}")
        if plot: plt.show()
        plt.close(fig)

    def _plot_timeline_helper(self, ax, intervals, label):
        """Draws horizontal interval bars for timelines."""
        for s, e in self.ground_truth: ax.hlines(1, s, e, colors='limegreen', lw=15)
        for s, e in intervals: ax.hlines(0, s, e, colors='darkviolet', lw=15)
        ax.set_yticks([0, 1]); ax.set_yticklabels([label, 'Ground Truth']); ax.set_ylim(-0.8, 1.8)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    def run_baseline(self, s_time=None, e_time=None, plot=True, save_dir=None):
        """Executes the standard threshold method and generates the original 7-panel visualization."""
        self.logger.info("Running Baseline Method...")
        ts, sig = self.df['timestamp'].values, self.df['Base_Signal'].values
        self.intervals['baseline'] = self._find_smart_intervals(ts, sig, self.peak_thresh, self.base_thresh)
        
        if not plot and not save_dir: return

        fig, axs = plt.subplots(7, 1, figsize=(14, 21), sharex=True, gridspec_kw={'height_ratios': [1.5]*6 + [1]})
        fig.suptitle("Original Baseline Analysis", fontsize=16, fontweight='bold', y=0.98)

        axs[0].plot(ts, self.df.get('AU12_r_unfiltered', []), 'b', alpha=0.5, label='Raw AU12')
        axs[0].plot(ts, self.df.get('AU06_r_unfiltered', []), 'r', alpha=0.5, label='Raw AU06')
        axs[1].plot(ts, self.df.get('AU12_r', []), 'b', label='Clean AU12')
        axs[1].plot(ts, self.df.get('AU06_r', []), 'r', label='Clean AU06')
        axs[1].fill_between(ts, 0, 5, where=self.df['bad_tracking'], color='gray', alpha=0.3, label='Tracking Fail')
        axs[2].plot(ts, self.df.get('AU12_r_smooth', []), 'darkblue', label='Smooth AU12')
        axs[2].plot(ts, self.df.get('AU06_r_smooth', []), 'darkred', label='Smooth AU06')
        axs[3].plot(ts, self.df.get('AU12_r', []) * self.df.get('AU06_r', []), 'm', alpha=0.6, label='Raw Mult')
        axs[4].plot(ts, (self.df.get('AU12_r', []) * self.df.get('AU06_r', [])).rolling(self.window_size, center=True, min_periods=1).mean(), 'olive', label='Smooth(Mult)')
        
        axs[5].plot(ts, sig, 'purple', label='Base Signal')
        axs[5].axhline(self.peak_thresh, color='k', linestyle='--'); axs[5].axhline(self.base_thresh, color='gray', linestyle='--')
        for s, e in self.intervals['baseline']: axs[5].axvspan(s, e, color='purple', alpha=0.15)

        for ax in axs[:6]: ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        self._plot_timeline_helper(axs[6], self.intervals['baseline'], 'Baseline Peak')
        self._handle_plot(fig, ts, s_time, e_time, save_dir, "m0_baseline.png", plot, bool(save_dir))

    def run_adaptive_zscore(self, s_time=None, e_time=None, plot=True, save_dir=None):
        """Dynamic thresholds based on valid signal mean and variance."""
        self.logger.info("Running Z-Score Method...")
        ts, sig = self.df['timestamp'].values, self.df['Base_Signal'].values
        valid = sig[~self.df['bad_tracking']]
        
        dyn_peak, dyn_base = valid.mean() + (2 * valid.std()), valid.mean() + (0.5 * valid.std())
        self.intervals['zscore'] = self._find_smart_intervals(ts, sig, dyn_peak, dyn_base)
        
        if not plot and not save_dir: return
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axs[0].plot(ts, sig, 'purple', label='Smile Signal'); axs[0].axhline(dyn_peak, color='r', ls='--'); axs[0].axhline(dyn_base, color='orange', ls='--')
        axs[0].set_title('Adaptive Z-Score Thresholding'); axs[0].legend(); axs[0].grid(alpha=0.3)
        self._plot_timeline_helper(axs[1], self.intervals['zscore'], "Z-Score")
        self._handle_plot(fig, ts, s_time, e_time, save_dir, "m1_zscore.png", plot, bool(save_dir))

    def run_antagonist_penalty(self, lmbda=0.5, s_time=None, e_time=None, plot=True, save_dir=None):
        """Reduces signal amplitude when frown muscles (AU15, AU20) are active."""
        self.logger.info("Running Antagonist Penalty...")
        ts, sig = self.df['timestamp'].values, self.df['Base_Signal'].values
        pen = lmbda * (self.df.get('AU15_r_smooth', 0) + self.df.get('AU20_r_smooth', 0))
        self.df['Penalized_Signal'] = np.clip(sig - pen, 0, None)
        
        self.intervals['penalty'] = self._find_smart_intervals(ts, self.df['Penalized_Signal'].values, self.peak_thresh, self.base_thresh)

        if not plot and not save_dir: return
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axs[0].plot(ts, sig, 'purple', label='Agonists'); axs[0].plot(ts, pen, 'r', label='Antagonists')
        axs[1].plot(ts, self.df['Penalized_Signal'], 'g', label='Penalized'); axs[1].axhline(self.peak_thresh, color='k', ls='--')
        for ax in axs[:2]: ax.legend(); ax.grid(alpha=0.3)
        self._plot_timeline_helper(axs[2], self.intervals['penalty'], "Penalty")
        self._handle_plot(fig, ts, s_time, e_time, save_dir, "m2_penalty.png", plot, bool(save_dir))

    def run_kinematics(self, s_time=None, e_time=None, plot=True, save_dir=None):
        """Filters intervals ensuring upward initial velocity."""
        self.logger.info("Running Kinematics Method...")
        ts, sig = self.df['timestamp'].values, self.df['Base_Signal'].values
        vel = np.gradient(sig)
        self.df['Velocity'] = vel
        
        peaks = np.where((vel[:-1] > 0) & (vel[1:] <= 0) & (sig[1:] >= self.peak_thresh))[0] + 1
        intervals, used = [], np.zeros(len(sig), dtype=bool)
        for p in peaks:
            if used[p]: continue
            s, e = p, p
            while s > 0 and sig[s] > self.base_thresh and vel[s] >= -0.05: s -= 1
            while e < len(sig)-1 and sig[e] > self.base_thresh: e += 1
            intervals.append((ts[s], ts[e])); used[s:e+1] = True
            
        self.intervals['kinematic'] = intervals

        if not plot and not save_dir: return
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axs[0].plot(ts, sig, 'purple'); axs[0].grid(alpha=0.3); axs[0].set_title("Amplitude")
        axs[1].plot(ts, vel, 'orange'); axs[1].axhline(0, color='k'); axs[1].grid(alpha=0.3); axs[1].set_title("Velocity")
        self._plot_timeline_helper(axs[2], self.intervals['kinematic'], "Kinematics")
        self._handle_plot(fig, ts, s_time, e_time, save_dir, "m3_kinematics.png", plot, bool(save_dir))

    def run_hmm(self, s_time=None, e_time=None, plot=True, save_dir=None):
        """Utilizes unsupervised 2-State Gaussian HMM for classification."""
        if not HMM_AVAILABLE: return self.logger.error("hmmlearn missing. Skipping HMM.")
        self.logger.info("Running HMM Method...")
        ts, sig = self.df['timestamp'].values, self.df['Base_Signal'].values
        valid_idx = ~self.df['bad_tracking'].values
        
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42).fit(sig[valid_idx].reshape(-1, 1))
        smile_mask = (model.predict(sig.reshape(-1, 1)) == np.argmax(model.means_[:, 0])) & valid_idx
        
        self.df['HMM_State'] = smile_mask
        self.intervals['hmm'] = self._mask_to_intervals(smile_mask, ts)

        if not plot and not save_dir: return
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axs[0].plot(ts, sig, 'gray', alpha=0.5, label='Rest'); axs[0].fill_between(ts, sig, 0, where=smile_mask, color='lime', alpha=0.6, label='Smile State')
        axs[0].legend(); axs[0].grid(alpha=0.3); axs[0].set_title('HMM States')
        self._plot_timeline_helper(axs[1], self.intervals['hmm'], "HMM")
        self._handle_plot(fig, ts, s_time, e_time, save_dir, "m4_hmm.png", plot, bool(save_dir))

    def plot_master_timeline(self, s_time=None, e_time=None, plot=True, save_dir=None):
        """Generates a stacked comparative Gantt chart of all methods."""
        if not plot and not save_dir: return
        self.logger.info("Generating Master Timeline...")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        methods = [
            ("Legacy Combo", self.legacy_intervals['combo'], "darkorange"),
            ("Legacy Simple", self.legacy_intervals['simple'], "royalblue"),
            ("Baseline", self.intervals['baseline'], "darkviolet"),
            ("Z-Score", self.intervals['zscore'], "teal"),
            ("Penalty", self.intervals['penalty'], "firebrick"),
            ("Kinematics", self.intervals['kinematic'], "dodgerblue"),
            ("HMM", self.intervals['hmm'], "mediumseagreen"),
            ("Ground Truth", self.ground_truth, "limegreen")
        ]

        for i, (name, ints, color) in enumerate(methods):
            for s, e in ints: ax.hlines(i, s, e, colors=color, lw=12)

        ax.set_yticks(range(len(methods))); ax.set_yticklabels([m[0] for m in methods])
        ax.set_ylim(-0.5, len(methods) - 0.5); ax.grid(axis='x', ls='--', alpha=0.7)
        ax.set_title("Master Timeline Comparison", fontweight='bold')
        self._handle_plot(fig, self.df['timestamp'].values, s_time, e_time, save_dir, "master_timeline.png", plot, bool(save_dir))

    def export_data(self, output_dir, prefix=""):
        """Exports processed signals to CSV and all interval sets to JSON."""
        if self.df is None: return self.logger.error("No data to export.")
        os.makedirs(output_dir, exist_ok=True)
        self.df.to_csv(os.path.join(output_dir, f"{prefix}processed_signals.csv"), index=False)
        with open(os.path.join(output_dir, f"{prefix}all_intervals.json"), 'w') as f:
            json.dump({"bad_tracking": self.bad_intervals, "legacy": self.legacy_intervals, **self.intervals}, f, indent=4)
        self.logger.info("Data exported successfully.")

    def render_video(self, in_vid, out_vid):
        """Renders stacked, color-coded indicators perfectly centered horizontally."""
        if not os.path.exists(in_vid): return self.logger.error(f"Missing video: {in_vid}")
            
        cap = cv2.VideoCapture(in_vid)
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        
        cfg = {"baseline": ("BASE", (211, 0, 148)), "zscore": ("Z-SCORE", (128, 128, 0)),
               "penalty": ("PENALTY", (34, 34, 178)), "kinematic": ("KINEM", (255, 144, 30)),
               "hmm": ("HMM", (113, 179, 60))}
        
        for i in tqdm(range(int(cap.get(7))), desc="Rendering", unit="frame"):
            ret, frame = cap.read()
            if not ret: break
                
            t, y_off = i / fps, 50
            
            # Check for tracking failure first
            if any(s <= t <= e for s, e in self.bad_intervals):
                text, color = "TRACKING LOST", (128, 128, 128)
                text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
                
                # Calculate X to perfectly center the circle (24px) + gap (10px) + text
                start_x = (w - (24 + 10 + text_w)) // 2 
                cv2.circle(frame, (start_x + 12, y_off), 12, color, -1)
                cv2.putText(frame, text, (start_x + 34, y_off + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                # Stack active methods dynamically in the top-middle
                for k, (name, color) in cfg.items():
                    if any(s <= t <= e for s, e in self.intervals.get(k, [])):
                        text = f"{name}: SMILE"
                        text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
                        
                        start_x = (w - (24 + 10 + text_w)) // 2 
                        cv2.circle(frame, (start_x + 12, y_off), 12, color, -1)
                        cv2.putText(frame, text, (start_x + 34, y_off + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        y_off += 40 # Move down for the next alert to prevent overlap
            
            # Keep timestamp in the top right corner
            cv2.putText(frame, f"{t:.2f}s", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)

        cap.release(); out.release()


if __name__ == "__main__":
    
    # Configuration
    VIDEO_ID = ['4','5','6','7']
    for i, vid in enumerate(VIDEO_ID):
        BASE_DIR = '/home/amir/Ados_lab/'
        OUT_DIR = os.path.join(BASE_DIR, 'out', vid)
        
        # Flags for precise execution control
        PLOT_FIGURES = False       # Display plots on screen
        SAVE_FIGURES = True        # Save plots to disk
        EXPORT_JSON_CSV = True     # Save interval/signal data
        RENDER_VIDEO = True        # Overlay analysis onto source video
        
        os.makedirs(OUT_DIR, exist_ok=True)
        try:
            from ground_truth import GROUND_TRUTH_INTERVALS_236 as gt_data
        except ImportError:
            gt_data = []

        analyzer = SmileAnalyzer(fps=30.0, window_size=15, peak_thresh=2.0, base_thresh=0.5)
        
        # 1. Initialization
        analyzer.load_and_preprocess(
            raw_csv=os.path.join(BASE_DIR, f'videos/openface_subjects/{vid}/{vid}_EMOTIONS_SUBJECT.csv'), 
            simple_csv=None, 
            combo_csv=None,
            ground_truth=None
        )
        
        # 2. Execution Suite
        Z_START, Z_END = 0, 405.0
        save_path = OUT_DIR if SAVE_FIGURES else None

        analyzer.run_baseline(s_time=Z_START, e_time=Z_END, plot=PLOT_FIGURES, save_dir=save_path)
        analyzer.run_adaptive_zscore(s_time=Z_START, e_time=Z_END, plot=PLOT_FIGURES, save_dir=save_path)
        analyzer.run_antagonist_penalty(lmbda=0.5, s_time=Z_START, e_time=Z_END, plot=PLOT_FIGURES, save_dir=save_path)
        analyzer.run_kinematics(s_time=Z_START, e_time=Z_END, plot=PLOT_FIGURES, save_dir=save_path)
        analyzer.run_hmm(s_time=Z_START, e_time=Z_END, plot=PLOT_FIGURES, save_dir=save_path)
        analyzer.plot_master_timeline(s_time=Z_START, e_time=Z_END, plot=PLOT_FIGURES, save_dir=save_path)

        # 3. Export & Render
        if EXPORT_JSON_CSV:
            analyzer.export_data(output_dir=OUT_DIR, prefix=f"{vid}_")
            
        if RENDER_VIDEO:
            analyzer.render_video(
                in_vid=os.path.join(BASE_DIR, f'/videos/emotions/subjects_videos/{vid}/{vid}_EMOTIONS_SUBJECT.mp4'),
                out_vid=os.path.join(BASE_DIR, f'videos/openface_subjects/{vid}/{vid}_smile_rendered.avi')
            )