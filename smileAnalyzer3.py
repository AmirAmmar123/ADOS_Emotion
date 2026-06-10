import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SmileSignalVisualizer:
    def __init__(self, fps=30.0, window_size=15):
        self.fps = fps
        self.window_size = window_size
        self.df = None
        
        self.smile_aus = ['AU12_r', 'AU06_r']
        self.contradictory_aus = ['AU04_r', 'AU09_r', 'AU10_r', 'AU14_r', 'AU15_r', 'AU20_r']
        self.all_aus = self.smile_aus + self.contradictory_aus

    def load_and_preprocess(self, raw_csv_path, penalty_sensitivity=0.5, noise_floor=0.2):
        """Loads data, smoothes, multiplies, and calculates the Master Smile Index using Weighted Exponential Decay."""
        print(f"Loading data from: {raw_csv_path}")
        
        if not os.path.exists(raw_csv_path):
            raise FileNotFoundError(f"Missing: {raw_csv_path}")
            
        df = pd.read_csv(raw_csv_path)
        df.columns = df.columns.str.strip()
        
        df['bad_tracking'] = (df.get('success', 1) == 0) | (df.get('confidence', 1) < 0.8)

        # 1. Clean and Smooth all AUs
        for c in self.all_aus:
            if c in df.columns:
                df.loc[df['bad_tracking'], c] = 0.0
                df[f'{c}_smooth'] = df[c].rolling(self.window_size, center=True, min_periods=1).mean()

        # 2. Duchenne Product: AU12 * AU06
        if 'AU12_r_smooth' in df.columns and 'AU06_r_smooth' in df.columns:
            df['Smile_Product'] = df['AU12_r_smooth'] * df['AU06_r_smooth']

        # 3. Smart Penalty: Anatomical Weighting + Exponential Decay
        weights = {
            'AU15_r_smooth': 2.0,  # Frown (Direct physical opposite - severe penalty)
            'AU14_r_smooth': 1.5,  # Contempt (Severe psychological opposite)
            'AU09_r_smooth': 1.0,  # Disgust (Strong penalty)
            'AU10_r_smooth': 1.0,  # Sneer (Strong penalty)
            'AU20_r_smooth': 0.8,  # Fear/Grimace (Moderate penalty)
            'AU04_r_smooth': 0.3   # Brow Lowerer (Weak penalty, can happen in intense joy)
        }
        
        df['Weighted_Contra'] = 0.0
        for au, weight in weights.items():
            if au in df.columns:
                # Apply noise floor: Ignore tiny OpenFace jitter below 0.2
                clean_au = df[au].where(df[au] > noise_floor, 0.0)
                df['Weighted_Contra'] += clean_au * weight

        # Apply Exponential Decay (Decays from 1.0 down to 0.0 as negative emotions rise)
        df['Contra_Penalty'] = np.exp(-penalty_sensitivity * df['Weighted_Contra'])
        
        # Smooth the penalty multiplier
        df['Contra_Penalty_Smooth'] = df['Contra_Penalty'].rolling(self.window_size, center=True, min_periods=1).mean()

        # 4. The Final Master Signal (Product * Exponential Penalty)
        if 'Smile_Product' in df.columns:
            df['Final_Smile_Index'] = df['Smile_Product'] * df['Contra_Penalty_Smooth']

        self.df = df.copy() 
        print("Data loaded, smoothed, multiplied, and exponentially penalized.")

    def visualize(self, s_time=None, e_time=None):
        """Generates a 4-panel plot with dynamic, proportional resolution scaling."""
        if self.df is None:
            print("Error: Load data first.")
            return
            
        ts = self.df['timestamp'].values
        
        fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        fig.suptitle("Master Smile Index vs. Base AUs and Contradictions", fontsize=16, fontweight='bold')

        # --- Dynamic Scaling Helper ---
        def apply_dynamic_scale(ax, columns):
            valid_cols = [c for c in columns if c in self.df.columns]
            if not valid_cols: return
            
            # Find the absolute maximum value among the selected columns
            max_val = self.df[valid_cols].max().max()
            
            if pd.notna(max_val) and max_val > 0.01:
                ax.set_ylim(0, max_val * 1.15) # Set Y-axis to 15% above the max value
            else:
                ax.set_ylim(0, 1.0) # Fallback if signal is completely flat

        # --- Dynamic Background Helper ---
        def add_tracking_fail_background(ax, show_label=False):
            label = 'Tracking Fail' if show_label else None
            ax.fill_between(ts, 0, 1, where=self.df['bad_tracking'], color='gray', alpha=0.3, 
                            transform=ax.get_xaxis_transform(), label=label)

        # --- Panel 1: The Base Smile Components ---
        p1_cols = ['AU12_r_smooth', 'AU06_r_smooth', 'Smile_Product']
        if 'Smile_Product' in self.df.columns:
            axs[0].plot(ts, self.df['AU12_r_smooth'], color='blue', alpha=0.5, label='Smoothed AU12 (Lips)')
            axs[0].plot(ts, self.df['AU06_r_smooth'], color='red', alpha=0.5, label='Smoothed AU06 (Cheeks)')
            axs[0].plot(ts, self.df['Smile_Product'], color='purple', linewidth=2, label='Smile Product (AU12 * AU06)')
            
        axs[0].set_title("1. Base Duchenne Components & Product")
        axs[0].set_ylabel("Intensity")
        apply_dynamic_scale(axs[0], p1_cols)
        add_tracking_fail_background(axs[0], show_label=True)
        axs[0].legend(loc='upper right')
        axs[0].grid(True, alpha=0.3)

        # --- Panel 2: THE FINAL MASTER SIGNAL ---
        p2_cols = ['Final_Smile_Index']
        if 'Final_Smile_Index' in self.df.columns:
            axs[1].plot(ts, self.df['Final_Smile_Index'], color='green', linewidth=3, label='Final Smile Index (Product * Exponential Penalty)')
            
        axs[1].set_title("2. Master Smile Index (Green Line = True Uncontradicted Smile)")
        axs[1].set_ylabel("Index Value")
        apply_dynamic_scale(axs[1], p2_cols) # This will scale perfectly relative to the penalty!
        add_tracking_fail_background(axs[1])
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)

        # --- Panel 3: Upper Face Contradictions ---
        p3_cols = ['AU04_r_smooth', 'AU09_r_smooth']
        colors_upper = {'AU04_r_smooth': 'darkred', 'AU09_r_smooth': 'darkorange'}
        labels_upper = {
            'AU04_r_smooth': 'Smoothed AU04 (Brow Lowerer)', 
            'AU09_r_smooth': 'Smoothed AU09 (Nose Wrinkler)'
        }
        for au in p3_cols:
            if au in self.df.columns:
                axs[2].plot(ts, self.df[au], color=colors_upper[au], linewidth=1.5, label=labels_upper[au])
        
        axs[2].set_title("3. Upper Face Contradictions (Anger / Disgust)")
        axs[2].set_ylabel("Intensity")
        apply_dynamic_scale(axs[2], p3_cols)
        add_tracking_fail_background(axs[2])
        axs[2].legend(loc='upper right')
        axs[2].grid(True, alpha=0.3)

        # --- Panel 4: Lower Face Contradictions ---
        p4_cols = ['AU10_r_smooth', 'AU14_r_smooth', 'AU15_r_smooth', 'AU20_r_smooth']
        colors_lower = {'AU10_r_smooth': 'purple', 'AU14_r_smooth': 'brown', 'AU15_r_smooth': 'black', 'AU20_r_smooth': 'olive'}
        labels_lower = {
            'AU10_r_smooth': 'Smoothed AU10 (Sneer)', 
            'AU14_r_smooth': 'Smoothed AU14 (Dimpler)',
            'AU15_r_smooth': 'Smoothed AU15 (Frown)', 
            'AU20_r_smooth': 'Smoothed AU20 (Lip Stretcher)'
        }
        for au in p4_cols:
            if au in self.df.columns:
                axs[3].plot(ts, self.df[au], color=colors_lower[au], linewidth=1.5, label=labels_lower[au])
                
        axs[3].set_title("4. Lower Face Contradictions (Sneer / Frown / Grimace)")
        axs[3].set_ylabel("Intensity")
        axs[3].set_xlabel("Time (seconds)")
        apply_dynamic_scale(axs[3], p4_cols)
        add_tracking_fail_background(axs[3])
        axs[3].legend(loc='upper right')
        axs[3].grid(True, alpha=0.3)

        # Apply Time Limits
        plt.xlim(max(s_time or ts.min(), ts.min()), min(e_time or ts.max(), ts.max()))
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    
    # Configure your paths
    VIDEO_ID = "251"
    BASE_DIR = '/home/amir/Ados_lab'
    RAW_CSV = os.path.join(BASE_DIR, f'videos/openface_marisa/{VIDEO_ID}/{VIDEO_ID}_Emotions_marissa.csv')
    
    visualizer = SmileSignalVisualizer(fps=30.0, window_size=15)
    
    # Load Data: k=0.5 for sensitivity, noise_floor=0.2 to ignore jitter
    visualizer.load_and_preprocess(raw_csv_path=RAW_CSV, penalty_sensitivity=0.5, noise_floor=0.2)
    
    # Plot Data
    visualizer.visualize(s_time=0, e_time=405.0)