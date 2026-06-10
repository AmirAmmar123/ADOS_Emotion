import os
import argparse
import pandas as pd
from smileAnalyzer2 import SmileAnalyzer
from mylogger import Logger

class SmartRunner:
    def __init__(self, openface_dir_path, output_base_dir, video_source_dir):
        self.openface_dir_path = openface_dir_path
        self.output_base_dir = output_base_dir
        self.video_source_dir = video_source_dir
        self.dirs = os.listdir(self.openface_dir_path)
        self.logger = Logger('SmartRunner', 'logs/smart_runner.log')
    
    def _find_video_file(self, video_id, video_source_dir):
        """Locates the source video file using the video_id prefix."""
        if not os.path.exists(video_source_dir):
            return None
            
        for file in os.listdir(video_source_dir):
            if file.startswith(f"{video_id}_") and file.lower().endswith(('.mp4', '.avi', '.mov')):
                return os.path.join(video_source_dir, file)
        return None
    
    def run(self, plot_figures=False, save_figures=True, render_video=False):
        self.logger.debug(f"Arguments - Plot: {plot_figures}, Save: {save_figures}, Render: {render_video}")
        for dir_name in self.dirs:
            # Skip hidden files/folders (like .DS_Store)
            if dir_name.startswith('.'):
                continue 
                
            video_id = dir_name
            video_folder_path = os.path.join(self.openface_dir_path, video_id)
            
            if not os.path.isdir(video_folder_path):
                continue

            self.logger.info(f"--- Processing video: {video_id} ---")
            
            csv_files = [f for f in os.listdir(video_folder_path) if f.lower().endswith('.csv')]
            
            if not csv_files:
                self.logger.error(f"No CSV file found in {video_folder_path}. Skipping.")
                continue

            raw_csv_path = os.path.join(video_folder_path, csv_files[0])
            
            if not os.path.exists(raw_csv_path):
                self.logger.error(f"Raw CSV not found at {raw_csv_path}. Skipping.")
                continue

            # Set up output directory for this specific video
            save_dir = os.path.join(self.output_base_dir, video_id)
            os.makedirs(save_dir, exist_ok=True)
            plot_save_path = save_dir if save_figures else None

            # 1. Initialize and Load Data
            analyzer = SmileAnalyzer(fps=30.0, window_size=15, peak_thresh=2.0, base_thresh=0.5)
            
            analyzer.load_and_preprocess(
                raw_csv=raw_csv_path, 
                simple_csv=None, 
                combo_csv=None, 
                ground_truth=None
            )
            
            z_start = analyzer.df['timestamp'].min()
            z_end = analyzer.df['timestamp'].max()
            
            self.logger.info(f"Extracted Timeline for {video_id}: {z_start:.2f}s to {z_end:.2f}s")
            
            analyzer.run_baseline(s_time=z_start, e_time=z_end, plot=plot_figures, save_dir=plot_save_path)
            analyzer.run_adaptive_zscore(s_time=z_start, e_time=z_end, plot=plot_figures, save_dir=plot_save_path)
            analyzer.run_antagonist_penalty(lmbda=0.5, s_time=z_start, e_time=z_end, plot=plot_figures, save_dir=plot_save_path)
            analyzer.run_kinematics(s_time=z_start, e_time=z_end, plot=plot_figures, save_dir=plot_save_path)
            analyzer.run_hmm(s_time=z_start, e_time=z_end, plot=plot_figures, save_dir=plot_save_path)
            analyzer.plot_master_timeline(s_time=z_start, e_time=z_end, plot=plot_figures, save_dir=plot_save_path)

            analyzer.export_data(output_dir=save_dir, prefix=f"{video_id}_")
            
            if render_video:
                input_video = self._find_video_file(video_id, self.video_source_dir)
                if input_video:
                    output_video = os.path.join(save_dir, f"{video_id}_smile_rendered.avi")
                    analyzer.render_video(in_vid=input_video, out_vid=output_video)
                else:
                    self.logger.error(f"Source video for ID {video_id} not found in {self.video_source_dir}. Skipping render.")
            
            self.logger.info(f"Successfully finished processing {video_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Smile Analyzer on OpenFace batch results.")
    
    # Required Path Arguments
    parser.add_argument("--openface_dir", type=str, required=True, help="Path to directory containing OpenFace CSV folders")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to save output plots and data")
    parser.add_argument("--video_src_dir", type=str, required=True, help="Path to the original source videos (for rendering)")
    
    # Optional Behavior Flags
    parser.add_argument("--plot", action="store_true", help="Pop up matplotlib figures on the screen")
    parser.add_argument("--no_save", action="store_true", help="Disable saving figures to disk")
    parser.add_argument("--render", action="store_true", help="Enable video rendering with bounding boxes/alerts")
    
    args = parser.parse_args()

    # Initialize and trigger the runner
    runner = SmartRunner(
        openface_dir_path=args.openface_dir,
        output_base_dir=args.out_dir,
        video_source_dir=args.video_src_dir
    )
    
    runner.run(
        plot_figures=args.plot, 
        save_figures=not args.no_save, 
        render_video=args.render
    )