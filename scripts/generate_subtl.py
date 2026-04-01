import pandas as pd
import math
import re
import ast
import argparse

def frames_to_srt_time(frame, fps=29.98):
    total_seconds = frame / fps
    hours = math.floor(total_seconds / 3600)
    minutes = math.floor((total_seconds % 3600) / 60)
    seconds = math.floor(total_seconds % 60)
    milliseconds = math.floor((total_seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def parse_peak_string(peak_str):
    if pd.isna(peak_str) or peak_str == "[]" or peak_str == "nan":
        return []
    if "int64" in str(peak_str):
        numbers = [int(num) for num in re.findall(r"int64\((\d+)\)", peak_str)]
        return [(numbers[i], numbers[i+1]) for i in range(0, len(numbers), 2)]
    else:
        try:
            return ast.literal_eval(str(peak_str))
        except (ValueError, SyntaxError):
            return []

def generate_dual_smile_srt(csv_path, subject, task, camera, output_filename, fps=29.98):
    df = pd.read_csv(csv_path)
    
    # Filter for specific video
    video_data = df[(df['subject'] == int(subject)) & 
                    (df['task'] == task) & 
                    (df['camera'] == camera)]
    
    if video_data.empty:
        print(f"No data found for {subject}_{task}_{camera}")
        return

    # Extract both types of peaks
    # Adjust 'simple_peaks' column name if it is named differently in your res.csv
    combo_str = str(video_data.iloc[0].get('combosmile_peaks', '[]'))
    simple_str = str(video_data.iloc[0].get('simplesmile_peaks', '[]')) 

    combo_intervals = parse_peak_string(combo_str)
    simple_intervals = parse_peak_string(simple_str)

    # Combine and sort intervals by start frame to keep SRT sequential
    all_events = []
    for start, end in combo_intervals:
        all_events.append((start, end, "Combo Smile"))
    for start, end in simple_intervals:
        all_events.append((start, end, "Simple (AU12) Smile"))
    
    # Sort events by start frame
    all_events.sort(key=lambda x: x[0])

    if not all_events:
        print("No smiles detected.")
        return

    with open(output_filename, 'w') as f:
        for idx, (start_f, end_f, label) in enumerate(all_events, start=1):
            start_t = frames_to_srt_time(start_f, fps)
            end_t = frames_to_srt_time(end_f, fps)
            
            f.write(f"{idx}\n")
            f.write(f"{start_t} --> {end_t}\n")
            f.write(f"[{label}] Frames: {start_f}-{end_f}\n\n")
            
    print(f"Generated SRT with {len(all_events)} total events.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SRT subtitles from smile detection data")
    parser.add_argument("id", type=int, help="Subject ID")
    parser.add_argument("task", type=str, help="Task name (e.g., 'emo')")
    parser.add_argument("camera", type=str, help="Camera identifier (e.g., 'I')")
    parser.add_argument("--base-path", type=str, default="/home/amir/open_face/out", 
                        help="Base path for output directory (default: /home/amir/open_face/out)")
    
    args = parser.parse_args()
    
    ID = args.id
    TEMP = f"{args.base_path}/{ID}"
    CSV_FILE = f"{TEMP}/res.csv"
    output_file = f"{TEMP}/{ID}_{args.task}_{args.camera}_smiles.srt"
    
    generate_dual_smile_srt(CSV_FILE, ID, args.task, args.camera, output_file)