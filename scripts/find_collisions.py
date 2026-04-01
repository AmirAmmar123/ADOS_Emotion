import pandas as pd
import re
import matplotlib.pyplot as plt

def parse_intervals(interval_str):
    """Parses string representation of intervals into a list of tuples."""
    pattern = r"np\.int64\((\d+)\),\s*np\.int64\((\d+)\)"
    matches = re.findall(pattern, str(interval_str))
    return [(int(start), int(end)) for start, end in matches]

def get_peaks_by_type(file_path):
    """Separates peaks into combosmile and simplesmile sets."""
    df = pd.read_csv(file_path)
    combo = []
    simple = []
    if 'combosmile_peaks' in df.columns:
        for cell in df['combosmile_peaks'].dropna():
            combo.extend(parse_intervals(cell))
    if 'simplesmile_peaks' in df.columns:
        for cell in df['simplesmile_peaks'].dropna():
            simple.extend(parse_intervals(cell))
    return combo, simple

def find_collisions_with_delay(file_a, file_b, delay):
    """Finds collisions between Examiner and Subject peak sets with a delay."""
    ex_combo, ex_simple = get_peaks_by_type(file_a)
    sub_combo, sub_simple = get_peaks_by_type(file_b)
    
    # Apply delay and separate into combo/simple types to match unpacking
    shifted_sub_combo = [(s - delay, e - delay) for s, e in sub_combo]
    shifted_sub_simple = [(s - delay, e - delay) for s, e in sub_simple]
    
    # Combine for collision detection
    examiner_intervals = ex_combo + ex_simple
    shifted_subject = shifted_sub_combo + shifted_sub_simple
    
    examiner_intervals.sort()
    shifted_subject.sort()
    
    collisions = []
    for start_a, end_a in examiner_intervals:
        for start_b, end_b in shifted_subject:
            if max(start_a, start_b) <= min(end_a, end_b):
                collisions.append(((start_a, end_a), (start_b, end_b)))
            elif start_b > end_a:
                break
                
    return ex_combo, ex_simple, sub_combo, sub_simple, shifted_sub_combo, shifted_sub_simple, collisions

def plot_intervals_as_bars(ex_combo, ex_simple, sub_combo, sub_simple, shifted_sub_combo, shifted_sub_simple, collisions):
    """Plots intervals with 'bold' combosmile and 'non-bold' simplesmile."""
    fig, ax = plt.subplots(figsize=(12, 7))

    def draw_bars(intervals, y_pos, color, label, bold=True):
        alpha = 1.0 if bold else 0.3
        edge = 'black' if bold else 'none'
        
        for start, end in intervals:
            ax.barh(y=y_pos, width=end - start, left=start, height=0.6, 
                    color=color, alpha=alpha, edgecolor=edge)
        ax.barh(y=y_pos, width=0, left=0, color=color, alpha=alpha, label=label)

    # Plotting tracks
    draw_bars(ex_combo, 3.2, 'blue', 'Examiner: Combo (Bold)', bold=True)
    draw_bars(ex_simple, 2.8, 'blue', 'Examiner: Simple (Not Bold)', bold=False)
    draw_bars(sub_combo, 2.2, 'green', 'Subject: Combo (Bold)', bold=True)
    draw_bars(sub_simple, 1.8, 'green', 'Subject: Simple (Not Bold)', bold=False)
    draw_bars(shifted_sub_combo, 1.2, 'orange', 'Subject (Del): Combo (Bold)', bold=True)
    draw_bars(shifted_sub_simple, 0.8, 'orange', 'Subject (Del): Simple (Not Bold)', bold=False)

    # Plot Collisions
    for (a, b) in collisions:
        overlap_start = max(a[0], b[0])
        overlap_end = min(a[1], b[1])
        ax.barh(y=0, width=overlap_end - overlap_start, left=overlap_start, height=0.8, color='red', alpha=0.8)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Collisions', 'Subject (Delayed)', 'Subject', 'Examiner'])
    ax.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig('interval_formatted_plot.png')
    print("Plot saved as 'interval_formatted_plot.png'")

# Usage
PATH1 = '/home/amir/open_face/temp/5_emo_mar/res1.csv'
PATH2 = '/home/amir/open_face/temp/6_emo_S/res2.csv'

delay = 90
ex_c, ex_s, sub_c, sub_s, shifted_c, shifted_s, overlaps = find_collisions_with_delay(PATH1, PATH2, delay)

plot_intervals_as_bars(ex_c, ex_s, sub_c, sub_s, shifted_c, shifted_s, overlaps)