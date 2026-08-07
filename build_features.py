import os
import glob
import pandas as pd
import numpy as np # <--- ADD THIS IMPORT

# --- Configuration Paths ---
BASE_RESULTS_DIR = '/home/amir/Ados_lab/videos2/results'
LABELS_CSV_PATH = 'mapped_labels_output.csv'  
FINAL_OUTPUT_PATH = '/home/amir/Ados_lab/lab/final_feature_no_correlation_matrix.csv'  # Output path for the final feature matrix

# Features to extract exactly as they appear in the stats CSV
FEATURES_TO_EXTRACT = [
    "Interviewer Smile Rate (per min)",
    "Subject Smile Rate (per min)",
    "Simultaneous Smiles (Intersections) Rate (per min)",
    "Interviewer Smile Duration Mean (s)",
    "Interviewer Smile Duration Std (s)",
    "Subject Smile Duration Mean (s)",
    "Subject Smile Duration Std (s)",
    "Interviewer Smile Intensity Mean",
    "Interviewer Smile Intensity Std",
    "Subject Smile Intensity Mean",
    "Subject Smile Intensity Std",
    "Average Intersection Duration (s)",
    "Total Reactions (Subject reacted to Intv)",
    "Average Delay of Reaction (s)",
    "Std Dev of Delay of Reaction (s)",
    "% of Subject's smiles that were reactions",
    "% of Interviewer's smiles that got a reaction",
    "Simultaneous Smiles fraction (Marisa)",
    "Simultaneous Smiles fraction (Subject)",
    "Intervals Crossing Threshold (per frame)",
    # "Mean of Optimal Cosines (>TH)",
    # "Std of Optimal Cosines (>TH)",
    # "Mean of Optimal Delays (s) (>TH)",
    # "Std of Optimal Delays (s) (>TH)"
]

def extract_features_from_report(report_path):
    """Parses a single stats report CSV and extracts requested features as a dictionary."""
    extracted_data = {}
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if ',' in line]
            
        for line in lines:
            # --- FIX 1: Safely unpack the split line ---
            parts = [p.strip() for p in line.split(',', 1)]
            if len(parts) != 2:
                continue # Skip malformed lines instead of crashing
                
            feature_name, feature_value_str = parts
            
            if feature_name in FEATURES_TO_EXTRACT:
                # --- FIX 2: Explicitly handle missing data strings ---
                if feature_value_str.upper() in ['N/A', 'NAN', 'NONE', '']:
                    extracted_data[feature_name] = np.nan
                else:
                    try:
                        # Remove percentages and cast to float
                        extracted_data[feature_name] = float(feature_value_str.replace('%', ''))
                    except ValueError:
                        # --- FIX 3: Use np.nan instead of pd.NA ---
                        extracted_data[feature_name] = np.nan 
                    
    except Exception as e:
        print(f"Error reading {report_path}: {e}")
        
    return extracted_data

def build_matrix():
    # 1. Load demographics/labels
    if not os.path.exists(LABELS_CSV_PATH):
        print(f"Error: Labels file not found at {LABELS_CSV_PATH}")
        return
        
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    labels_df['Subject Number'] = labels_df['Subject Number'].astype(str) # Ensure string for reliable merging
    
    # 2. Locate all stat reports
    search_pattern = os.path.join(BASE_RESULTS_DIR, 'Results_Video_*', 'Smile_Synchrony_Stats_Video_*.csv')
    report_files = glob.glob(search_pattern)
    
    if not report_files:
        print(f"No stat reports found matching: {search_pattern}")
        return
        
    print(f"Found {len(report_files)} report files to process.")
    
    # 3. Extract data loop
    all_subjects_data = []
    for report_path in report_files:
        # Extract Subject ID from filename (e.g. 'Smile_Synchrony_Stats_Video_123.csv' -> '123')
        subject_id = os.path.basename(report_path).replace('Smile_Synchrony_Stats_Video_', '').replace('.csv', '')
        
        subject_features = extract_features_from_report(report_path)
        subject_features['Subject Number'] = subject_id
        
        all_subjects_data.append(subject_features)
        
    features_df = pd.DataFrame(all_subjects_data)
    
    # 4. Merge features with labels (Inner join ensures we only keep subjects that have both stats and labels)
    final_matrix_df = pd.merge(features_df, labels_df, on='Subject Number', how='inner')
    
    # 5. Reorder columns: Meta-data first, then features
    meta_cols = ['Subject Number', 'Label', 'Gender']
    feature_cols = [c for c in final_matrix_df.columns if c not in meta_cols]
    final_matrix_df = final_matrix_df[meta_cols + feature_cols]
    
    # 6. Save final dataset
    final_matrix_df.to_csv(FINAL_OUTPUT_PATH, index=False)
    print(f"\nSuccessfully created feature matrix with {len(final_matrix_df)} rows and {len(final_matrix_df.columns)} columns.")
    print(f"Saved final dataset to: {FINAL_OUTPUT_PATH}")

if __name__ == "__main__":
    build_matrix()