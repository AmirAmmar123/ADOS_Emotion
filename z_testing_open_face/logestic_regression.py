import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


R_236 = '/home/amir/open_face/videos/openface_marisa/236/236_Emotions_marissa.csv'
R_249 = '/home/amir/open_face/videos/openface_marisa/249/249_Emotions_marissa.csv'

# =========================================================
# 1. GROUND TRUTH INTERVALS
# =========================================================
TP_236 = [(30.0, 31.0), (32.0, 33.0), (42.0, 44.0), (55.0, 59.0), 
          (86.0, 87.0), (91.0, 92.0), (98.0, 99.0), (282.0, 284.0), 
          (323.0, 325.0), (399.0, 400.0)]
TN_236 = [(7, 10), (14, 16), (17, 19), ("2:03", "2:05"), ("2:08", "2:10"), 
          ("2:12", "2:15"), ("3:05", "3:08"), ("3:10", "3:12"), ("5:07", "5:10")]

TP_249 = [(1.0, 9.0), (17.0, 18.0), (36.0, 39.0), (66.0, 69.0), (99.0, 100.0), 
          (106.0, 116.0), (137.0, 139.0), (142.0, 144.0), (170.0, 172.0), 
          (174.0, 177.0), (179.0, 180.0), (192.0, 197.0), (207.0, 211.0), 
          (242.0, 245.0), (264.0, 266.0), (311.0, 312.0), (357.0, 359.0), 
          (362.0, 365.0), (368.0, 369.0), (504.0, 506.0), (603.0, 605.0)]
TN_249 = [(10, 13), (14, 16), ("1:10", "1:13"), ("1:20", "1:23"), ("1:30", "1:33"), 
          ("2:25", "2:28"), ("3:22", "3:25"), ("4:05", "4:08"), ("4:10", "4:13"), 
          ("4:45", "4:48"), ("4:50", "4:53"), ("4:55", "4:58"), ("5:25", "5:28"), 
          ("5:31", "5:34"), ("6:25", "6:28"), ("8:03", "8:06")]

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def parse_time(t):
    if isinstance(t, str) and ':' in t:
        mins, secs = t.split(':')
        return float(mins) * 60 + float(secs)
    return float(t)

def clean_intervals(intervals):
    return [(parse_time(start), parse_time(end)) for start, end in intervals]

def extract_sequences(df, intervals, label, features):
    sequences = []
    labels = []
    for start, end in intervals:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        seq = df.loc[mask, features].values
        
        if len(seq) > 0: 
            sequences.append(seq)
            labels.append(label)
            
    return sequences, labels

def load_data(csv_path):
    return pd.read_csv(csv_path)

if __name__ == "__main__":

    data_236 = load_data(R_236)
    data_249 = load_data(R_249)

    features = ['AU12_r', 'AU06_r']

    seqs_tp_236, labels_tp_236 = extract_sequences(data_236, clean_intervals(TP_236), 1, features)
    seqs_tn_236, labels_tn_236 = extract_sequences(data_236, clean_intervals(TN_236), 0, features)
    
    seqs_tp_249, labels_tp_249 = extract_sequences(data_249, clean_intervals(TP_249), 1, features)
    seqs_tn_249, labels_tn_249 = extract_sequences(data_249, clean_intervals(TN_249), 0, features)

    all_sequences = seqs_tp_236 + seqs_tn_236 + seqs_tp_249 + seqs_tn_249
    all_labels = labels_tp_236 + labels_tn_236 + labels_tp_249 + labels_tn_249

    # =========================================================
    # 3. SEQUENCE PADDING & FLATTENING
    # =========================================================
    max_frames = max(len(seq) for seq in all_sequences)
    print(f"The longest interval contains {max_frames} frames. Padding all others to match...")

    X_padded = []
    for seq in all_sequences:
        frames_missing = max_frames - len(seq)
        if frames_missing > 0:
            padding = np.zeros((frames_missing, len(features)))
            padded_seq = np.vstack((seq, padding))
        else:
            padded_seq = seq
        X_padded.append(padded_seq.flatten())

    X = np.array(X_padded)
    y = np.array(all_labels)

    print(f"Final dataset shape: {X.shape[0]} intervals (rows), {X.shape[1]} features per interval (cols).\n")

    # =========================================================
    # 4. CROSS-VALIDATION MODELING
    # =========================================================
    log_reg = LogisticRegression(
        penalty='l2',          # Explicitly state L2 regularization
        C=0.1,                 # Stronger regularization (default is 1.0)
        class_weight='balanced', 
        max_iter=1000,
        solver='lbfgs'         # lbfgs is highly optimized for L2
    )
    

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', log_reg)
    ])
    
    # Setup Stratified 5-Fold CV (ensures each fold has a good mix of 1s and 0s)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We ask cross_validate to calculate multiple metrics for us across the 5 folds
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)

    # =========================================================
    # 5. EVALUATION & RESULTS
    # =========================================================
    print("--- 5-Fold Cross-Validation Results ---")
    print(f"Mean Accuracy:  {cv_results['test_accuracy'].mean():.4f} (std: {cv_results['test_accuracy'].std():.4f})")
    print(f"Mean Precision: {cv_results['test_precision'].mean():.4f} (std: {cv_results['test_precision'].std():.4f})")
    print(f"Mean Recall:    {cv_results['test_recall'].mean():.4f} (std: {cv_results['test_recall'].std():.4f})")
    print(f"Mean F1-Score:  {cv_results['test_f1'].mean():.4f} (std: {cv_results['test_f1'].std():.4f})\n")

    # Generate Cross-Validated Predictions for the Confusion Matrix
    # This trains 5 separate models, and makes predictions on the held-out folds, 
    # then stitches them all together for one big matrix.
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    
    print("--- Overall Classification Report (Aggregated CV) ---")
    print(classification_report(y, y_pred_cv, target_names=['No Smile (0)', 'Smile (1)']))

    # Plot the Cross-Validated Confusion Matrix
    cm = confusion_matrix(y, y_pred_cv)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Smile', 'Smile'])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Logistic Regression - 5-Fold CV Confusion Matrix')
    plt.show()