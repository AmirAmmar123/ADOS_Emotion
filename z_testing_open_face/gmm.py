import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LogNorm
R_236 = '/home/amir/Ados_lab/out/236/csv/236_Emotions_marissa.csv'
R_249 = '/home/amir/Ados_lab/out/249/csv/249_Emotions_marissa.csv'

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
    if not os.path.exists(csv_path):
        print(f"File missing: {csv_path}. Please check the path.")
        return pd.DataFrame()
    return pd.read_csv(csv_path)

if __name__ == "__main__":

    data_236 = load_data(R_236)
    data_249 = load_data(R_249)

    # We only proceed if data loaded successfully
    if not data_236.empty and not data_249.empty:
        features = ['AU12_r', 'AU06_r']

        seqs_tp_236, labels_tp_236 = extract_sequences(data_236, clean_intervals(TP_236), 1, features)
        seqs_tn_236, labels_tn_236 = extract_sequences(data_236, clean_intervals(TN_236), 0, features)
        
        seqs_tp_249, labels_tp_249 = extract_sequences(data_249, clean_intervals(TP_249), 1, features)
        seqs_tn_249, labels_tn_249 = extract_sequences(data_249, clean_intervals(TN_249), 0, features)

        all_sequences = seqs_tp_236 + seqs_tn_236 + seqs_tp_249 + seqs_tn_249
        y = np.array(labels_tp_236 + labels_tn_236 + labels_tp_249 + labels_tn_249)

        # =========================================================
        # 3. SEQUENCE PADDING
        # =========================================================
        max_frames = max(len(seq) for seq in all_sequences)
        print(f"Padding sequences to {max_frames} frames...")

        X_padded = []
        for seq in all_sequences:
            frames_missing = max_frames - len(seq)
            if frames_missing > 0:
                padding = np.zeros((frames_missing, len(features)))
                padded_seq = np.vstack((seq, padding))
            else:
                padded_seq = seq
            X_padded.append(padded_seq.flatten())

        X_raw = np.array(X_padded)

        # =========================================================
        # 4. DIMENSIONALITY REDUCTION (PCA)
        # =========================================================
        print("Applying PCA to reduce dimensionality for the GMM...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_raw)
        
        variance_retained = np.sum(pca.explained_variance_ratio_) * 100
        print(f"PCA retained {variance_retained:.2f}% of the variance in just 2 dimensions.")

        # =========================================================
        # 5. SEMI-SUPERVISED GMM INITIALIZATION
        # =========================================================
        # To "teach" the GMM using our ground truth labels, we calculate 
        # the centroids of the 0 class and 1 class to initialize the Gaussians.
        centroid_0 = X_pca[y == 0].mean(axis=0)
        centroid_1 = X_pca[y == 1].mean(axis=0)
        means_init = np.array([centroid_0, centroid_1])

        print("Fitting the 2-Component Gaussian Mixture Model...")
        gmm = GaussianMixture(
            n_components=2, 
            covariance_type='full', 
            means_init=means_init, # Force GMM to anchor to our ground truth
            random_state=42
        )
        
        gmm.fit(X_pca)
        
        # Predict clusters (Note: Because we forced the means, cluster 0 aligns with label 0)
        y_pred = gmm.predict(X_pca)

        # =========================================================
        # 6. EVALUATION
        # =========================================================
        print("\n--- GMM Clustering vs Ground Truth ---")
        print(classification_report(y, y_pred, target_names=['Rest (0)', 'Smile (1)']))

        # =========================================================
        # 7. VISUALIZING THE DECISION BOUNDARIES
        # =========================================================
        print("Plotting the GMM Feature Space...")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a mesh grid to plot the probability contours
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        XX, YY = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = -gmm.score_samples(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)

        # Draw the GMM density contours
        contour = ax.contour(XX, YY, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10), cmap='viridis', alpha=0.5)

        # Scatter plot the actual data points colored by GROUND TRUTH
        scatter_rest = ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='red', label='True Rest (0)', edgecolors='k', s=60)
        scatter_smile = ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='lime', label='True Smile (1)', edgecolors='k', s=60)

        # Mark the Gaussians centers
        ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='X', s=200, color='black', label='Gaussian Centroids')

        ax.set_title('Gaussian Mixture Model - Smile Separation (PCA Space)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Principal Component 1 (Intensity / Duration Vector)')
        ax.set_ylabel('Principal Component 2 (Secondary Variance)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()