import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =============================================================================
# --- 0. Configuration: Paths and Filenames ---
# =============================================================================
# Define where the data comes from
INPUT_FILE_PATH = "/home/amir/Ados_lab/lab/final_feature_no_correlation_matrix.csv"

# Define where the output images should be saved
OUTPUT_DIR = "/home/amir/Ados_lab/lab/results" 

# Define the specific filenames for the output images
FEATURE_IMPORTANCE_FILENAME = "rf_no_corr_feature_importance1.png"
CONFUSION_MATRIX_FILENAME = "rf_no_corr_confusion_matrix1.png"

# Ensure the output directory exists before trying to save files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# --- 1. Load Data ---
# =============================================================================
data = pd.read_csv(INPUT_FILE_PATH)

# --- 2. Data Preparation ---
# Drop metadata columns to isolate the behavioral features
meta_cols = ['Subject Number', 'Label', 'Gender']
X = data.drop(columns=meta_cols)
y = data['Label']

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# --- 3. Classifier Initialization ---
rf = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight='balanced', random_state=42)

# --- 4. 7-Fold Cross-Validation ---
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
confusion_mat = np.zeros((len(y.unique()), len(y.unique())))

for train_index, test_index in skf.split(X_normalized, y):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    # Use 'binary' or 'weighted' depending on preference; 'weighted' matches your original script
    precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    confusion_mat += confusion_matrix(y_test, y_pred)

# --- 5. Calculate & Print Average Metrics ---
accuracy_avg = np.mean(accuracy_scores)
precision_avg = np.mean(precision_scores)
recall_avg = np.mean(recall_scores)
f1_avg = np.mean(f1_scores)

print("Accuracy: {:.2f}".format(accuracy_avg))
print("Precision: {:.2f}".format(precision_avg))
print("Recall: {:.2f}".format(recall_avg))
print("F1 Score: {:.2f}".format(f1_avg))
print("Confusion Matrix (Sum of all 7 folds):")
print(confusion_mat)

# --- 6. Feature Importance ---
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

# Select top 7 features for a slightly wider view
top_n = 7
feat_labels = X.columns[sorted_indices[:top_n]]  

# Plot feature ranking
plt.figure(figsize=(10, 6))
plt.title(f'Top {top_n} Feature Importances')
plt.bar(range(top_n), importances[sorted_indices[:top_n]], align='center', color='skyblue')
plt.xticks(range(top_n), feat_labels, rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()

# Save the figure dynamically
feat_imp_path = os.path.join(OUTPUT_DIR, FEATURE_IMPORTANCE_FILENAME)
plt.savefig(feat_imp_path, dpi=300)
print(f"Saved feature importance plot to: {feat_imp_path}")
# plt.show()

# --- 7. Plot Confusion Matrix ---
labels = sorted(y.unique())
cm_df = pd.DataFrame(confusion_mat, index=labels, columns=labels)

plt.figure(figsize=(7, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('All Features\nAccuracy: {:.3f}'.format(accuracy_avg))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Save the figure dynamically
cm_path = os.path.join(OUTPUT_DIR, CONFUSION_MATRIX_FILENAME)
plt.savefig(cm_path, dpi=300)
print(f"Saved confusion matrix plot to: {cm_path}")
# plt.show()