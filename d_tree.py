import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load the data
file_path = "final_feature_matrix.csv"
data = pd.read_csv(file_path)

# 2. Preprocess the data
# Map the categorical 'Gender' column to numeric values so the model can process it
data['Gender'] = data['Gender'].map({'F': 1, 'M': 0})

# Define features (X) and target (y)
# Drop 'Subject Number' (identifier) and 'Label' (target variable)
X = data.drop(['Subject Number', 'Label'], axis=1)
y = data['Label']

# 3. Normalize the data using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# 4. Initialize the classifier
# Added random_state for reproducibility
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# 5. Set up 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
confusion_mat = np.zeros((len(y.unique()), len(y.unique())))

# Run Cross Validation
for train_index, test_index in skf.split(X_normalized, y):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    confusion_mat += confusion_matrix(y_test, y_pred, labels=y.unique())

# Calculate average metrics
accuracy_avg = np.mean(accuracy_scores)
precision_avg = np.mean(precision_scores)
recall_avg = np.mean(recall_scores)
f1_avg = np.mean(f1_scores)

# Print average metrics
print("Accuracy: {:.2f}".format(accuracy_avg))
print("Precision: {:.2f}".format(precision_avg))
print("Recall: {:.2f}".format(recall_avg))
print("F1 Score: {:.2f}".format(f1_avg))
print("Confusion Matrix (Sum of all 5 folds):")
print(confusion_mat)

# 6. Feature Importance Plot
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
feat_labels = X.columns[sorted_indices[:4]]  # Select top 4 features

plt.figure(figsize=(8, 7))
plt.title('Top 4 Feature Importances')
plt.bar(range(4), importances[sorted_indices[:4]], align='center', color='skyblue')
plt.xticks(range(4), feat_labels, rotation=45, ha='right') # Rotated for better readability
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig("out/feature_importance.png", dpi=300)

# 7. Confusion Matrix Plot
labels = sorted(y.unique())
cm_df = pd.DataFrame(confusion_mat, index=labels, columns=labels)

plt.figure(figsize=(7, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Binary Classification\nAccuracy: {:.3f}'.format(accuracy_avg))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("out/confusion_matrix.png", dpi=300)