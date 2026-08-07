import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

def build_and_plot_survival_matrix(scores_dict, phase_title):
    """
    Converts a dictionary of fold scores into a DataFrame matrix, calculates
    the mean survival score, and plots a dual-dashboard (Heatmap + Leaderboard).
    Returns the sorted DataFrame.
    """
    # 1. Convert scores into a DataFrame
    scores_df = pd.DataFrame(scores_dict)
    
    # 2. Calculate the Average Survival Score and Sort
    scores_df['Mean_Survival_Score'] = scores_df.mean(axis=1)
    scores_df = scores_df.sort_values(by='Mean_Survival_Score', ascending=False)
    
    # 3. Dynamic figure height based on the number of features
    fig_height = max(6.0, len(scores_df) * 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(20, fig_height))
    fig.suptitle(f'{phase_title}: Feature Survival Score Analysis', fontsize=20, fontweight='bold', y=1.02)
    
    # Plot A: The Heatmap (Shows exact rank in every fold)
    heatmap_data = scores_df.drop(columns=['Mean_Survival_Score'])
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap='viridis', 
                ax=axes[0], linewidths=0.5, cbar_kws={'label': 'Survival Rank (Higher is Better)'})
    axes[0].set_title('Survival Rank per Fold', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Features', fontsize=14)
    axes[0].set_xlabel('K-Folds', fontsize=14)
    
    # Plot B: The Leaderboard (Mean Score)
    sns.barplot(x='Mean_Survival_Score', y=scores_df.index, data=scores_df, palette='magma', ax=axes[1])
    axes[1].set_title('Overall Leaderboard: Mean Survival Score', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Average Rank Across All Folds', fontsize=14)
    axes[1].set_ylabel('')
    
    # Add values to the bars
    for i in axes[1].containers:
        axes[1].bar_label(i, fmt='%.2f', padding=5, size=11, weight='bold')

    sns.despine()
    plt.tight_layout()
    plt.show()
    
    return scores_df