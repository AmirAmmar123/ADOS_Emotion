import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_boxplot(df_all_classes,important_features,title):
  sns.set(style="ticks")
  fig, ax = plt.subplots(1, figsize=(10, 10))
  df_melted = pd.melt(df_all_classes, id_vars=["Class"], value_vars=important_features)
  # Plot the boxplot using seaborn
  p =sns.boxplot(x="variable", y="value", hue="Class", data=df_melted)
  labels = [label.get_text() for label in ax.get_xticklabels()]
  ax.set_xticklabels([label[::-1] if not label.isascii() else label for label in labels])
  plt.xticks(rotation=45, ha='right')
  plt.subplots_adjust(bottom=0.2)
  p.set_title(title)
  plt.show()

#parameters:
#1. df_all_classes - pandas dataframe with a column named ["Class"] with the classification (0/1/2..)
# 2. important_features- list of the important features to present in the plot
# 3. title - title of the plot


