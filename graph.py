import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Labels: 0 = Stable, 1 = Unstable

# Based on confusion matrix
# Predicted Stable: 16480 (TP), 0 (FP)
# Predicted Unstable: 1480 (FN), 160 (TN)

y_true = [0] * 16480 + [0] * 1480 + [1] * 160  # True labels
y_pred = [0] * 16480 + [1] * 1480 + [1] * 160  # Predicted labels

# Compute scores
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Plotting
scores = [precision, recall, f1]
labels = ['Precision', 'Recall', 'F1 Score']
colors = ['#00BFC4', '#F8766D', '#7CAE00']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, scores, color=colors)
plt.ylim(0, 1.1)

# Annotate the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.03,
             f'{height:.2f}', ha='center', va='bottom', fontsize=12)

plt.title('Precision, Recall, and F1 Score', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
