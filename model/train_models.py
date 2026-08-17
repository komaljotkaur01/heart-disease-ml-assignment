"""
Heart Disease Classification - ML Assignment 2
Model Training Script
BITS WILP M.Tech AIML/DSE | Machine Learning

This script trains all 5 classification models on the Heart Disease dataset
and evaluates them using 6 metrics: Accuracy, AUC, Precision, Recall, F1, MCC.

Usage:
    python model/train_models.py

Requirements:
    pip install scikit-learn numpy pandas matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Generate / Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Heart Disease Classification - ML Assignment 2")
print("  BITS WILP M.Tech AIML/DSE | Machine Learning")
print("=" * 70)

np.random.seed(42)
n = 1000

age      = np.random.randint(29, 77, n)
sex      = np.random.randint(0, 2, n)
cp       = np.random.randint(0, 4, n)
trestbps = np.random.randint(94, 200, n)
chol     = np.random.randint(126, 564, n)
fbs      = np.random.randint(0, 2, n)
restecg  = np.random.randint(0, 3, n)
thalach  = np.random.randint(71, 202, n)
exang    = np.random.randint(0, 2, n)
oldpeak  = np.round(np.random.uniform(0, 6.2, n), 1)
slope    = np.random.randint(0, 3, n)
ca       = np.random.randint(0, 4, n)
thal     = np.random.randint(0, 3, n)

score = (
    0.03 * age - 0.2 * sex + 0.3 * cp + 0.01 * trestbps
    + 0.001 * chol + 0.1 * fbs - 0.001 * thalach
    + 0.4 * exang + 0.3 * oldpeak - 0.2 * slope
    + 0.3 * ca + 0.2 * thal + np.random.normal(0, 0.5, n)
)
target = (score > score.mean()).astype(int)

df = pd.DataFrame({
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
    'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
    'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
    'ca': ca, 'thal': thal, 'target': target
})

print(f"\n[INFO] Dataset shape     : {df.shape}")
print(f"[INFO] Features          : {df.shape[1] - 1}")
print(f"[INFO] Target distribution:\n{df['target'].value_counts().to_string()}")
print(f"[INFO] Missing values    : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != 'target']
X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\n[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Define Models
# ─────────────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
results = []
trained_models = {}

print("\n" + "=" * 70)
print("  MODEL TRAINING & EVALUATION")
print("=" * 70)

for name, model in models.items():
    print(f"\n[MODEL] {name}")
    print("-" * 50)

    # Train
    model.fit(X_train_s, y_train)
    trained_models[name] = model

    # Predict
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_test, y_pred)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  MCC       : {mcc:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    print(f"  CV Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Classification report
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])}")

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4)
    })

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Summary Table
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).set_index("Model")

print("\n" + "=" * 70)
print("  COMPARISON TABLE - ALL MODELS")
print("=" * 70)
print(results_df.to_string())

best_model = results_df['F1 Score'].idxmax()
print(f"\n[WINNER] Best Model (by F1 Score): {best_model}")
print(f"   Accuracy  : {results_df.loc[best_model, 'Accuracy']:.4f}")
print(f"   AUC       : {results_df.loc[best_model, 'AUC']:.4f}")
print(f"   F1 Score  : {results_df.loc[best_model, 'F1 Score']:.4f}")
print(f"   MCC       : {results_df.loc[best_model, 'MCC']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Visualizations
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Generating visualizations...")

# 6a. Metrics comparison bar chart
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(results_df.columns))
width = 0.15
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for i, (model_name, row) in enumerate(results_df.iterrows()):
    offset = (i - 2) * width
    ax.bar(x + offset, row.values, width, label=model_name,
           color=colors[i], alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(results_df.columns, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("All Models - Metrics Comparison", fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.4, label='0.8 threshold')
plt.tight_layout()
plt.savefig('model/metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[INFO] Saved: model/metrics_comparison.png")

# 6b. Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for idx, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                ax=axes[idx], linewidths=0.5)
    axes[idx].set_title(name, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")
axes[-1].set_visible(False)
plt.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("[INFO] Saved: model/confusion_matrices.png")

# 6c. Feature importance (Random Forest)
rf_model = trained_models["Random Forest"]
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
feat_imp.plot(kind='barh', ax=ax, color='#e74c3c', edgecolor='white')
ax.set_title("Random Forest - Feature Importance", fontsize=13, fontweight='bold')
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig('model/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("[INFO] Saved: model/feature_importance.png")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE!")
print("=" * 70)
print("\nFiles generated:")
print("  - model/metrics_comparison.png")
print("  - model/confusion_matrices.png")
print("  - model/feature_importance.png")
print("\nTo run the Streamlit app:")
print("  streamlit run app.py")
