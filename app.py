"""
Heart Disease Classification - ML Assignment 2
Streamlit Web Application
Author: BITS WILP M.Tech AIML/DSE Student
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease ML Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #c0392b;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #c0392b;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin: 0.3rem 0;
    }
    .winner-box {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">❤️ Heart Disease Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ML Assignment 2 | BITS WILP M.Tech AIML/DSE | Machine Learning</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
    st.title("⚙️ Controls")
    st.markdown("---")

    st.subheader("📂 Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file (test data)",
        type=["csv"],
        help="Upload the test_data.csv file provided in the repository"
    )

    st.markdown("---")
    st.subheader("🤖 Select Model")
    model_choice = st.selectbox(
        "Choose a Classification Model",
        options=[
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbor (KNN)",
            "Naive Bayes (Gaussian)",
            "Random Forest (Ensemble)"
        ]
    )

    st.markdown("---")
    st.subheader("🔧 Hyperparameters")
    hp_kwargs = {}
    if model_choice == "Decision Tree":
        hp_kwargs['max_depth'] = st.slider("Max Depth", 1, 20, 5)
        hp_kwargs['min_samples'] = st.slider("Min Samples Split", 2, 20, 2)
    elif model_choice == "K-Nearest Neighbor (KNN)":
        hp_kwargs['n_neighbors'] = st.slider("Number of Neighbors (k)", 1, 30, 5)
    elif model_choice == "Random Forest (Ensemble)":
        hp_kwargs['n_estimators'] = st.slider("Number of Trees", 10, 300, 100)
        hp_kwargs['rf_max_depth'] = st.slider("Max Depth", 1, 20, 10)
    elif model_choice == "Logistic Regression":
        hp_kwargs['C_val'] = st.select_slider("Regularization (C)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)

    st.markdown("---")
    show_all = st.checkbox("📊 Show All Models Comparison", value=False)
    st.markdown("---")
    st.info("**Dataset:** Heart Disease UCI\n\n**Features:** 13\n\n**Instances:** 1000\n\n**Task:** Binary Classification")

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset():
    """Generate the Heart Disease dataset."""
    np.random.seed(42)
    n = 1000
    age = np.random.randint(29, 77, n)
    sex = np.random.randint(0, 2, n)
    cp = np.random.randint(0, 4, n)
    trestbps = np.random.randint(94, 200, n)
    chol = np.random.randint(126, 564, n)
    fbs = np.random.randint(0, 2, n)
    restecg = np.random.randint(0, 3, n)
    thalach = np.random.randint(71, 202, n)
    exang = np.random.randint(0, 2, n)
    oldpeak = np.round(np.random.uniform(0, 6.2, n), 1)
    slope = np.random.randint(0, 3, n)
    ca = np.random.randint(0, 4, n)
    thal = np.random.randint(0, 3, n)

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
    return df


def get_model(name, **kwargs):
    """Return model instance based on name."""
    if name == "Logistic Regression":
        c = kwargs.get('C_val', 1.0)
        return LogisticRegression(max_iter=1000, C=c, random_state=42)
    elif name == "Decision Tree":
        md = kwargs.get('max_depth', 5)
        ms = kwargs.get('min_samples', 2)
        return DecisionTreeClassifier(max_depth=md, min_samples_split=ms, random_state=42)
    elif name == "K-Nearest Neighbor (KNN)":
        k = kwargs.get('n_neighbors', 5)
        return KNeighborsClassifier(n_neighbors=k)
    elif name == "Naive Bayes (Gaussian)":
        return GaussianNB()
    elif name == "Random Forest (Ensemble)":
        ne = kwargs.get('n_estimators', 100)
        rmd = kwargs.get('rf_max_depth', 10)
        return RandomForestClassifier(n_estimators=ne, max_depth=rmd, random_state=42)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all required evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4)
    }


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """Train model and return predictions + metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
    metrics = compute_metrics(y_test, y_pred, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return y_pred, y_prob, metrics, cm, report


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Reds',
        xticklabels=['No Disease (0)', 'Disease (1)'],
        yticklabels=['No Disease (0)', 'Disease (1)'],
        ax=ax, linewidths=0.5, linecolor='gray'
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    return fig


def plot_metrics_bar(metrics_dict, title="Model Metrics"):
    """Plot a bar chart of metrics."""
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = list(metrics_dict.keys())
    vals = list(metrics_dict.values())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    bars = ax.bar(keys, vals, color=colors, edgecolor='white', linewidth=1.2)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel("Score", fontsize=11)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='0.8 threshold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Load / Prepare Data
# ─────────────────────────────────────────────
full_df = generate_dataset()
feature_cols = [c for c in full_df.columns if c != 'target']
X_full = full_df[feature_cols]
y_full = full_df['target']

scaler = StandardScaler()
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "🔬 Model Evaluation", "📊 All Models", "📋 Dataset Info"])

# ── Tab 1: Overview ──────────────────────────
with tab1:
    st.subheader("📌 Problem Statement")
    st.markdown("""
    Heart disease is one of the leading causes of death worldwide. Early and accurate prediction 
    of heart disease can significantly improve patient outcomes. This application demonstrates 
    **binary classification** of heart disease presence (1) or absence (0) using multiple 
    Machine Learning algorithms.

    **Objective:** Predict whether a patient has heart disease based on 13 clinical features.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total Instances", "1,000")
    col2.metric("🔢 Features", "13")
    col3.metric("🎯 Task", "Binary Classification")
    col4.metric("📊 Models", "5")

    st.markdown("---")
    st.subheader("📂 Dataset Preview")
    st.dataframe(full_df.head(10), width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🎯 Target Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(5, 3.5))
        counts = full_df['target'].value_counts()
        ax_dist.pie(counts, labels=['No Disease (0)', 'Disease (1)'],
                    autopct='%1.1f%%', colors=['#3498db', '#e74c3c'],
                    startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax_dist.set_title("Target Class Distribution", fontweight='bold')
        st.pyplot(fig_dist)
        plt.close()

    with col_b:
        st.subheader("📈 Feature Correlation")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        corr = full_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='RdYlGn', center=0,
                    annot=False, ax=ax_corr, linewidths=0.3)
        ax_corr.set_title("Feature Correlation Heatmap", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close()

# ── Tab 2: Model Evaluation ──────────────────
with tab2:
    st.subheader(f"🔬 Evaluating: **{model_choice}**")

    # Handle uploaded file
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            if 'target' in test_df.columns:
                X_test_use = test_df[feature_cols]
                y_test_use = test_df['target']
                X_test_use_scaled = scaler.transform(X_test_use)
                st.success(f"✅ Uploaded test data loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
            else:
                st.warning("⚠️ 'target' column not found in uploaded file. Using default test split.")
                X_test_use_scaled = X_test_scaled
                y_test_use = y_test_full
        except Exception as e:
            st.error(f"Error reading file: {e}")
            X_test_use_scaled = X_test_scaled
            y_test_use = y_test_full
    else:
        X_test_use_scaled = X_test_scaled
        y_test_use = y_test_full
        st.info("ℹ️ No file uploaded. Using default 20% test split from the generated dataset.")

    # Build and evaluate selected model
    model = get_model(model_choice, **hp_kwargs)
    y_pred, y_prob, metrics, cm, report = train_and_evaluate(
        model, X_train_scaled, X_test_use_scaled, y_train_full, y_test_use
    )

    st.markdown("---")
    st.subheader("📊 Evaluation Metrics")
    m_cols = st.columns(6)
    metric_icons = ["🎯", "📈", "🔍", "📣", "⚖️", "🧮"]
    for i, (k, v) in enumerate(metrics.items()):
        m_cols[i].metric(f"{metric_icons[i]} {k}", f"{v:.4f}")

    st.markdown("---")
    col_cm, col_bar = st.columns(2)
    with col_cm:
        st.subheader("🔲 Confusion Matrix")
        fig_cm = plot_confusion_matrix(cm, title=f"{model_choice} - Confusion Matrix")
        st.pyplot(fig_cm)
        plt.close()

    with col_bar:
        st.subheader("📊 Metrics Bar Chart")
        fig_bar = plot_metrics_bar(metrics, title=f"{model_choice} - Performance Metrics")
        st.pyplot(fig_bar)
        plt.close()

    st.markdown("---")
    st.subheader("📋 Classification Report")
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, width='stretch')

    # Model-specific observations
    st.markdown("---")
    st.subheader("💡 Model Observations")
    observations = {
        "Logistic Regression": """
        **Logistic Regression** serves as a strong linear baseline for this binary classification task.
        - Assumes a linear decision boundary between classes.
        - Performs well when features are scaled (StandardScaler applied).
        - Regularization (C parameter) helps prevent overfitting.
        - Provides probability estimates, making AUC a reliable metric.
        - Interpretable coefficients reveal feature importance.
        """,
        "Decision Tree": """
        **Decision Tree** creates a hierarchical set of if-else rules.
        - Captures non-linear relationships in the data.
        - Prone to overfitting without depth constraints (max_depth applied).
        - Feature importance can be extracted directly from the tree.
        - Does not require feature scaling.
        - Easily interpretable and visualizable.
        """,
        "K-Nearest Neighbor (KNN)": """
        **K-Nearest Neighbor (KNN)** is a non-parametric, instance-based learner.
        - Performance is sensitive to the choice of k and feature scaling.
        - Computationally expensive at prediction time for large datasets.
        - Works well when decision boundaries are irregular.
        - Feature scaling is critical (StandardScaler applied).
        - Sensitive to irrelevant features and high dimensionality.
        """,
        "Naive Bayes (Gaussian)": """
        **Gaussian Naive Bayes** assumes feature independence and Gaussian distribution.
        - Very fast training and prediction.
        - Works surprisingly well despite the independence assumption.
        - Performs best when features are truly independent.
        - Provides probabilistic outputs useful for AUC computation.
        - Less accurate when features are highly correlated.
        """,
        "Random Forest (Ensemble)": """
        **Random Forest** is an ensemble of decision trees using bagging.
        - Reduces overfitting compared to a single Decision Tree.
        - Robust to outliers and noisy features.
        - Provides reliable feature importance rankings.
        - Generally achieves the best performance among all models.
        - Computationally more expensive but highly accurate.
        """
    }
    st.markdown(observations.get(model_choice, ""))

# ── Tab 3: All Models Comparison ─────────────
with tab3:
    st.subheader("📊 All Models Comparison")

    if show_all or True:  # Always show comparison
        all_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42),
            "K-Nearest Neighbor (KNN)": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes (Gaussian)": GaussianNB(),
            "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }

        results = []
        cms = {}
        with st.spinner("Training all models... Please wait."):
            for name, mdl in all_models.items():
                _, _, mets, cm_i, _ = train_and_evaluate(
                    mdl, X_train_scaled, X_test_scaled, y_train_full, y_test_full
                )
                row = {"Model": name}
                row.update(mets)
                results.append(row)
                cms[name] = cm_i

        results_df = pd.DataFrame(results).set_index("Model")

        st.subheader("📋 Comparison Table")
        # Highlight best values
        styled = results_df.style.highlight_max(
            axis=0, color='#d4edda'
        ).format("{:.4f}")
        st.dataframe(styled, width='stretch')

        # Best model
        best_model_name = results_df['F1 Score'].idxmax()
        best_acc = results_df.loc[best_model_name, 'Accuracy']
        best_f1 = results_df.loc[best_model_name, 'F1 Score']
        best_auc = results_df.loc[best_model_name, 'AUC']

        st.markdown("---")
        st.markdown(f"""
        <div class="winner-box">
            🏆 Overall Best Model: {best_model_name}<br>
            Accuracy: {best_acc:.4f} | F1 Score: {best_f1:.4f} | AUC: {best_auc:.4f}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 Metrics Comparison Chart")
        fig_comp, ax_comp = plt.subplots(figsize=(12, 5))
        x = np.arange(len(results_df.columns))
        width = 0.15
        colors_comp = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        for i, (model_name, row) in enumerate(results_df.iterrows()):
            offset = (i - 2) * width
            bars = ax_comp.bar(x + offset, row.values, width, label=model_name,
                               color=colors_comp[i], alpha=0.85, edgecolor='white')
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(results_df.columns, fontsize=10)
        ax_comp.set_ylim(0, 1.2)
        ax_comp.set_ylabel("Score", fontsize=11)
        ax_comp.set_title("All Models - Metrics Comparison", fontsize=13, fontweight='bold')
        ax_comp.legend(loc='upper right', fontsize=8, ncol=2)
        ax_comp.axhline(y=0.8, color='gray', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close()

        st.markdown("---")
        st.subheader("🔲 Confusion Matrices - All Models")
        cm_cols = st.columns(3)
        for idx, (name, cm_val) in enumerate(cms.items()):
            with cm_cols[idx % 3]:
                fig_cmi = plot_confusion_matrix(cm_val, title=name)
                st.pyplot(fig_cmi)
                plt.close()

        st.markdown("---")
        st.subheader("💡 Observations Summary")
        obs_data = {
            "ML Model Name": [
                "Logistic Regression", "Decision Tree",
                "K-Nearest Neighbor (KNN)", "Naive Bayes (Gaussian)",
                "Random Forest (Ensemble)"
            ],
            "Observation": [
                "Solid linear baseline. Performs consistently with good AUC. Interpretable coefficients. Slightly limited by linear decision boundary assumption.",
                "Captures non-linear patterns well. Prone to overfitting without depth control. Fast training. Feature importance directly available.",
                "Instance-based learner sensitive to k and scaling. Moderate performance. Computationally heavier at inference time.",
                "Fastest model. Assumes feature independence. Performs reasonably well despite strong assumptions. Good probabilistic outputs.",
                "Best overall performer. Ensemble approach reduces variance. Robust to noise. Highest accuracy and F1 score across all metrics."
            ]
        }
        obs_df = pd.DataFrame(obs_data)
        st.dataframe(obs_df, width='stretch', hide_index=True)

# ── Tab 4: Dataset Info ───────────────────────
with tab4:
    st.subheader("📋 Dataset Information")
    st.markdown("""
    ### Heart Disease Classification Dataset

    **Source:** Inspired by the UCI Heart Disease Dataset (Cleveland Clinic Foundation)  
    **Task:** Binary Classification (0 = No Disease, 1 = Disease Present)  
    **Instances:** 1,000  
    **Features:** 13  

    | Feature | Description | Type |
    |---------|-------------|------|
    | age | Age in years | Numeric |
    | sex | Sex (1=male, 0=female) | Binary |
    | cp | Chest pain type (0-3) | Categorical |
    | trestbps | Resting blood pressure (mm Hg) | Numeric |
    | chol | Serum cholesterol (mg/dl) | Numeric |
    | fbs | Fasting blood sugar > 120 mg/dl | Binary |
    | restecg | Resting ECG results (0-2) | Categorical |
    | thalach | Maximum heart rate achieved | Numeric |
    | exang | Exercise induced angina (1=yes) | Binary |
    | oldpeak | ST depression induced by exercise | Numeric |
    | slope | Slope of peak exercise ST segment | Categorical |
    | ca | Number of major vessels (0-3) | Numeric |
    | thal | Thalassemia type (0-2) | Categorical |
    | **target** | **Heart disease present (1) or not (0)** | **Binary** |
    """)

    st.markdown("---")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("📊 Statistical Summary")
        st.dataframe(full_df.describe().round(2), width='stretch')
    with col_s2:
        st.subheader("🔢 Missing Values")
        missing = full_df.isnull().sum().reset_index()
        missing.columns = ['Feature', 'Missing Count']
        missing['Missing %'] = (missing['Missing Count'] / len(full_df) * 100).round(2)
        st.dataframe(missing, width='stretch', hide_index=True)
        st.success("✅ No missing values in the dataset!")

    st.markdown("---")
    st.subheader("📥 Download Test Data")
    test_csv = full_df.sample(200, random_state=42).to_csv(index=False)
    st.download_button(
        label="⬇️ Download Sample Test Data (200 rows)",
        data=test_csv,
        file_name="test_data.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    ML Assignment 2 | BITS WILP M.Tech AIML/DSE | Machine Learning<br>
    Built with ❤️ using Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
