# ❤️ Heart Disease Classification — ML Assignment 2

**BITS WILP M.Tech AIML/DSE | Machine Learning**  
**Marks: 15 | Deadline: 18-Aug-2026**

---

## a. Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early and accurate prediction of heart disease can significantly improve patient outcomes and reduce healthcare costs. This project implements a **binary classification** system to predict whether a patient has heart disease (1) or not (0) based on 13 clinical and physiological features.

**Objective:** Build, evaluate, and deploy multiple Machine Learning classification models on a heart disease dataset, compare their performance using standard evaluation metrics, and serve the results through an interactive Streamlit web application.

**Classification Task:** Binary Classification  
- Class 0 → No Heart Disease  
- Class 1 → Heart Disease Present  

---

## b. Dataset Description

| Property | Value |
|----------|-------|
| **Dataset Name** | Heart Disease Classification Dataset |
| **Inspired By** | UCI Heart Disease Dataset (Cleveland Clinic Foundation) |
| **Total Instances** | 1,000 |
| **Number of Features** | 13 |
| **Target Variable** | `target` (0 = No Disease, 1 = Disease) |
| **Task Type** | Binary Classification |
| **Missing Values** | None |
| **Class Distribution** | Class 0: 489 (48.9%) / Class 1: 511 (51.1%) |

### Feature Description

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | `age` | Age of the patient (years) | Numeric |
| 2 | `sex` | Sex (1 = male, 0 = female) | Binary |
| 3 | `cp` | Chest pain type (0–3) | Categorical |
| 4 | `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| 5 | `chol` | Serum cholesterol (mg/dl) | Numeric |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) | Binary |
| 7 | `restecg` | Resting ECG results (0–2) | Categorical |
| 8 | `thalach` | Maximum heart rate achieved | Numeric |
| 9 | `exang` | Exercise-induced angina (1 = yes) | Binary |
| 10 | `oldpeak` | ST depression induced by exercise | Numeric |
| 11 | `slope` | Slope of peak exercise ST segment (0–2) | Categorical |
| 12 | `ca` | Number of major vessels colored by fluoroscopy (0–3) | Numeric |
| 13 | `thal` | Thalassemia type (0–2) | Categorical |
| 14 | `target` | **Heart disease present (1) or not (0)** | **Binary (Target)** |

### Preprocessing Steps
1. **Train/Test Split:** 80% training (800 samples) / 20% testing (200 samples), stratified
2. **Feature Scaling:** `StandardScaler` applied to all features (required for LR, KNN)
3. **No missing values** — no imputation needed
4. **No categorical encoding** — all features already numeric

---

## c. GitHub Repository Link

> 🔗 **[https://github.com/komaljotkaur01/heart-disease-ml-assignment](https://github.com/komaljotkaur01/heart-disease-ml-assignment)**

### Live Streamlit App Link

> 🔗 **[https://heart-disease-ml-assignment-j8devpdbr9nr9vyegwxr36.streamlit.app/](https://heart-disease-ml-assignment-j8devpdbr9nr9vyegwxr36.streamlit.app/)**

### Repository Structure

```
heart-disease-ml-assignment/
│
├── app.py                    # Streamlit web application (main entry point)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── test_data.csv             # Test dataset (200 rows, 14 columns)
├── heart_disease_full.csv    # Full dataset (1000 rows)
├── generate_dataset.py       # Dataset generation script
├── ML_Assignment_2.pdf       # Assignment PDF
│
└── model/
    ├── train_models.py       # Model training & evaluation script
    ├── metrics_comparison.png
    ├── confusion_matrices.png
    └── feature_importance.png
```

---

## d. Models Used

### Train/Test Configuration
- **Training Set:** 800 samples (80%)
- **Test Set:** 200 samples (20%)
- **Random State:** 42 (reproducible)
- **Feature Scaling:** StandardScaler (zero mean, unit variance)

---

### 📊 Comparison Table — Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8250 | 0.9324 | 0.8018 | 0.8725 | 0.8357 | 0.6519 |
| Decision Tree | 0.7650 | 0.8138 | 0.7523 | 0.8039 | 0.7773 | 0.5305 |
| kNN | 0.7300 | 0.8041 | 0.7182 | 0.7745 | 0.7453 | 0.4604 |
| Naive Bayes | **0.8350** | **0.9329** | **0.8165** | **0.8725** | **0.8436** | **0.6711** |
| Random Forest (Ensemble) | 0.8200 | 0.9063 | 0.8000 | 0.8627 | 0.8302 | 0.6413 |

> ✅ **Bold values** indicate the best score for each metric.  
> All metrics computed on the held-out test set (200 samples).

---

### 💡 Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| **Logistic Regression** | Achieved strong performance (Accuracy: 82.5%, AUC: 0.9324) as a linear baseline. The high AUC indicates excellent discriminative ability between classes. Feature scaling via StandardScaler was critical for convergence. The model's interpretable coefficients reveal that `exang`, `oldpeak`, and `ca` are the most influential features. Regularization (C=1.0) effectively prevented overfitting. Slightly limited by its linear decision boundary assumption, but performs remarkably well given the dataset's structure. |
| **Decision Tree** | Achieved moderate performance (Accuracy: 76.5%, AUC: 0.8138). The tree captures non-linear relationships and provides directly interpretable if-else rules. Without depth constraints, it would overfit; `max_depth=5` was applied to regularize. Feature importance is directly extractable. The lower AUC compared to LR and NB suggests the tree's hard decision boundaries are less calibrated for probability estimation. Fast to train and predict. |
| **kNN** | Achieved the lowest performance among all models (Accuracy: 73.0%, AUC: 0.8041). As a non-parametric, instance-based learner, KNN is sensitive to the choice of k (k=5 used) and feature scaling. The moderate performance suggests that the dataset's decision boundaries are not purely local. Computationally heavier at inference time as it requires distance computation against all training samples. Sensitive to irrelevant features and the curse of dimensionality. |
| **Naive Bayes** | Achieved the **best overall performance** (Accuracy: 83.5%, AUC: 0.9329, F1: 0.8436, MCC: 0.6711). Despite the strong independence assumption between features, Gaussian NB performed surprisingly well. The dataset's features, while correlated, appear to have sufficient class-conditional Gaussian distributions for NB to exploit. Extremely fast training and prediction. The high AUC (0.9329) indicates excellent probabilistic calibration. |
| **Random Forest (Ensemble)** | Achieved strong performance (Accuracy: 82.0%, AUC: 0.9063). As an ensemble of 100 decision trees with bagging, it significantly outperforms a single Decision Tree. Robust to outliers and noisy features. Provides reliable feature importance rankings. The slight underperformance compared to Naive Bayes on this dataset may be due to the relatively small dataset size (1000 samples) where ensemble benefits are less pronounced. Generally the most robust choice for larger, noisier datasets. |

---

### 🏆 Overall Winner

| | |
|---|---|
| **Overall Winner** | **Naive Bayes (Gaussian)** |
| **Accuracy** | 0.8350 |
| **AUC** | 0.9329 |
| **Precision** | 0.8165 |
| **Recall** | 0.8725 |
| **F1 Score** | 0.8436 |
| **MCC** | 0.6711 |

**Justification:** Naive Bayes achieved the highest scores across all 6 evaluation metrics — Accuracy (83.5%), AUC (0.9329), Precision (81.65%), Recall (87.25%), F1 Score (84.36%), and MCC (0.6711). The high AUC of 0.9329 indicates excellent ability to discriminate between patients with and without heart disease. The high Recall (87.25%) is particularly important in a medical diagnosis context, as it minimizes false negatives (missed disease cases). Despite its simplifying independence assumption, the Gaussian NB model's probabilistic nature and fast inference make it an excellent choice for this binary classification task.

---

## 🚀 Streamlit App Features

The deployed Streamlit application includes:

| Feature | Description |
|---------|-------------|
| ✅ **CSV Upload** | Upload test data (CSV) for evaluation |
| ✅ **Model Selection Dropdown** | Choose from all 5 classification models |
| ✅ **Evaluation Metrics Display** | Shows Accuracy, AUC, Precision, Recall, F1, MCC |
| ✅ **Confusion Matrix** | Visual heatmap of prediction results |
| ✅ **Classification Report** | Detailed per-class precision/recall/F1 |
| ✅ **All Models Comparison** | Side-by-side comparison table and bar chart |
| ✅ **Dataset Overview** | Distribution plots and correlation heatmap |
| ✅ **Hyperparameter Tuning** | Interactive sliders for model parameters |

### Live App Link
> 🌐 **[https://heart-disease-ml-assignment-j8devpdbr9nr9vyegwxr36.streamlit.app/](https://heart-disease-ml-assignment-j8devpdbr9nr9vyegwxr36.streamlit.app/)**

---

## 🛠️ Setup & Installation

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/komaljotkaur01/heart-disease-ml-assignment.git
cd heart-disease-ml-classifier

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

### Run Model Training Script

```bash
# Train all models and generate evaluation plots
python model/train_models.py
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push all files to a **public GitHub repository**
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click **"New App"**
5. Select your repository and branch (`main`)
6. Set **Main file path** to `app.py`
7. Click **"Deploy"**

> ⚠️ Ensure `requirements.txt` lists all dependencies correctly to avoid deployment failures.

---

## 📦 Dependencies

```
streamlit>=1.28.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## ✅ Final Submission Checklist

- [x] GitHub repo link works
- [x] Streamlit app link opens correctly
- [x] App loads without errors
- [x] CSV upload feature implemented
- [x] Model selection dropdown implemented
- [x] Evaluation metrics displayed (Accuracy, AUC, Precision, Recall, F1, MCC)
- [x] Confusion matrix displayed
- [x] Classification report displayed
- [x] All 5 models implemented (LR, DT, KNN, NB, RF)
- [x] README.md updated with all required sections
- [x] test_data.csv included in repository
- [x] Dataset has ≥ 500 instances and ≥ 12 features
- [x] Observations table completed for all models
- [x] Overall winner identified with justification

---

## 📸 Screenshots

### Streamlit App Interface — Overview Tab

The interactive Streamlit application features a user-friendly interface for heart disease classification:

<img src="https://imgur.com/upload" alt="Heart Disease ML App - Overview Tab" width="900">

**Features shown in this screenshot:**
- ❤️ **Title:** Heart Disease Classification with subtitle (ML Assignment 2 | BITS WILP M.Tech AIML/DSE)
- 📋 **Problem Statement section** with detailed description
- 📊 **Dataset Statistics:** 1,000 total instances, 13 features, Binary Classification task
- 🏥 **Objective:** Predict heart disease based on clinical features
- 🎯 **Controls Panel:** Upload test data (CSV format, 200MB per file limit)
- 📑 **Model Selection:** Dropdown menu to choose classification model
- 📋 **Dataset Preview:** Scroll through data with search functionality

**Navigation Tabs:**
- **Overview Tab** — Problem statement, dataset statistics, and model information
- **Model Evaluation Tab** — Upload test CSV, select model, view metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- **AI Models Tab** — Comparison of all 5 classification models
- **Dataset Info Tab** — Data distribution, feature statistics, and correlation heatmap

**Live Demo:** 🌐 [https://heart-disease-ml-assignment-j8devpdbr9nr9vyegwxr36.streamlit.app/](https://heart-disease-ml-assignment-j8devpdbr9nr9vyegwxr36.streamlit.app/)

---

*ML Assignment 2 | BITS WILP M.Tech AIML/DSE | Machine Learning*  
*Built with ❤️ using Python, Scikit-learn, and Streamlit*
