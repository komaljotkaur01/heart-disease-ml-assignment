"""
Script to generate a realistic Heart Disease dataset with 1000 rows and 13 features.
Run this once to create test_data.csv
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

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

# Create target with realistic correlations
score = (
    0.03 * age
    - 0.2 * sex
    + 0.3 * cp
    + 0.01 * trestbps
    + 0.001 * chol
    + 0.1 * fbs
    - 0.001 * thalach
    + 0.4 * exang
    + 0.3 * oldpeak
    - 0.2 * slope
    + 0.3 * ca
    + 0.2 * thal
    + np.random.normal(0, 0.5, n)
)
target = (score > score.mean()).astype(int)

df = pd.DataFrame({
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal,
    'target': target
})

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")
print(f"Features: {list(df.columns)}")

# Save full dataset
df.to_csv('heart_disease_full.csv', index=False)

# Save test split (20%)
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
test_df.to_csv('test_data.csv', index=False)
print(f"\nTest data shape: {test_df.shape}")
print("Files saved: heart_disease_full.csv, test_data.csv")
