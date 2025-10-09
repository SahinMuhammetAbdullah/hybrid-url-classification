import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ----- Argümanları Tanımla -----
parser = argparse.ArgumentParser(description="Binary Model Test Tool")
parser.add_argument("-binary", type=str, required=True, help="Model type: rf, svm, dt, etc.")
args = parser.parse_args()

# ----- Dosya Yolları -----
data_path = "data/cleaned_feature_data.csv"
model_dir = "binary-model-save"

# ----- Veri Seti Yükleme ve Hazırlama -----
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found: {data_path}")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"
# Creating binary label: 0 for benign/good, 1 for malicious
df['binary_label'] = df[target_column].apply(lambda x: 0 if x.lower() in ["benign", "good"] else 1)

# Selecting only numeric features
X = df.drop(columns=[target_column, 'binary_label']).select_dtypes(include=[np.number])
y = df['binary_label']

# ----- Model Yükleme -----
model_name = f"{model_dir}/{args.binary}_binary_model.pkl"
zip_model_name = f"{model_dir}/{args.binary}_binary_model.zip"

try:
    model = joblib.load(model_name)
    print(f"Model loaded: {model_name}")
except FileNotFoundError:
    try:
        model = joblib.load(zip_model_name)
        print(f"Model loaded: {zip_model_name}")
    except FileNotFoundError:
        raise FileNotFoundError(f"{args.binary} model not found as .pkl or .zip.")

# ----- Tahmin ve Raporlama -----
y_pred = model.predict(X)

acc = accuracy_score(y, y_pred) * 100
prec = precision_score(y, y_pred, average='weighted') * 100
rec = recall_score(y, y_pred, average='weighted') * 100
f1 = f1_score(y, y_pred, average='weighted') * 100

print("\n=== Binary Model Test Results ===")
print(f"Model: {args.binary.upper()}")
print(f"Accuracy: {acc:.6f}")
print(f"Precision: {prec:.6f}")
print(f"Recall: {rec:.6f}")
print(f"F1-Score: {f1:.6f}")

print("\n--- Detailed Classification Report ---")
print(classification_report(y, y_pred, target_names=["Benign (0)", "Malicious (1)"]))

# ----- CONFUSION MATRIX VISUALIZATION (ENGLISH) -----

# Calculate Confusion Matrix
cm = confusion_matrix(y, y_pred)
labels = ["Benign (0)", "Malicious (1)"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, 
    annot=True,       
    fmt='d',          
    cmap='Blues',     
    xticklabels=labels,
    yticklabels=labels 
)
plt.title(f'{args.binary.upper()} Model Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

print("\n Confusion Matrix visualization successfully created.")