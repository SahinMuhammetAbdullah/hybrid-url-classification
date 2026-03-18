import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# === File Path ===
data_path = "data/cleaned_feature_data.csv"

# === Step 1: Data Preparation ===
df = pd.read_csv(data_path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"

# Binary labeling: 0 = benign/good, 1 = malicious/other
df["binary_label"] = df[target_column].apply(lambda x: 0 if x in ["benign", "good"] else 1)

# Select only numerical features
X = df.drop(columns=[target_column, "binary_label"]).select_dtypes(include=[np.number])
y_binary = df["binary_label"]

# === Step 2: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# === Step 3: Random Forest Model Training ===
rf_binary = RandomForestClassifier(n_estimators=100, random_state=42)
rf_binary.fit(X_train, y_train)

# === Step 4: Evaluation ===
y_pred_rf_binary = rf_binary.predict(X_test)
print("\nRandom Forest Model Performance (Malicious Detection):")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_binary))
print("Classification Report:\n", classification_report(y_test, y_pred_rf_binary, target_names=["Benign", "Malicious"]))

# === Step 5: Save Model ===
save_dir = "binary-model-save"
os.makedirs(save_dir, exist_ok=True)  # klasör yoksa oluşturur

joblib.dump(rf_binary, os.path.join(save_dir, "rf_binary_model.pkl"))
