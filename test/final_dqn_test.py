import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from stable_baselines3 import DQN
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

print("\n--- Training Complete: Final Test (DQN Multiclass) ---")

# === Load Data ===
DATA_PATH = "data/cleaned_feature_data.csv"
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"ERROR: Data file not found: {DATA_PATH}")
    exit()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"
# Filter only malicious URL types (Layer 2 task)
df = df[df[target_column].str.lower().isin(['phishing', 'malware', 'defacement', 'spam'])].copy()

# X_test must contain only numeric features
X_test = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
y_test = df[target_column].str.lower().values

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = DQN.load("multiclass-model-save/multiclass_dqn_model", device=device)
except FileNotFoundError:
    print("ERROR: DQN model 'multiclass-model-save/multiclass_dqn_model' not found.")
    exit()

# === Load Label Map ===
try:
    with open("multiclass-model-save/multiclass_labels.json", "r") as f:
        label_map = json.load(f)
except FileNotFoundError:
    print("ERROR: Multiclass label map file not found.")
    exit()

# === Make Predictions ===
y_pred = []
for i in range(len(X_test)):
    obs = X_test.iloc[i].values
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
        action = torch.argmax(q_values, dim=1).item()
        y_pred.append(action)

# === Convert predictions to string labels and filter ===
y_pred_labels = [label_map.get(str(i), "unknown") for i in y_pred]

# Filter out 'benign' and 'good' from valid labels and sort them
valid_labels = sorted(list(set(label_map.values()) - {'benign', 'good'}))
valid_labels_set = set(valid_labels)

# Filter the test set based on valid malicious labels
filtered_indices = [i for i, label in enumerate(y_test) if label in valid_labels_set]
y_test_filtered = [y_test[i] for i in filtered_indices]
y_pred_filtered = [y_pred_labels[i] for i in filtered_indices]

# === Generate Report ===
report_dict = classification_report(
    y_test_filtered,
    y_pred_filtered,
    target_names=valid_labels,
    zero_division=0,
    output_dict=True
)

print(classification_report(
    y_test_filtered,
    y_pred_filtered,
    target_names=valid_labels,
    zero_division=0
))

# === Weighted Averages ===
accuracy = accuracy_score(y_test_filtered, y_pred_filtered) * 100
precision = report_dict["weighted avg"]["precision"] * 100
recall = report_dict["weighted avg"]["recall"] * 100
f1 = report_dict["weighted avg"]["f1-score"] * 100

print("\n--- Weighted Average Results ---")
print(f"Precision : {precision:.6f}%")
print(f"Recall    : {recall:.6f}%")
print(f"F1-Score  : {f1:.6f}%")
print(f"Accuracy  : {accuracy:.6f}%")

# ----- CONFUSION MATRIX VISUALIZATION (ENGLISH) -----

# Calculate Confusion Matrix
cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=valid_labels)

plt.figure(figsize=(8, 7))
sns.heatmap(
    cm, 
    annot=True,         
    fmt='d',            
    cmap='magma',       
    xticklabels=valid_labels, 
    yticklabels=valid_labels 
)
plt.title('DQN Multiclass Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Type', fontsize=12)
plt.ylabel('True Type', fontsize=12)
plt.show()

print("\n Confusion Matrix visualization successfully created.")