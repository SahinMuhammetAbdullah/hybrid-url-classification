import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from stable_baselines3 import DQN
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

print("\n--- Eğitim Tamamlandı: Son Test ---")

# === Veriyi yükle ===
DATA_PATH = "data/cleaned_feature_data.csv"
df = pd.read_csv(DATA_PATH)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"
# Sadece zararlı URL türlerini al
df = df[df[target_column].str.lower() != "benign"]

X_test = df.drop(columns=[target_column])
y_test = df[target_column].str.lower().values

# === Modeli yükle ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN.load("multiclass-model-save/multiclass_dqn_model", device=device)

# === Etiket haritasını yükle ===
with open("multiclass-model-save/multiclass_labels.json", "r") as f:
    label_map = json.load(f)

# benign olmayanları filtrele
label_map = {k: v for k, v in label_map.items() if v.lower() != "benign"}

# === Tahmin yap ===
y_pred = []
for i in range(len(X_test)):
    obs = X_test.iloc[i].values
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
        action = torch.argmax(q_values, dim=1).item()
        y_pred.append(action)

# === Tahminleri string etiketlere çevir ===
y_pred_labels = [label_map.get(str(i), "unknown") for i in y_pred]

# === Sadece tanımlı etiketleri filtrele ===
valid_labels = list(label_map.values())
filtered_indices = [i for i, label in enumerate(y_test) if label in valid_labels]
y_test_filtered = [y_test[i] for i in filtered_indices]
y_pred_filtered = [y_pred_labels[i] for i in filtered_indices]

# === Rapor oluştur ===
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

# === Weighted ortalamalar ===
accuracy = accuracy_score(y_test_filtered, y_pred_filtered) * 100
precision = report_dict["weighted avg"]["precision"] * 100
recall = report_dict["weighted avg"]["recall"] * 100
f1 = report_dict["weighted avg"]["f1-score"] * 100

print("\n--- Weighted Average Results ---")
print(f"Precision : {precision:.6f}%")
print(f"Recall    : {recall:.6f}%")
print(f"F1-Score  : {f1:.6f}%")
print(f"Accuracy  : {accuracy:.6f}%")
