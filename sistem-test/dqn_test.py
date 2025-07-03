from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN
import pandas as pd
import numpy as np
import torch

data_path = "data/cleaned_dataV4.csv"
df = pd.read_csv(data_path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"
df["binary_label"] = df[target_column].apply(
    lambda x: 0 if x in ["benign", "good"] else 1
)
df_malicious = df[df["binary_label"] == 1]
df_malicious = df_malicious[~df_malicious[target_column].isin(["malicious"])]

features = df_malicious.drop(columns=[target_column, "binary_label"]).values
labels = df_malicious[target_column].astype("category").cat.codes.values

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, stratify=labels, random_state=42
)
url_types = df_malicious[target_column].unique()

print("\n--- DQN Test ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN.load(
    "multiclass-model-add-spam-92-new/multiclass_dqn_modelV4-92-new", device=device
)

y_pred = []
for obs in X_test:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
        action = torch.argmax(q_values, dim=1).item()
        y_pred.append(action)

from scipy.special import softmax

y_proba = [
    softmax(
        model.q_net(torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device))
        .detach()
        .cpu()
        .numpy()[0]
    )
    for x in X_test
]
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize

# Gerçek etiketleri one-hot encode et
y_test_onehot = label_binarize(y_test, classes=np.unique(labels))

# Log loss hesapla
log_loss_val = log_loss(y_test_onehot, y_proba)
print(f"Log Loss: {log_loss_val:.4f}")



y_pred_bin = label_binarize(np.argmax(y_proba, axis=1), classes=np.unique(labels))
roc_auc = roc_auc_score(y_test_onehot, y_pred_bin, average="macro", multi_class="ovr")
print(f"ROC-AUC: {roc_auc:.4f}")


print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1-Score: ", f1_score(y_test, y_pred, average="weighted"))
print("Precision: ", precision_score(y_test, y_pred, average="weighted"))
print("Recall: ", recall_score(y_test, y_pred, average="weighted"))
print(classification_report(y_test, y_pred, target_names=[str(i) for i in url_types]))
