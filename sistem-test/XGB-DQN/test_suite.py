import joblib
from stable_baselines3 import DQN
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    cohen_kappa_score,
    log_loss,
    hamming_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    balanced_accuracy_score,
    jaccard_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Model yükleme
try:
    rf_model = joblib.load("sistem-test/XGB-DQN/xgb_binary.pkl")
    dqn_model = DQN.load("sistem-test/XGB-DQN/multiclass_dqn_ozel6V4")
    print("Modeller başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()

# Veri yükleme ve ön işleme
try:
    df = pd.read_csv("data/cleaned_dataV4.csv")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print("Veri başarıyla yüklendi ve temizlendi.")
except Exception as e:
    print(f"Veri yüklenirken hata oluştu: {e}")
    exit()

# Etiketleme (Binary ve Multiclass ayrımı)
target_column = "URL_Type_obf_Type"
df["binary_label"] = df[target_column].apply(
    lambda x: 0 if x.lower() in ["benign", "good"] else 1
)
X = df.drop(columns=[target_column, "binary_label"])
y = df[target_column].str.lower().values

# Veri bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Tahmin başlatılıyor...")

final_predictions = []
binary_predictions = []
true_labels = []

label_map = {0: "defacement", 1: "malware", 2: "phishing", 3: "spam"}

for i in range(len(X_test)):
    feature = pd.DataFrame([X_test.iloc[i]], columns=X_test.columns)
    true_label = y_test[i]
    true_labels.append(true_label)

    try:
        rf_prediction = rf_model.predict(feature)[0]
    except Exception as e:
        print(f"XGB tahmin hatası: {e}")
        continue

    binary_predictions.append(rf_prediction)

    if rf_prediction == 0:
        final_predictions.append("benign")
    else:
        # sadece malicious için multiclass tahmini yapılır
        try:
            obs = feature.values.flatten()
            action, _ = dqn_model.predict(obs)
            multiclass_label = label_map.get(int(action), "unknown")
            final_predictions.append(multiclass_label)
        except Exception as e:
            print(f"DQN tahmin hatası: {e}")
            final_predictions.append("unknown")

# Gerçek etiketleri normalize et
y_test_corrected = ["benign" if label == "good" else label for label in y_test]
binary_true = [0 if label in ["benign", "good"] else 1 for label in y_test]

# === Binary Değerlendirme ===
print("\n=== Binary Değerlendirme ===")
try:
    print(
        classification_report(
            binary_true,
            binary_predictions,
            target_names=["Benign", "Malicious"],
            zero_division=0,
        )
    )
    f1 = f1_score(binary_true, binary_predictions)
    roc_auc = roc_auc_score(binary_true, binary_predictions)
    mcc = matthews_corrcoef(binary_true, binary_predictions)
    gini = 2 * roc_auc - 1

    acc = accuracy_score(binary_true, binary_predictions)
    balanced_acc = balanced_accuracy_score(binary_true, binary_predictions)
    jaccard = jaccard_score(binary_true, binary_predictions)

    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Jaccard Score: {jaccard:.4f}")

    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Gini: {gini:.4f}")
    print(f"Matthews CorrCoef (MCC): {mcc:.4f}")

    cm_bin = confusion_matrix(binary_true, binary_predictions, labels=[0, 1])
    sns.heatmap(
        cm_bin,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Binary Confusion Matrix")
    plt.show()
except Exception as e:
    print(f"Binary metrik hesaplanırken hata oluştu: {e}")

# === Multiclass Değerlendirme (Sadece malicious veriler) ===
print("\n=== Multiclass Değerlendirme ===")
filtered_preds = []
filtered_trues = []

for pred, true in zip(final_predictions, y_test_corrected):
    if true in ["defacement", "malware", "phishing", "spam"] and pred in [
        "defacement",
        "malware",
        "phishing",
        "spam",
    ]:
        filtered_preds.append(pred)
        filtered_trues.append(true)

if filtered_preds:
    labels = ["defacement", "malware", "phishing", "spam"]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    print(
        classification_report(
            filtered_trues, filtered_preds, labels=labels, zero_division=0
        )
    )
    print(f"Cohen's Kappa: {cohen_kappa_score(filtered_trues, filtered_preds):.4f}")
    print(f"Hamming Loss: {hamming_loss(filtered_trues, filtered_preds):.4f}")

    acc_multi = accuracy_score(filtered_trues, filtered_preds)
    balanced_acc_multi = balanced_accuracy_score(filtered_trues, filtered_preds)
    jaccard_multi = jaccard_score(filtered_trues, filtered_preds, average="macro")

    print(f"Accuracy: {acc_multi:.4f}")
    print(f"Balanced Accuracy: {balanced_acc_multi:.4f}")
    print(f"Jaccard Score (macro): {jaccard_multi:.4f}")

    true_encoded_multi = label_encoder.transform(filtered_trues)
    pred_encoded_multi = label_encoder.transform(filtered_preds)
    mcc_multi = matthews_corrcoef(true_encoded_multi, pred_encoded_multi)
    print(f"Matthews CorrCoef (MCC): {mcc_multi:.4f}")

    try:
        log_loss_val = log_loss(
            pd.get_dummies(filtered_trues), pd.get_dummies(filtered_preds)
        )
        print(f"Log Loss: {log_loss_val:.4f}")
    except Exception as e:
        print(f"Log Loss hesaplanamadı: {e}")

    cm_multi = confusion_matrix(filtered_trues, filtered_preds, labels=labels)
    sns.heatmap(
        cm_multi,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Multiclass Confusion Matrix")
    plt.show()
else:
    print("Multiclass için yeterli uygun veri yok.")

# === End-to-End Sistem Değerlendirmesi ===
print("\n=== End-to-End Sistem Değerlendirmesi ===")
try:
    labels = ["benign", "defacement", "malware", "phishing", "spam"]

    filtered_preds = []
    filtered_trues = []

    for pred, true in zip(final_predictions, y_test_corrected):
        if true in labels:
            filtered_trues.append(true)
            filtered_preds.append(pred)

    print(
        classification_report(
            filtered_trues, filtered_preds, labels=labels, zero_division=0
        )
    )
    print(f"Cohen's Kappa: {cohen_kappa_score(filtered_trues, filtered_preds):.4f}")
    print(f"Hamming Loss: {hamming_loss(filtered_trues, filtered_preds):.4f}")
    
    acc_all = accuracy_score(filtered_trues, filtered_preds)
    balanced_acc_all = balanced_accuracy_score(filtered_trues, filtered_preds)
    jaccard_all = jaccard_score(filtered_trues, filtered_preds, average="macro")

    print(f"Accuracy: {acc_all:.4f}")
    print(f"Balanced Accuracy: {balanced_acc_all:.4f}")
    print(f"Jaccard Score (macro): {jaccard_all:.4f}")

    label_encoder_end = LabelEncoder()
    label_encoder_end.fit(labels)
    true_encoded_end = label_encoder_end.transform(filtered_trues)
    pred_encoded_end = label_encoder_end.transform(filtered_preds)
    mcc_end = matthews_corrcoef(true_encoded_end, pred_encoded_end)
    print(f"Matthews CorrCoef (MCC): {mcc_end:.4f}")

    try:
        log_loss_val = log_loss(
            pd.get_dummies(filtered_trues), pd.get_dummies(filtered_preds)
        )
        print(f"Log Loss: {log_loss_val:.4f}")
    except Exception as e:
        print(f"Log Loss hesaplanamadı: {e}")

    cm_all = confusion_matrix(filtered_trues, filtered_preds, labels=labels)
    sns.heatmap(
        cm_all,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("End-to-End Confusion Matrix")
    plt.show()

except Exception as e:
    print(f"End-to-End metrik hesaplanırken hata oluştu: {e}")
