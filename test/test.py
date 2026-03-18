import joblib
from stable_baselines3 import DQN
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Sabitler ---
MALICIOUS_LABELS = ["phishing", "malware", "defacement", "spam"]
ALL_LABELS = MALICIOUS_LABELS + ["benign"]  # Tüm 5 sınıf


def load_data_for_test(data_path):
    df = pd.read_csv(data_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    target_column = "URL_Type_obf_Type"
    df[target_column] = df[target_column].str.lower().replace("good", "benign")

    df["binary_label"] = df[target_column].apply(lambda x: 0 if x == "benign" else 1)

    X = df.drop(columns=[target_column, "binary_label"])
    y_multiclass = df[target_column].values
    y_binary = df["binary_label"].values

    X_train, X_test, y_multi_train, y_multi_test, y_bin_train, y_bin_test = train_test_split(
        X, y_multiclass, y_binary, test_size=0.3, random_state=42, stratify=df["binary_label"]
    )

    print(f"Test data loaded. Total samples: {len(X_test)}")
    return X_test, y_multi_test, y_bin_test


def get_binary_predictions_and_probs(X_test, binary_model, binary_model_type):
    if binary_model_type == "ml":
        if hasattr(binary_model, 'predict_proba'):
            y_pred_proba = binary_model.predict_proba(X_test)
            y_pred_binary = binary_model.predict(X_test)
        else:
            y_pred_binary = binary_model.predict(X_test)
            y_pred_proba = np.zeros((len(X_test), 2))
    else:
        y_pred_binary = []
        for i in range(len(X_test)):
            feature_values_np = X_test.iloc[[i]].values.flatten()
            pred, _ = binary_model.predict(feature_values_np, deterministic=True)
            y_pred_binary.append(pred)
        y_pred_binary = np.array(y_pred_binary)
        y_pred_proba = np.zeros((len(X_test), 2))

    return y_pred_binary, y_pred_proba


def run_hybrid_prediction(X_test, y_true_multiclass, binary_predictions, dqn_multiclass_model, multiclass_label_map):
    final_predictions = []
    dqn_device = dqn_multiclass_model.device
    dqn_giden_sayi = 0

    for i in range(len(X_test)):
        feature_row_df = X_test.iloc[[i]]
        binary_pred = binary_predictions[i]

        if binary_pred == 0:
            final_predictions.append("benign")
        else:
            dqn_giden_sayi += 1
            feature_values_np = feature_row_df.values.flatten()
            obs_tensor = torch.tensor(feature_values_np, dtype=torch.float32).unsqueeze(0).to(dqn_device)
            with torch.no_grad():
                q_values = dqn_multiclass_model.q_net(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()

            multiclass_label = multiclass_label_map.get(str(action), "unknown_malicious")
            final_predictions.append(multiclass_label)

    coverage_ratio = (dqn_giden_sayi / len(X_test)) * 100
    return final_predictions, coverage_ratio


def evaluate_and_plot(y_true, y_pred, y_true_binary, y_pred_binary, coverage_ratio, title):
    print(f"\n--- {title} ---")
    print(f"\n--- Hibrit Sistem Metriği ---")
    print(f"1. Aşama Kapsam Oranı (DQN'e Gönderilen Pay): {coverage_ratio:.4f}%")

    try:
        auc_score = roc_auc_score(y_true_binary, y_pred_binary)
        print(f"Binary Model AUC Skoru (0/1): {auc_score:.4f}")
    except Exception as e:
        print(f"AUC hesaplanırken hata oluştu: {e}")

    print("\n--- Çok Sınıflı Detaylı Performans (5 Sınıf) ---")
    report_dict = classification_report(y_true, y_pred, labels=ALL_LABELS, zero_division=0, output_dict=True)

    print("\n--- Sınıf Başına (Per-Class) F1-Skorları ---")
    for label in ALL_LABELS:
        if label in report_dict and 'f1-score' in report_dict[label]:
            f1 = report_dict[label]['f1-score'] * 100
            print(f"  {label.ljust(10)}: {f1:.4f}%")

    weighted_avg = report_dict["weighted avg"]
    accuracy = report_dict["accuracy"] * 100
    weighted_precision = weighted_avg["precision"] * 100
    weighted_recall = weighted_avg["recall"] * 100
    weighted_f1 = weighted_avg["f1-score"] * 100

    print("\n--- Weighted Average (Percentage Format) (Genel Özet) ---")
    print(f"Accuracy  : {accuracy:.4f}%")
    print(f"Precision : {weighted_precision:.4f}%")
    print(f"Recall    : {weighted_recall:.4f}%")
    print(f"F1-Score  : {weighted_f1:.4f}%")

    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=ALL_LABELS, yticklabels=ALL_LABELS)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def main():
    # --- Sabit yollar ve model tipi ---
    DATA_PATH = "data/cleaned_feature_data.csv"
    MULTICLASS_DQN_PATH = "multiclass-model-save/multiclass_dqn_model1"
    LABEL_MAP_PATH = "multiclass-model-save/multiclass_labels.json"
    BINARY_MODEL_PATH = "binary-model-save/rf_binary_model.pkl"
    BINARY_MODEL_TYPE = "ml"  # RF -> ML tipi

    print("Yükleniyor: Modeller ve Veriler...")
    try:
        dqn_multiclass_model = DQN.load(MULTICLASS_DQN_PATH)
        binary_model = joblib.load(BINARY_MODEL_PATH)

        with open(LABEL_MAP_PATH, "r") as f:
            multiclass_label_map = json.load(f)

        X_test, y_multi_test, y_bin_test = load_data_for_test(DATA_PATH)
        print("Tüm bileşenler başarıyla yüklendi.")
    except FileNotFoundError as e:
        print(f"HATA: Gerekli dosya bulunamadı: {e.filename}")
        return
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")
        return

    print("\nAdım 1: Binary Model Tahminleri Hesaplanıyor...")
    y_pred_binary, _ = get_binary_predictions_and_probs(X_test, binary_model, BINARY_MODEL_TYPE)

    print("\nAdım 2: Hibrit Boru Hattı Çalıştırılıyor (Binary: RF)...")
    final_predictions, coverage_ratio = run_hybrid_prediction(
        X_test, y_multi_test, y_pred_binary, dqn_multiclass_model, multiclass_label_map
    )

    evaluate_and_plot(
        y_multi_test, final_predictions, y_bin_test, y_pred_binary,
        coverage_ratio, "Filtresiz Uçtan Uca Performans (Binary: RF)"
    )


if __name__ == "__main__":
    main()
