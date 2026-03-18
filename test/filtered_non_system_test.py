import joblib
from stable_baselines3 import DQN
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import os
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning)


# --- Veri Yükleme ---
def load_data_for_test(data_path):
    print(f"Veri yükleniyor: {data_path}")
    df = pd.read_csv(data_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    target_column = "URL_Type_obf_Type"
    # Etiketleri Türkçeleştirme (Veri setindeki karşılıklara göre)
    df[target_column] = (
        df[target_column]
        .str.lower()
        .replace(
            {
                "good": "Güvenli",
                "benign": "Güvenli",
                "phishing": "Oltalama",
                "malware": "Zararlı Yazılım",
                "defacement": "Tahrif",
                "spam": "Spam",
            }
        )
    )

    df["binary_label"] = df[target_column].apply(lambda x: 0 if x == "Güvenli" else 1)

    X = df.drop(columns=[target_column, "binary_label"])
    y_multiclass = df[target_column].values

    _, X_test, _, y_multi_test = train_test_split(
        X, y_multiclass, test_size=0.3, random_state=42, stratify=df["binary_label"]
    )

    return X_test, y_multi_test


# --- Hibrit Tahmin ---
def run_hybrid_prediction_fast(
    X_test, binary_model, dqn_multiclass_model, multiclass_label_map, binary_model_type
):
    total_samples = len(X_test)
    final_predictions = np.array(["Güvenli"] * total_samples, dtype=object)

    if binary_model_type == "ml":
        binary_preds = binary_model.predict(X_test)
    else:
        binary_preds, _ = binary_model.predict(X_test.values, deterministic=True)

    mask_malicious = binary_preds == 1
    indices_to_process = np.where(mask_malicious)[0]

    if len(indices_to_process) > 0:
        malicious_features = X_test.iloc[indices_to_process].values
        dqn_device = dqn_multiclass_model.device
        obs_tensor = torch.tensor(malicious_features, dtype=torch.float32).to(
            dqn_device
        )

        with torch.no_grad():
            q_values = dqn_multiclass_model.q_net(obs_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # DQN etiketlerini Türkçeye çevirme haritası
        tr_map = {
            "phishing": "Oltalama",
            "malware": "Zararlı Yazılım",
            "defacement": "Tahrif",
            "spam": "Spam",
            "benign": "Güvenli",
        }

        multiclass_results = [
            tr_map.get(multiclass_label_map.get(str(a)).lower(), "bilinmeyen")
            for a in actions
        ]
        final_predictions[indices_to_process] = multiclass_results

    return final_predictions


# --- Detaylı ve Açık Renkli Değerlendirme ---
def evaluate_and_plot_detailed(y_true, y_pred, title):
    all_labels = sorted(list(set(y_true) | set(y_pred)))

    print(f"\n{'='*65}\n{title}\n{'='*65}")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    report_dict = classification_report(
        y_true, y_pred, labels=all_labels, zero_division=0, output_dict=True
    )

    acc = report_dict["accuracy"] * 100
    pre = report_dict["weighted avg"]["precision"] * 100
    rec = report_dict["weighted avg"]["recall"] * 100
    f1 = report_dict["weighted avg"]["f1-score"] * 100

    print(f"--- DETAYLI PERFORMANS ANALİZİ (%) ---")
    print(f"Doğruluk (Accuracy)    : {acc:.5f}%")
    print(f"Keskinlik (Precision)  : {pre:.5f}%")
    print(f"Duyarlılık (Recall)    : {rec:.5f}%")
    print(f"F1-Skoru (F1-Score)    : {f1:.5f}%")
    print(f"{'='*65}")

    # --- Görselleştirme (Açık Renk Teması) ---
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(12, 8))

    # cmap="Blues" parametresi açık mavi tonlarında bir matris oluşturur
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=all_labels,
        yticklabels=all_labels,
        cbar_kws={"label": "Örnek Sayısı"},
        linewidths=0.5,
        linecolor="lightgrey",
    )

    plt.title(title, fontsize=14, pad=20)
    plt.ylabel("Gerçek Sınıf (True Label)", fontsize=12)
    plt.xlabel("Tahmin Edilen Sınıf (Predicted Label)", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def main():
    binary_choice = "rf"
    if len(sys.argv) > 1:
        binary_choice = sys.argv[1].lower()

    DATA_PATH = "data/cleaned_feature_data.csv"
    MULTICLASS_DQN_PATH = "multiclass-model-save/multiclass_dqn_model"
    LABEL_MAP_PATH = "multiclass-model-save/multiclass_labels.json"

    model_paths = {
        "rf": ("binary-model-save/rf_binary_model.pkl", "ml"),
        "svm": ("binary-model-save/svm_binary_model.pkl", "ml"),
        "xgb": ("binary-model-save/xgb_binary_model.pkl", "ml"),
        "dqn": ("binary-model-save/binary_dqn_model", "dqn"),
    }

    if binary_choice not in model_paths:
        print(f"Seçim geçersiz: {binary_choice}")
        return

    path, m_type = model_paths[binary_choice]

    try:
        dqn_multiclass_model = DQN.load(MULTICLASS_DQN_PATH)
        binary_model = joblib.load(path) if m_type == "ml" else DQN.load(path)

        with open(LABEL_MAP_PATH, "r") as f:
            multiclass_label_map = json.load(f)

        X_test, y_multi_test = load_data_for_test(DATA_PATH)

        final_preds = run_hybrid_prediction_fast(
            X_test, binary_model, dqn_multiclass_model, multiclass_label_map, m_type
        )

        evaluate_and_plot_detailed(
            y_multi_test,
            final_preds,
            f"Hibrit Sistem Performansı",
        )

    except Exception as e:
        print(f"HATA: {e}")


if __name__ == "__main__":
    main()
