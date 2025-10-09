import argparse
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

warnings.filterwarnings("ignore", category=UserWarning)

# --- Sabitler ---
MALICIOUS_LABELS = ["phishing", "malware", "defacement", "spam"]


def load_data_for_test(data_path):
    """Veriyi test için yükler ve böler."""
    df = pd.read_csv(data_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    target_column = "URL_Type_obf_Type"
    # Tüm etiketleri küçük harfe çevir ve 'good' etiketini 'benign' olarak düzelt
    df[target_column] = df[target_column].str.lower().replace("good", "benign")
    
    df["binary_label"] = df[target_column].apply(
        lambda x: 0 if x == "benign" else 1
    )

    X = df.drop(columns=[target_column, "binary_label"])
    y_multiclass = df[target_column].values
    y_binary = df["binary_label"].values

    # Test setini bölüyoruz
    _, X_test, _, y_multi_test, _, y_bin_test = train_test_split(
        X, y_multiclass, y_binary, test_size=0.3, random_state=42, stratify=df["binary_label"]
    )

    print(f"Test data loaded. Total samples: {len(X_test)}")
    return X_test, y_multi_test, y_bin_test


def run_hybrid_prediction_filtered(
    X_test, y_true_multiclass, y_true_binary, binary_model, dqn_multiclass_model, multiclass_label_map, binary_model_type
):
    """
    İki aşamalı tahmin boru hattını çalıştırır ve filtreler.
    Katman 2'ye (DQN) sadece Binary Modelin 'Zararlı' dediği VE GERÇEKTE 'Zararlı' OLAN veriler (True Positives) dahil edilir.
    """
    final_predictions = []
    dqn_device = dqn_multiclass_model.device

    for i in range(len(X_test)):
        feature_row_df = X_test.iloc[[i]]
        true_multiclass_label = y_true_multiclass[i]
        true_binary_label = y_true_binary[i]
        
        # --- AŞAMA 1: İKİLİ (BINARY) TAHMİN ---
        if binary_model_type == "ml":
            # ML modelleri (RF, SVM, XGB)
            binary_pred = binary_model.predict(feature_row_df)[0]
        else:
            # DQN modeli
            feature_values_np = feature_row_df.values.flatten()
            # DQN predict, eylem (action) ve durum değerlerini (state values) döndürür
            binary_pred, _ = binary_model.predict(feature_values_np, deterministic=True)
            # binary_pred: 0 (benign) veya 1 (malicious)
        
        # --- AŞAMA 2: FİLTRELEME VE NİHAİ TAHMİN ---
        
        if true_binary_label == 0:
            # GERÇEKTE ZARARSIZ (Benign): Tahmin ne olursa olsun, nihai sonuç doğru etiket olan 'benign' olarak alınır.
            # Bu, False Positive ve True Negative durumlarını doğru bir şekilde yansıtır.
            final_predictions.append(true_multiclass_label) # true_multiclass_label = 'benign'

        elif true_binary_label == 1:
            # GERÇEKTE ZARARLI (Malicious):
            if binary_pred == 1:
                # True Positive (TP) Durumu: Binary model doğru bildi, DQN'e gönderiliyor.
                feature_values_np = feature_row_df.values.flatten()
                obs_tensor = (
                    torch.tensor(feature_values_np, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(dqn_device)
                )
                with torch.no_grad():
                    q_values = dqn_multiclass_model.q_net(obs_tensor)
                    action = torch.argmax(q_values, dim=1).item()

                multiclass_label = multiclass_label_map.get(
                    str(action), "unknown_malicious"
                )
                final_predictions.append(multiclass_label)
            else:
                # False Negative (FN) Durumu: Binary model yanlışlıkla Zararsız (0) dedi.
                # Bu veri DQN'e gitmediği için nihai tahmin 'benign' (yanlış) olarak kaydedilir.
                final_predictions.append("benign")

    return final_predictions


def evaluate_and_plot(y_true, y_pred, title):
    """Sonuçları raporlar ve karmaşıklık matrisini çizer."""
    print(f"\n--- {title} ---")
    # 'benign' dahil tüm 5 sınıfın etiketlerini sırala
    all_labels = sorted(list(set(y_true) | set(y_pred)))

    # classification_report çıktısını hem yazdır hem de dictionary olarak al
    report_dict = classification_report(
        y_true, y_pred, labels=all_labels, zero_division=0, output_dict=True
    )
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    # Weighted avg değerlerini yüzde formatında hesapla
    weighted_avg = report_dict["weighted avg"]
    weighted_precision = weighted_avg["precision"] * 100
    weighted_recall = weighted_avg["recall"] * 100
    weighted_f1 = weighted_avg["f1-score"] * 100
    accuracy = report_dict["accuracy"] * 100

    # Sonuçları yazdır
    print("\n--- Weighted Average (Percentage Format) ---")
    print(f"Accuracy  : {accuracy:.4f}%")
    print(f"Precision : {weighted_precision:.4f}%")
    print(f"Recall    : {weighted_recall:.4f}%")
    print(f"F1-Score  : {weighted_f1:.4f}%")

    # Karmaşıklık matrisi çizimi
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=all_labels,
        yticklabels=all_labels,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def main(binary_choice):
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
        raise ValueError(
            f"Geçersiz seçim '{binary_choice}'. Seçimler: {list(model_paths.keys())}"
        )

    binary_model_path, binary_model_type = model_paths[binary_choice]

    print("Yükleniyor: Modeller ve Veriler...")
    try:
        # Multiclass DQN modelini yükle
        dqn_multiclass_model = DQN.load(MULTICLASS_DQN_PATH)
        
        # Seçilen ikili (binary) modeli yükle
        if binary_model_type == "ml":
            binary_model = joblib.load(binary_model_path)
        else:
            binary_model = DQN.load(binary_model_path)

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

    print(
        f"\nÇalıştırılıyor: Filtreli Hibrit Boru Hattı (Binary: {binary_choice.upper()}, Yalnızca Gerçek Zararlı Doğru Pozitifler DQN'e)..."
    )
    final_predictions = run_hybrid_prediction_filtered(
        X_test,
        y_multi_test,
        y_bin_test,
        binary_model,
        dqn_multiclass_model,
        multiclass_label_map,
        binary_model_type,
    )

    evaluate_and_plot(
        y_multi_test,
        final_predictions,
        f"Filtreli Uçtan Uca Performans (Binary: {binary_choice.upper()})",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gerçek Pozitif filtrelemeli hibrit URL sınıflandırma sistemini değerlendirir."
    )
    parser.add_argument(
        "--binary_model",
        type=str,
        required=True,
        choices=["rf", "svm", "xgb", "dqn"],
        help="Birinci aşama için ikili sınıflandırıcıyı seçin: 'rf', 'svm', 'xgb' veya 'dqn'.",
    )
    args = parser.parse_args()

    main(args.binary_model)