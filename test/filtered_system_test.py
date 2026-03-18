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

# --- Sabitler ---
DATA_PATH = "data/cleaned_feature_data.csv"
MULTICLASS_DQN_PATH = "multiclass-model-save/multiclass_dqn_model1"
LABEL_MAP_PATH = "multiclass-model-save/multiclass_labels.json"

def load_data_for_test(data_path):
    """Veriyi yükler, temizler ve test setine böler."""
    print(f"Veri yükleniyor: {data_path}")
    df = pd.read_csv(data_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    target_column = "URL_Type_obf_Type"
    # Etiket standardizasyonu
    df[target_column] = df[target_column].str.lower().replace("good", "benign")
    
    # İkili etiketleme: Benign=0, Diğerleri=1
    df["binary_label"] = df[target_column].apply(lambda x: 0 if x == "benign" else 1)

    X = df.drop(columns=[target_column, "binary_label"])
    y_multiclass = df[target_column].values
    y_binary = df["binary_label"].values

    # Stratified split ile veri dengesini koru
    _, X_test, _, y_multi_test, _, y_bin_test = train_test_split(
        X, y_multiclass, y_binary, test_size=0.3, random_state=42, stratify=df["binary_label"]
    )

    print(f"Veri hazır. Toplam test örneği: {len(X_test)}")
    return X_test, y_multi_test, y_bin_test

def run_hybrid_prediction_fast(X_test, y_true_binary, binary_model, dqn_multiclass_model, multiclass_label_map, binary_model_type):
    """Vektörize edilmiş hızlı tahmin süreci."""
    total_samples = len(X_test)
    final_predictions = np.array(["benign"] * total_samples, dtype=object)

    # 1. AŞAMA: Binary Tahmin (Toplu İşlem)
    if binary_model_type == "ml":
        binary_preds = binary_model.predict(X_test)
    else:
        # DQN binary modeli için toplu tahmin
        binary_preds, _ = binary_model.predict(X_test.values, deterministic=True)

    # 2. AŞAMA: Filtreleme Mantığı
    # Koşul: Gerçekte Zararlı (1) VE Modelin Zararlı Dediği (1) -> Sadece bunlar DQN'e gider
    mask_to_dqn = (y_true_binary == 1) & (binary_preds == 1)
    indices_to_process = np.where(mask_to_dqn)[0]

    if len(indices_to_process) > 0:
        print(f"DQN katmanına {len(indices_to_process)} örnek gönderiliyor...")
        malicious_features = X_test.iloc[indices_to_process].values
        dqn_device = dqn_multiclass_model.device
        
        # Tek seferde büyük bir tensor oluşturup GPU/CPU'ya gönderiyoruz (Hızın anahtarı)
        obs_tensor = torch.tensor(malicious_features, dtype=torch.float32).to(dqn_device)

        with torch.no_grad():
            q_values = dqn_multiclass_model.q_net(obs_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Etiketleri haritalandır
        malicious_results = [multiclass_label_map.get(str(a), "unknown_malicious") for a in actions]
        final_predictions[indices_to_process] = malicious_results

    return final_predictions

def evaluate_and_plot_detailed(y_true, y_pred, title):
    """5 basamaklı hassasiyetle raporlama ve görselleştirme."""
    all_labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Standart Rapor
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    # Detaylı Sözlük Çıktısı
    report_dict = classification_report(y_true, y_pred, labels=all_labels, zero_division=0, output_dict=True)
    
    acc = report_dict["accuracy"] * 100
    pre = report_dict["weighted avg"]["precision"] * 100
    rec = report_dict["weighted avg"]["recall"] * 100
    f1  = report_dict["weighted avg"]["f1-score"] * 100

    print(f"--- DETAYLI İSTATİSTİKLER (%) ---")
    print(f"Doğruluk (Accuracy)  : {acc:.5f}%")
    print(f"Keskinlik (Precision): {pre:.5f}%")
    print(f"Duyarlılık (Recall)   : {rec:.5f}%")
    print(f"F1-Skoru (F1-Score)  : {f1:.5f}%")
    print(f"{'='*60}")

    # Karmaşıklık Matrisi
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels)
    plt.title(f"{title}\n(Detaylı Matris)")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.show()

def main():
    # Model seçimi için argüman kontrolü
    binary_choice = "rf" # Varsayılan
    if len(sys.argv) > 1:
        binary_choice = sys.argv[1].lower()

    model_paths = {
        "rf": ("binary-model-save/rf_binary_model.pkl", "ml"),
        "svm": ("binary-model-save/svm_binary_model.pkl", "ml"),
        "xgb": ("binary-model-save/xgb_binary_model.pkl", "ml"),
        "dqn": ("binary-model-save/binary_dqn_model", "dqn"),
    }

    if binary_choice not in model_paths:
        print(f"Hata: {binary_choice} geçerli değil. Seçenekler: rf, svm, xgb, dqn")
        return

    path, m_type = model_paths[binary_choice]

    try:
        # Modelleri Yükle
        dqn_multiclass_model = DQN.load(MULTICLASS_DQN_PATH)
        binary_model = joblib.load(path) if m_type == "ml" else DQN.load(path)
        
        with open(LABEL_MAP_PATH, "r") as f:
            multiclass_label_map = json.load(f)

        # Veriyi Yükle
        X_test, y_multi_test, y_bin_test = load_data_for_test(DATA_PATH)

        # Tahmin Süreci
        print(f"\nHibrit Pipeline başlatılıyor ({binary_choice.upper()})...")
        final_preds = run_hybrid_prediction_fast(
            X_test, y_bin_test, binary_model, dqn_multiclass_model, multiclass_label_map, m_type
        )

        # Değerlendirme
        evaluate_and_plot_detailed(
            y_multi_test, 
            final_preds, 
            f"Filtreli Hibrit Model Performansı (Binary: {binary_choice.upper()})"
        )

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()