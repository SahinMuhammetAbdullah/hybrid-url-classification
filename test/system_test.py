import joblib
from stable_baselines3 import DQN
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Feature Importance için eklendi
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Sabitler ---
MALICIOUS_LABELS = ["phishing", "malware", "defacement", "spam"]
TARGET_COLUMN = "URL_Type_obf_Type"
DATA_PATH = "data/cleaned_feature_data.csv"
MULTICLASS_DQN_PATH = "multiclass-model-save/multiclass_dqn_model"
LABEL_MAP_PATH = "multiclass-model-save/multiclass_labels.json"


# --- 1. Veri Yükleme Fonksiyonu ---
def load_data_for_test(data_path):
    """Loads and splits data into test set."""
    df = pd.read_csv(data_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df["binary_label"] = df[TARGET_COLUMN].apply(
        lambda x: 0 if x.lower() in ["benign", "good"] else 1
    )

    X = df.drop(columns=[TARGET_COLUMN, "binary_label"])
    y = df[TARGET_COLUMN].str.lower().values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=df["binary_label"]
    )

    y_test_corrected = np.array(
        ["benign" if label == "good" else label for label in y_test]
    )

    print(f"Test data loaded. Total samples: {len(X_test)}")
    return X_test, y_test_corrected, X.columns.tolist()


# --- 2. Hibrit Tahmin Fonksiyonu ---
def run_hybrid_prediction(
    X_test, binary_model, dqn_multiclass_model, multiclass_label_map, binary_model_type
):
    # 1. Adım: Tüm veriyi tek seferde Binary Classifier'dan geçirin
    if binary_model_type == "ml":
        binary_preds = binary_model.predict(X_test)
    else:
        # DQN binary ise toplu tahmin (deterministic=True)
        binary_preds, _ = binary_model.predict(X_test.values, deterministic=True)

    # Sonuçları tutacak liste (varsayılan benign dolduralım)
    final_predictions = np.array(["benign"] * len(X_test), dtype=object)
    
    # 2. Adım: Malicious (1) olarak işaretlenenlerin indekslerini bulun
    malicious_indices = np.where(binary_preds == 1)[0]
    
    if len(malicious_indices) > 0:
        # Sadece malicious olan satırları seçin
        malicious_features = X_test.iloc[malicious_indices].values
        
        # 3. Adım: DQN Multiclass tahminini toplu (Batch) yapın
        dqn_device = dqn_multiclass_model.device
        obs_tensor = torch.tensor(malicious_features, dtype=torch.float32).to(dqn_device)
        
        with torch.no_grad():
            q_values = dqn_multiclass_model.q_net(obs_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        
        # Etiketleri haritalayın
        malicious_labels = [multiclass_label_map.get(str(a), "unknown_malicious") for a in actions]
        final_predictions[malicious_indices] = malicious_labels

    return final_predictions.tolist()

# --- 3. Değerlendirme Fonksiyonu ---
def evaluate_and_plot(y_true, y_pred, title):
    """Reports results and plots the confusion matrix."""
    print(f"\n--- {title} ---")
    all_labels = sorted(list(set(y_true) | set(y_pred)))

    report_dict = classification_report(
        y_true, y_pred, labels=all_labels, zero_division=0, output_dict=True
    )
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    weighted_avg = report_dict["weighted avg"]
    weighted_precision = weighted_avg["precision"] * 100
    weighted_recall = weighted_avg["recall"] * 100
    weighted_f1 = weighted_avg["f1-score"] * 100
    accuracy = report_dict["accuracy"] * 100

    print("\n--- Weighted Average (Percentage Format) ---")
    print(f"Precision : {weighted_precision:.6f}%")
    print(f"Recall    : {weighted_recall:.6f}%")
    print(f"F1-Score  : {weighted_f1:.6f}%")
    print(f"Accuracy  : {accuracy:.6f}%")

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
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()


# --- 4. Ana Akış Fonksiyonu ---
def main():
    binary_choice = "rf"  # Argüman gerekmiyor, doğrudan Random Forest kullanılacak

    model_paths = {
        "rf": ("binary-model-save/rf_binary_model.pkl", "ml"),
        "svm": ("binary-model-save/svm_binary_model.pkl", "ml"),
        "xgb": ("binary-model-save/xgb_binary_model.pkl", "ml"),
        "dqn": ("binary-model-save/binary_dqn_model", "dqn"),
    }

    binary_model_path, binary_model_type = model_paths[binary_choice]

    print("Loading models and data...")
    try:
        dqn_multiclass_model = DQN.load(MULTICLASS_DQN_PATH)
        binary_model = joblib.load(binary_model_path)

        with open(LABEL_MAP_PATH, "r") as f:
            multiclass_label_map = json.load(f)

        X_test, y_test_corrected, _ = load_data_for_test(DATA_PATH)
        print("All components loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e.filename}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return

    print("\nRunning hybrid pipeline with 'RF' as binary classifier...")
    final_predictions = run_hybrid_prediction(
        X_test,
        binary_model,
        dqn_multiclass_model,
        multiclass_label_map,
        binary_model_type,
    )

    evaluate_and_plot(
        y_test_corrected,
        final_predictions,
        "End-to-End Performance (Binary: RF)",
    )


if __name__ == "__main__":
    main()
