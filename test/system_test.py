import argparse
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
import os
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

    # Splitting data for test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=df["binary_label"]
    )

    y_test_corrected = np.array(
        ["benign" if label == "good" else label for label in y_test]
    )

    print(f"Test data loaded. Total samples: {len(X_test)}")
    return X_test, y_test_corrected, X.columns.tolist()


# --- 2. Hibrit Tahmin Fonksiyonu (End-to-End) ---
def run_hybrid_prediction(
    X_test, binary_model, dqn_multiclass_model, multiclass_label_map, binary_model_type
):
    """Runs the two-tiered prediction pipeline (end-to-end)."""
    final_predictions = []
    dqn_device = dqn_multiclass_model.device

    for i in range(len(X_test)):
        # Layer 1 Prediction
        if binary_model_type == "ml":
            feature_row_df = X_test.iloc[[i]]
            binary_pred = binary_model.predict(feature_row_df)[0]
        else:
            feature_values_np = X_test.iloc[i].values
            # DQN returns action (0 or 1) as the first element
            binary_pred, _ = binary_model.predict(feature_values_np, deterministic=True)

        if binary_pred == 0:
            final_predictions.append("benign")
        else:
            # Layer 2 Prediction (DQN)
            feature_values_np = X_test.iloc[i].values
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

    return final_predictions


# --- 3. Değerlendirme ve Çizim Fonksiyonu (İngilizce) ---
def evaluate_and_plot(y_true, y_pred, title):
    """Reports results and plots the confusion matrix (English)."""
    print(f"\n--- {title} ---")
    all_labels = sorted(list(set(y_true) | set(y_pred)))

    # classification_report outputs and dictionary
    report_dict = classification_report(
        y_true, y_pred, labels=all_labels, zero_division=0, output_dict=True
    )
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    # Calculate weighted averages
    weighted_avg = report_dict["weighted avg"]
    weighted_precision = weighted_avg["precision"] * 100
    weighted_recall = weighted_avg["recall"] * 100
    weighted_f1 = weighted_avg["f1-score"] * 100
    accuracy = report_dict["accuracy"] * 100

    # Print results
    print("\n--- Weighted Average (Percentage Format) ---")
    print(f"Precision : {weighted_precision:.6f}%")
    print(f"Recall    : {weighted_recall:.6f}%")
    print(f"F1-Score  : {weighted_f1:.6f}%")
    print(f"Accuracy  : {accuracy:.6f}%")

    # Confusion Matrix Visualization (English)
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


# --- 4. Öznitelik Önem Karşılaştırma Fonksiyonu (İngilizce) ---
def plot_feature_importances_comparison(X_data, y_multi_full, binary_model_choice):
    """
    Compares feature importances of Binary Model and Multi-class RF Proxy (for DQN).
    X-axis labels show the RANK of the feature in each model.
    """
    print("\n--- Feature Importance Analysis Started ---")
    
    X_data_num = X_data.select_dtypes(include=[np.number])
    feature_names = X_data_num.columns
    y_bin_full = y_multi_full.apply(lambda x: 0 if x == "benign" else 1)

    # Label definitions
    binary_label_key = 'Binary Importance'
    multi_label_key = 'Multiclass Importance'

    # --- 1. Binary Model Importance (RF/XGB or RF Proxy) ---
    # ... (Model yükleme ve önem hesaplama mantığı aynı kalır) ...
    try:
        if binary_model_choice == "dqn":
            print("Using RF Proxy for Binary Model Importance.")
            binary_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            binary_model.fit(X_data_num, y_bin_full)
            binary_importances = binary_model.feature_importances_
        else:
            model_path = f"binary-model-save/{binary_model_choice}_binary_model.pkl"
            try:
                binary_model = joblib.load(model_path)
                binary_importances = binary_model.feature_importances_
            except (FileNotFoundError, AttributeError):
                print(f"Warning: Could not load {binary_model_choice}. Falling back to RF Proxy.")
                binary_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                binary_model.fit(X_data_num, y_bin_full)
                binary_importances = binary_model.feature_importances_
    except Exception as e:
        print(f"Error during Binary Model Importance calculation: {e}. Aborting.")
        return

    # --- 2. Multiclass Model Importance (RF Proxy for DQN) ---
    print("Using RF Proxy for Multi-class Model Importance.")
    malicious_df = pd.DataFrame(
        {'features': X_data_num.values.tolist(), 'label': y_multi_full.values}
    ).set_index(X_data_num.index)
    malicious_df = malicious_df[malicious_df['label'].isin(MALICIOUS_LABELS)].copy()
    X_multi = pd.DataFrame(malicious_df['features'].tolist(), columns=feature_names)
    y_multi = malicious_df['label']

    multi_class_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    multi_class_rf.fit(X_multi, y_multi)
    multi_class_importances = multi_class_rf.feature_importances_

    # --- 3. Create Final DataFrame and Calculate Ranks ---
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        binary_label_key: binary_importances,
        multi_label_key: multi_class_importances
    })
    
    # RANK hesaplama: En yüksek değere (1. sıraya) en düşük sıra numarasını atar
    importances_df['Binary_Rank'] = importances_df[binary_label_key].rank(ascending=False, method='min').astype(int)
    importances_df['Multi_Rank'] = importances_df[multi_label_key].rank(ascending=False, method='min').astype(int)
    
    # Ortalama öneme göre sırala (Grafik sırası)
    importances_df['Average Importance'] = (importances_df[binary_label_key] + importances_df[multi_label_key]) / 2
    importances_df = importances_df.sort_values(by='Average Importance', ascending=False).reset_index(drop=True)
    
    # Yeni X ekseni etiketlerini oluştur: <FeatureName, BinaryRank, MulticlassRank>
    importances_df['Plot_Label'] = importances_df.apply(
        lambda row: f"<{row['Feature']}, {row['Binary_Rank']}, {row['Multi_Rank']}>", axis=1
    )

    # --- 4. Output in desired text format (Rank) ---
    print("\n--- Feature Importances in Text Format (Sorted by Average Importance) ---")
    print("Feature/Binary Rank/Multiclass Rank")
    
    for index, row in importances_df.iterrows():
        print(f"{row['Feature']}/{row['Binary_Rank']}/{row['Multi_Rank']}")
    
    print("--- End of Text Output ---")

    # --- 5. Visualization (Sorted by Importance - English) ---
    
    binary_plot_label = f'Binary Model ({binary_model_choice.upper()} Proxy)' if binary_model_choice == "dqn" else f'Binary Model ({binary_model_choice.upper()})'
    multi_plot_label = 'Multi-class Model (RF Proxy for DQN)'

    N = len(feature_names)
    ind = np.arange(N)
    width = 0.35

    plt.figure(figsize=(30, 12))
    plt.bar(ind - width/2, importances_df[binary_label_key], width, label=binary_plot_label)
    plt.bar(ind + width/2, importances_df[multi_label_key], width, label=multi_plot_label, color='orangered')

    plt.ylabel('Feature Importance', fontsize=16)
    plt.title('Feature Importance Comparison (Sorted by Average Importance)', fontsize=18)
    
    # YENİ ETİKETLEME: Sıra numaralarını içeren Plot_Label kullanılır
    plt.xticks(ind, importances_df['Plot_Label'], rotation=90, fontsize=10) 
    
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, importances_df[[binary_label_key, multi_label_key]].max().max() * 1.05)
    plt.tight_layout()
    plt.show()
    print("--- Feature Importance Plot Generated ---")

# --- 5. Ana Akış Fonksiyonu ---
def main(binary_choice):
    model_paths = {
        "rf": ("binary-model-save/rf_binary_model.pkl", "ml"),
        "svm": ("binary-model-save/svm_binary_model.pkl", "ml"),
        "xgb": ("binary-model-save/xgb_binary_model.pkl", "ml"),
        "dqn": ("binary-model-save/binary_dqn_model", "dqn"),
    }

    if binary_choice not in model_paths:
        raise ValueError(
            f"Invalid choice '{binary_choice}'. Must be one of {list(model_paths.keys())}"
        )

    binary_model_path, binary_model_type = model_paths[binary_choice]

    print("Loading models and data...")
    try:
        dqn_multiclass_model = DQN.load(MULTICLASS_DQN_PATH)
        if binary_model_type == "ml":
            binary_model = joblib.load(binary_model_path)
        else:
            binary_model = DQN.load(binary_model_path)

        with open(LABEL_MAP_PATH, "r") as f:
            multiclass_label_map = json.load(f)

        X_test, y_test_corrected, feature_cols = load_data_for_test(DATA_PATH)
        print("All components loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e.filename}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return

    # 1. Run End-to-End Prediction and Evaluate
    print(
        f"\nRunning hybrid pipeline with '{binary_choice.upper()}' as binary classifier..."
    )
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
        f"End-to-End Performance (Binary: {binary_choice.upper()})",
    )
    
    # 2. Run Feature Importance Comparison
    # Load full dataset for more robust importance calculation
    df_full = pd.read_csv(DATA_PATH)
    df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_full.dropna(inplace=True)
    
    X_full = df_full.drop(columns=[TARGET_COLUMN]).select_dtypes(include=[np.number])
    y_multi_full = df_full[TARGET_COLUMN].str.lower().replace("good", "benign")

    plot_feature_importances_comparison(
        X_full, 
        y_multi_full, 
        binary_choice
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the hybrid URL classification system and compare feature importances."
    )
    parser.add_argument(
        "--binary_model",
        type=str,
        required=True,
        choices=["rf", "svm", "xgb", "dqn"],
        help="Choose the binary classifier for the first stage: 'rf', 'svm', 'xgb', or 'dqn'.",
    )
    args = parser.parse_args()

    main(args.binary_model)