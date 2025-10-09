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


def load_data_for_test(data_path):
    """Veriyi test için yükler ve böler."""
    df = pd.read_csv(data_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    target_column = "URL_Type_obf_Type"
    df["binary_label"] = df[target_column].apply(
        lambda x: 0 if x.lower() in ["benign", "good"] else 1
    )

    X = df.drop(columns=[target_column, "binary_label"])
    y = df[target_column].str.lower().values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=df["binary_label"]
    )

    y_test_corrected = np.array(
        ["benign" if label == "good" else label for label in y_test]
    )

    print(f"Test data loaded. Total samples: {len(X_test)}")
    return X_test, y_test_corrected


def run_hybrid_prediction(
    X_test, binary_model, dqn_multiclass_model, multiclass_label_map, binary_model_type
):
    """İki aşamalı tahmin boru hattını çalıştırır."""
    final_predictions = []
    dqn_device = dqn_multiclass_model.device

    for i in range(len(X_test)):
        if binary_model_type == "ml":
            feature_row_df = X_test.iloc[[i]]
            binary_pred = binary_model.predict(feature_row_df)[0]
        else:
            feature_values_np = X_test.iloc[i].values
            binary_pred, _ = binary_model.predict(feature_values_np, deterministic=True)

        if binary_pred == 0:
            final_predictions.append("benign")
        else:
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


def evaluate_and_plot(y_true, y_pred, title):
    """Sonuçları raporlar ve karmaşıklık matrisini çizer."""
    print(f"\n--- {title} ---")
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
    print(f"Precision : {weighted_precision:.6f}%")
    print(f"Recall    : {weighted_recall:.6f}%")
    print(f"F1-Score  : {weighted_f1:.6f}%")
    print(f"Accuracy  : {accuracy:.6f}%")

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

        X_test, y_test_corrected = load_data_for_test(DATA_PATH)
        print("All components loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e.filename}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the hybrid URL classification system."
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
