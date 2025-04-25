from stable_baselines3 import DQN
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.model_selection import train_test_split

# Modelleri yükle
dqn_binary_model = DQN.load("binary-pkl/binary_dqn_model")
dqn_model = DQN.load("multiclass-pkl/multiclass_dqn_model")

# Veriyi yükle
df = pd.read_csv("data/input_data.csv")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Etiketleri ikili sınıflandırmaya göre çevir
target_column = "URL_Type_obf_Type"
df['binary_label'] = df[target_column].str.lower().apply(lambda x: 0 if x in ["benign", "good"] else 1)

# Özellikler ve etiketler
X = df.drop(columns=[target_column, "binary_label"])
y = df[target_column].str.lower().values  # küçük harfe normalize

# Eğitim ve test veri bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

final_predictions = []

# Tahmin döngüsü
for index, row in X_test.iterrows():
    feature = pd.DataFrame([row], columns=X_test.columns)
    rf_prediction = dqn_binary_model.predict(feature)[0]

    if rf_prediction == 0:
        final_predictions.append("benign")
    else:
        obs = feature.values.flatten()
        action, _ = dqn_model.predict(obs)
        label_map = {0: "defacement", 1: "malware", 2: "phishing"}
        predicted_label = label_map.get(int(action), "unknown")
        final_predictions.append(predicted_label)

# Etiket düzeltme: 'good' -> 'benign'
y_test_corrected = ["benign" if label == "good" else label for label in y_test]

# Malicious etiketli kayıtları filtrele
filtered_predictions = []
filtered_true_labels = []

for pred, true_label in zip(final_predictions, y_test_corrected):
    if true_label not in ["malicious", "spam"]:  # 'malicious' ve 'spam' etiketlerini dışarıda bırak
        filtered_predictions.append(pred)
        filtered_true_labels.append(true_label)

# Sonuçları yazdır
print("\nTahmin Edilen Sonuçlar:", Counter(filtered_predictions))
print("Gerçek Etiketler:", Counter(filtered_true_labels))

# Sınıflandırma raporu
print("\nSınıflandırma Raporu:\n")
print(classification_report(filtered_true_labels, filtered_predictions))
