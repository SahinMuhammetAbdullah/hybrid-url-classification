import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env_url_type import URLTypeClassificationEnv

# --- 1. Adım: Veri Setini Yükle ---
data_path = "data/input_data.csv"
df = pd.read_csv(data_path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"
df['binary_label'] = df[target_column].apply(lambda x: 0 if x in ["benign", "good"] else 1)
df_malicious = df[df['binary_label'] == 1]
df_malicious = df_malicious[~df_malicious[target_column].isin(["malicious", "spam"])]

features = df_malicious.drop(columns=[target_column, "binary_label"]).values
labels = df_malicious[target_column].astype("category").cat.codes.values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# --- 2. Adım: Class Ağırlıkları Hesapla ---
counts = Counter(y_train)
class_weights = {cls: max(1.0, 1000 / count) for cls, count in counts.items()}

# --- 3. Adım: Ortamı Tanımla ---
env = DummyVecEnv([lambda: Monitor(URLTypeClassificationEnv(X_train, y_train, class_weights=class_weights))])

# --- 4. Adım: DQN Modeli Oluştur ---
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    batch_size=256,
    learning_rate=0.0003,
    gamma=0.99
)

# --- 5. Adım: Değerlendirme Fonksiyonu ---
def evaluate_model(model, X_test, y_test, target_names):
    test_env = URLTypeClassificationEnv(X_test, y_test, class_weights=class_weights)
    obs, _ = test_env.reset()
    predictions, true_labels = [], []

    for _ in range(len(X_test)):
        action, _ = model.predict(obs, deterministic=True)
        predictions.append(action)
        true_labels.append(y_test[test_env.current_index])
        obs, _, done, _, _ = test_env.step(action)
        if done:
            break

    print("\n--- Test Sonuçları ---\n")
    print(classification_report(true_labels, predictions, target_names=target_names))

# --- 6. Adım: Eğitim Döngüsü ---
total_timesteps = 2_000_000
checkpoint_intervals = np.linspace(0, total_timesteps, 11, dtype=int)[1:]
current_steps = 0
url_types = df_malicious[target_column].unique()

for checkpoint in checkpoint_intervals:
    train_steps = checkpoint - current_steps
    model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
    print(f"\n--- {checkpoint} timestep sonrası değerlendirme ---")
    evaluate_model(model, X_test, y_test, target_names=[str(i) for i in url_types])
    current_steps = checkpoint

# --- 7. Adım: Model Kaydet ---
model.save("multiclass_dqn_model")

# --- 8. Adım: Son Test ---
print("\n--- Eğitim Tamamlandı: Son Test ---")
model = DQN.load("multiclass_dqn_model")
evaluate_model(model, X_test, y_test, target_names=[str(i) for i in url_types])
