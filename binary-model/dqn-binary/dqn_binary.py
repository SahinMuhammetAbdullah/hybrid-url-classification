import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from env_url_type import URLTypeClassificationEnv

# Veri yükleme ve temizleme
data_path = "data/cleaned_feature_data.csv"
df = pd.read_csv(data_path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Binary etiket oluşturma: 0 -> benign/good, 1 -> malicious
target_column = "URL_Type_obf_Type"
df["binary_label"] = df[target_column].apply(lambda x: 0 if x in ["benign", "good"] else 1)

# Özellik ve etiketler
features = df.drop(columns=[target_column, "binary_label"]).values
labels = df["binary_label"].values

# Eğitim-test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, stratify=labels, random_state=42
)

# RL ortamı
env = DummyVecEnv([lambda: Monitor(URLTypeClassificationEnv(X_train, y_train))])

# DQN modeli
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    batch_size=256,
    learning_rate=0.0003,
    gamma=0.99,
)

# Değerlendirme fonksiyonu
def evaluate_model(model, X_test, y_test):
    test_env = URLTypeClassificationEnv(X_test, y_test)

    obs, _ = test_env.reset()
    predictions, true_labels = [], []

    for _ in range(len(X_test)):
        action, _ = model.predict(obs, deterministic=True)
        predictions.append(action)
        idx = test_env.shuffled_indices[test_env.current_step]
        true_labels.append(y_test[idx])
        obs, _, done, _, _ = test_env.step(action)
        if done:
            obs, _ = test_env.reset()

    print("\n--- Test Sonuçları ---\n")
    print(classification_report(true_labels, predictions, target_names=["benign", "malicious"]))

# Eğitim döngüsü
total_timesteps = 2_000_000
checkpoint_intervals = np.linspace(0, total_timesteps, 11, dtype=int)[1:]
current_steps = 0

for checkpoint in checkpoint_intervals:
    train_steps = checkpoint - current_steps
    model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
    print(f"\n--- {checkpoint} timestep sonrası değerlendirme ---")
    evaluate_model(model, X_test, y_test)
    current_steps = checkpoint

# Modeli kaydet
model.save("binary_dqn_model")

print("\n--- Eğitim Tamamlandı: Son Test ---")

# Torch kullanarak final tahmin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN.load("binary_dqn_model", device=device)

y_pred = []
for obs in X_test:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
        action = torch.argmax(q_values, dim=1).item()
        y_pred.append(action)

print(classification_report(y_test, y_pred, target_names=["benign", "malicious"]))
