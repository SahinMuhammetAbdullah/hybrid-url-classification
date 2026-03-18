import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import matplotlib.pyplot as plt

from env_url_type import URLTypeClassificationEnv

# === Veri hazırlığı ===
data_path = "data/cleaned_feature_data.csv"
df = pd.read_csv(data_path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"
df["binary_label"] = df[target_column].apply(
    lambda x: 0 if x in ["benign", "good"] else 1
)
df_malicious = df[df["binary_label"] == 1]
df_malicious = df_malicious[~df_malicious[target_column].isin(["malicious"])]

features = df_malicious.drop(columns=[target_column, "binary_label"]).values
labels = df_malicious[target_column].astype("category").cat.codes.values

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, stratify=labels, random_state=42
)

# === Ortam ve model ===
env = DummyVecEnv([lambda: Monitor(URLTypeClassificationEnv(X_train, y_train))])

policy_kwargs = dict(net_arch=[128, 128, 64])  # Ağ derinliğini artırma (F1 için kritik)

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    device="auto",
    learning_rate=0.0001,
    gamma=0.99,
    batch_size=128,
    target_update_interval=20000,
    exploration_fraction=0.2,
    exploration_final_eps=0.005,
    policy_kwargs=policy_kwargs,
)


# === Callback sınıfı ===
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.losses = []
        self.current_reward = 0

    def _on_step(self):
        rewards = self.locals.get("rewards", [0])
        self.current_reward += np.mean(rewards)
        if "loss" in self.model.logger.name_to_value:
            self.losses.append(self.model.logger.name_to_value["loss"])
        return True

    def _on_rollout_end(self):
        self.episode_rewards.append(self.current_reward)
        self.current_reward = 0


# === Değerlendirme fonksiyonu ===
def evaluate_model(model, X_test, y_test, target_names):
    unique_classes = np.unique(y_test)
    equal_weights = {cls: 1.0 for cls in unique_classes}
    test_env = URLTypeClassificationEnv(X_test, y_test, class_weights=equal_weights)
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
    print(classification_report(true_labels, predictions, target_names=target_names))

    return (
        precision_score(true_labels, predictions, average="macro", zero_division=0),
        recall_score(true_labels, predictions, average="macro", zero_division=0),
        f1_score(true_labels, predictions, average="macro", zero_division=0),
    )


# === Eğitim parametreleri ===
total_timesteps = 2_000_000
url_types = df_malicious[target_column].unique()

callback = CustomCallback()
precision_list, recall_list, f1_list = [], [], []

# === Checkpoint tabanlı eğitim ve değerlendirme ===
checkpoint_intervals = [int(i) for i in np.linspace(200_000, total_timesteps, 5)]
current_steps = 0

for checkpoint in checkpoint_intervals:
    train_steps = checkpoint - current_steps
    model.learn(
        total_timesteps=train_steps, reset_num_timesteps=False, callback=callback
    )
    print(f"\n--- {checkpoint} timestep sonrası değerlendirme ---")
    p, r, f = evaluate_model(
        model, X_test, y_test, target_names=[str(i) for i in url_types]
    )
    precision_list.append(p)
    recall_list.append(r)
    f1_list.append(f)
    current_steps = checkpoint

model.save("multiclass-model-save/multiclass_dqn_model")

print("\n--- Eğitim Tamamlandı: Son Test ---")

# === Ortalama Q-değerlerini ölç ===
avg_q_values = []
for _ in range(200):
    obs = env.reset()
    episode_q_values = []
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_values = model.q_net(torch.tensor(obs, dtype=torch.float32).to(device))
        episode_q_values.append(q_values.mean().item())
        obs, rewards, dones, infos = env.step(action)
        if dones:
            break
    avg_q_values.append(np.mean(episode_q_values))

# === GRAFİKLER ===
plt.figure(figsize=(12, 5))
plt.plot(callback.episode_rewards, label="Training Reward", color="tab:blue")
plt.title("Training Reward over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(callback.losses, label="Loss", color="tab:orange")
plt.title("DQN Loss over Time")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(avg_q_values, label="Average Q-values", color="tab:green")
plt.title("Average Q-Value over Episodes")
plt.xlabel("Episode")
plt.ylabel("Q-Value")
plt.legend()
plt.grid(True)
plt.show()

if len(precision_list) > 0:
    plt.figure(figsize=(12, 5))
    plt.plot(precision_list, label="Precision", color="tab:red", marker="o")
    plt.plot(recall_list, label="Recall", color="tab:purple", marker="s")
    plt.plot(f1_list, label="F1 Score", color="tab:cyan", marker="^")
    plt.title("Model Performance Metrics over Checkpoints")
    plt.xlabel("Checkpoint (Training Steps)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(
        "Not: Precision/Recall/F1 grafiği çizilmedi çünkü değerlendirme döngüsü devre dışıydı."
    )

print("\n--- Tüm grafikler oluşturuldu. Eğitim analizi tamamlandı. ---")
