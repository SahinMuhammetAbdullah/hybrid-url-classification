import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ----- 1. Adım: Veri Setini Hazırla -----
data_path = "data/input_dataV3.csv"
df = pd.read_csv(data_path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

target_column = "URL_Type_obf_Type"
df['binary_label'] = df[target_column].apply(lambda x: 0 if x in ["benign", "good"] else 1)

features = df.drop(columns=[target_column, "binary_label"]).values
labels = df['binary_label'].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# ----- 2. Adım: Gymnasium Ortamını Tanımla -----
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.metrics import precision_score, recall_score, f1_score

class URLBinaryClassificationEnv(gym.Env):
    def __init__(self, features, labels, class_weights=None):
        super(URLBinaryClassificationEnv, self).__init__()
        self.features = features
        self.labels = labels
        self.current_index = 0
        self.action_space = spaces.Discrete(2)  # Binary: 0 veya 1
        self.observation_space = spaces.Box(
            low=np.min(self.features),
            high=np.max(self.features),
            shape=(self.features.shape[1],),
            dtype=np.float32
        )

        if class_weights is None:
            unique_classes, counts = np.unique(labels, return_counts=True)
            class_weights = {cls: max(1.0, 1000 / count) for cls, count in zip(unique_classes, counts)}
        self.class_weights = class_weights

        self.predictions = []
        self.true_labels_log = []

    def reset(self, seed=None, options=None):
        self.current_index = np.random.randint(0, len(self.features))
        self.predictions = []
        self.true_labels_log = []
        return self.features[self.current_index], {}

    def step(self, action):
        true_label = self.labels[self.current_index]
        self.predictions.append(action)
        self.true_labels_log.append(true_label)

        # Anlık ödül: doğruysa sınıf ağırlığı kadar, yanlışsa sabit ceza
        if action == true_label:
            reward = 10 * self.class_weights[true_label]
        else:
            reward = -5

        # Sıradaki örneğe geç
        self.current_index = (self.current_index + 1) % len(self.features)
        done = self.current_index == 0

        # Epoch sonunda toplu metrik bazlı ödül
        if done and len(self.predictions) > 0:
            reward = self.calculate_multimetric_reward()
            self.predictions = []
            self.true_labels_log = []

        return self.features[self.current_index], reward, done, False, {}

    def calculate_multimetric_reward(self):
        if len(set(self.true_labels_log)) > 1:  # F1 anlamlı olsun
            precision = precision_score(self.true_labels_log, self.predictions, average='binary', zero_division=0)
            recall = recall_score(self.true_labels_log, self.predictions, average='binary', zero_division=0)
            f1 = f1_score(self.true_labels_log, self.predictions, average='binary', zero_division=0)
            reward = (precision + recall + f1) * 30  # Aynı mantıkta çarpan
        else:
            reward = -10  # sınıf çeşitliliği yoksa ceza
        return reward

# ----- 3. Adım: Modeli Eğit ve Ara Raporlama -----
total_timesteps = 2_000_000
checkpoint_intervals = np.linspace(0, total_timesteps, 11, dtype=int)[1:]

env = DummyVecEnv([lambda: Monitor(URLBinaryClassificationEnv(X_train, y_train))])
model = DQN("MlpPolicy", env, verbose=0, device="cpu", batch_size=256, learning_rate=0.0003, gamma=0.99)

def evaluate_model(model, X_test, y_test):
    test_env = URLBinaryClassificationEnv(X_test, y_test)
    test_env.current_index = 0
    obs = test_env.features[test_env.current_index]
    predictions, true_labels = [], []

    for _ in range(len(X_test)):
        action, _ = model.predict(obs)
        predictions.append(action)
        true_labels.append(y_test[test_env.current_index])
        obs, _, done, _, _ = test_env.step(action)
        if done:
            break

    print(classification_report(true_labels, predictions, target_names=["Benign (0)", "Malicious (1)"]))

current_steps = 0
for checkpoint in checkpoint_intervals:
    train_steps = checkpoint - current_steps
    model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
    print(f"\n--- {checkpoint} timestep sonrası değerlendirme ---")
    evaluate_model(model, X_test, y_test)
    current_steps = checkpoint

model.save("binary-pkl/binary_dqn_modelV3")
