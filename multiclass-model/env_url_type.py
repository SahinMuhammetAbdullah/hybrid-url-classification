import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.metrics import precision_score, recall_score, f1_score

class URLTypeClassificationEnv(gym.Env):
    def __init__(self, features, labels, class_weights=None):
        super(URLTypeClassificationEnv, self).__init__()
        self.features = features
        self.labels = labels
        self.current_index = 0
        self.action_space = spaces.Discrete(len(np.unique(labels)))
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

        # Ajanın tahmin ve etiket geçmişini kaydetmek için
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

        # Anlık ödül hesaplama: doğruysa sınıf ağırlığı kadar, yanlışsa sabit ceza
        if action == true_label:
            reward = 10 * self.class_weights[true_label]
        else:
            reward = -5

        # Bir sonraki örneğe geç
        self.current_index = (self.current_index + 1) % len(self.features)
        done = self.current_index == 0

        # Eğer tüm veri dönmüşse F1, Precision, Recall bazlı genel ödül
        if done and len(self.predictions) > 0:
            reward = self.calculate_multimetric_reward()
            self.predictions = []
            self.true_labels_log = []

        return self.features[self.current_index], reward, done, False, {}

    def calculate_multimetric_reward(self):
        if len(set(self.true_labels_log)) > 1:  # F1 anlamlı olsun
            precision = precision_score(self.true_labels_log, self.predictions, average='macro', zero_division=0)
            recall = recall_score(self.true_labels_log, self.predictions, average='macro', zero_division=0)
            f1 = f1_score(self.true_labels_log, self.predictions, average='macro', zero_division=0)
            reward = (precision + recall + f1) * 30  # daha etkili katkı için 30 çarpanı
        else:
            reward = -10  # sınıf çeşitliliği yoksa cezalandır
        return reward
