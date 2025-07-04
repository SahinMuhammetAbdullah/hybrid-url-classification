import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.metrics import precision_score, recall_score, f1_score

class URLTypeClassificationEnv(gym.Env):
    def __init__(self, features, labels, class_weights=None, gamma=0.99):
        super(URLTypeClassificationEnv, self).__init__()
        self.features = features
        self.labels = labels
        self.gamma = gamma
        self.action_space = spaces.Discrete(len(np.unique(labels)))
        self.observation_space = spaces.Box(
            low=np.min(self.features),
            high=np.max(self.features),
            shape=(self.features.shape[1],),
            dtype=np.float32
        )
        self.class_weights = class_weights or self.compute_class_weights(labels)
        self.reset_indices()

        self.predictions = []
        self.true_labels_log = []
        self.last_action_probs = None

    def reset_indices(self):
        indices = np.arange(len(self.features))
        np.random.shuffle(indices)
        self.shuffled_indices = indices
        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.reset_indices()
        self.predictions = []
        self.true_labels_log = []
        self.last_action_probs = None
        self.transition_log = []
        return self.features[self.shuffled_indices[self.current_step]], {}

    def set_action_probs(self, probs):
        self.last_action_probs = probs

    def step(self, action):
        idx = self.shuffled_indices[self.current_step]
        true_label = self.labels[idx]

        self.predictions.append(action)
        self.true_labels_log.append(true_label)

        if action == true_label:
            reward = self.class_weights[true_label]
        else:
            reward = -(1 / self.class_weights[true_label])

        next_step = (self.current_step + 1) % len(self.features)
        next_idx = self.shuffled_indices[next_step]

        self.transition_log.append({
            "state": self.features[idx].tolist(),
            "action": int(action),
            "reward": float(reward),
            "next_state": self.features[next_idx].tolist(),
            "done": next_step == 0
        })

        self.current_step = next_step
        done = self.current_step == 0

        if done:
            reward += self.calculate_multimetric_reward()
            self.predictions = []
            self.true_labels_log = []

        return self.features[self.shuffled_indices[self.current_step]], reward, done, False, {}

    def compute_class_weights(self, labels, min_weight=0, max_weight=None, normalize=True):
        unique_classes, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        weights = {}

        for cls, count in zip(unique_classes, counts):
            weight = total_samples / (len(unique_classes) * count)
            weight = max(min_weight, weight)
            if max_weight is not None:
                weight = min(max_weight, weight)
            weights[cls] = weight

        if normalize:
            total_weight = sum(weights.values())
            for cls in weights:
                weights[cls] /= total_weight

        return weights

    def calculate_multimetric_reward(self):
        if len(set(self.true_labels_log)) > 1:
            precision = precision_score(self.true_labels_log, self.predictions, average='macro', zero_division=0)
            recall = recall_score(self.true_labels_log, self.predictions, average='macro', zero_division=0)
            f1 = f1_score(self.true_labels_log, self.predictions, average='macro', zero_division=0)
            return (0.3 * precision + 0.2 * recall + 0.5 * f1) * 1.5
        else:
            return -1.5
