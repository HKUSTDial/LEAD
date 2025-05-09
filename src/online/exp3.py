
import numpy as np
from math import exp, log

def categorical_draw(probs):
    return np.random.choice(len(probs), p=probs)

class EXP3S:
    def __init__(self, n_arms, gamma=0.1, c=2):
        self.n_arms = n_arms
        self.gamma = gamma
        self.c = c
        self.reward_history = []
        self.rewards = [0 for _ in range(n_arms)]
        self.weights = [1.0 for _ in range(n_arms)]
        self.counts = [0 for _ in range(n_arms)]

    def select_arm(self):
        total_weight = sum(self.weights)
        probs = [(1 - self.gamma) * (self.weights[i] / total_weight) + (self.gamma / self.n_arms) for i in range(self.n_arms)]
        return categorical_draw(probs)


    def normalize_reward(self, reward):
        if len(self.reward_history) < 2:
            return reward
        q_lo = min(self.reward_history)
        q_hi = max(self.reward_history)
        return 2 * (reward - q_lo) / (q_hi - q_lo) - 1

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.reward_history.append(reward)
        normalized_reward = self.normalize_reward(reward)
        self.rewards[chosen_arm] += normalized_reward

        total_weight = sum(self.weights)
        probs = [(1 - self.gamma) * (self.weights[i] / total_weight) + (self.gamma / self.n_arms) for i in range(self.n_arms)]
        x = normalized_reward / probs[chosen_arm]

        growth_factor = exp((self.gamma / self.n_arms) * x)
        self.weights[chosen_arm] *= growth_factor


    def get_average_reward(self):
        return np.sum(np.array(self.rewards)) / (np.sum(np.array(self.counts)) + 1e-10)

