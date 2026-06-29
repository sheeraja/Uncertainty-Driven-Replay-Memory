from collections import deque
import torch
import numpy as np

class PrioritizedReplayBuffer_Proportional:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.00000768, epsilon=1e-6):
        self._capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self._memory = deque(maxlen=capacity)
        self._priorities = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self._memory.append(transition)

        max_priority = max(self._priorities, default=1.0)
        self._priorities.append(max_priority)

    def sample(self, batch_size):
        total = self.__len__()
        
        self.increase_beta()
        priorities = np.array(self._priorities)

        # Convert priorities to probabilities (proportional sampling)
        # probabilities = priorities ** self.alpha
        probabilities = priorities / np.sum(priorities)

        idxs = np.random.choice(total, batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (total * probabilities) ** -self.beta
        weights /= np.max(weights)  # Normalize for stability
        states, actions, rewards, next_states, dones = zip(*[self._memory[idx] for idx in idxs])
        weights = [weights[idx] for idx in idxs]
        
        states = torch.stack(states, dim=0)
        rewards = torch.stack(rewards, dim=0)
        next_states = torch.stack(next_states, dim=0)
        dones = torch.stack(dones, dim=0)
        actions = torch.stack(actions, dim=0)

        return (states, actions, rewards, next_states, dones, idxs, weights)

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities.abs()):
            self._priorities[idx] = (abs(priority.item()) + self.epsilon) ** self.alpha

    def increase_beta(self):
        self.beta = np.min([1.0, self.beta + self.beta_increment])

    def is_full(self):
        return self._capacity == self.size

    def __len__(self):
        return len(self._memory)
