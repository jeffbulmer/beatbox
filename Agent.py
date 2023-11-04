import numpy as np
import pickle
from collections import defaultdict
import random
class Agent:
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = (min_exploration / exploration_rate) ** (1 / 50000000)
        self.min_exploration = min_exploration
        self.action_space_size = action_space_size
        self._initialize_q_table()

    def _default_q_values(self):
        """Return default Q-values."""
        return [0] * self.action_space_size

    def _initialize_q_table(self):
        """Initialize the Q-table with zeros."""
        self.q_table = defaultdict(self._default_q_values)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load(self, filename):
        """ Load Q-values from a file. """
        try:
            with open(filename, 'rb') as file:
                self.q_table = pickle.load(file)
        except Exception as e:
            self._initialize_q_table()

    def set_exploration_rate(self, rate):
        self.exploration_rate = rate

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def get_action(self, state):
        # Explore with probability exploration_rate, otherwise exploit the best action
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.randint(0, self.action_space_size)
        else:
            # Get Q values for all actions for the given state
            q_values = [self.q_table.get((state, a), 0) for a in range(self.action_space_size)]
            # Find the maximum Q value
            max_q_value = max(q_values)
            # Find the actions that have the maximum Q value
            max_actions = [action for action, q_value in enumerate(q_values) if q_value == max_q_value]
            # Return a random action among those with the maximum Q value
            return random.choice(max_actions)

    def update(self, state, action, reward, next_state):
        # Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the next state
        old_value = self.q_table.get((state, action), 0)
        next_max = max([self.q_table.get((next_state, a), 0) for a in range(self.action_space_size)])

        # Q-learning update rule
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[(state, action)] = new_value

        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
