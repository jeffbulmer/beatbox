import math
from gym import spaces
from Environment import Environment
import numpy as np
import random
from utils import (
    discretize_distance,
    discretize_angle,
    angle_between_points
)


class BunnyEnvironment(Environment):
    def __init__(self, grid_size=20):
        super(BunnyEnvironment, self).__init__(grid_size)
        self.num_angle_bins = 32
        self.distance_bins = [1, 2, 3, 5, 10, 20, 40]
        self.action_space = spaces.Discrete(9)  # Only bunny actions
        self.observation_space = spaces.Tuple([
            spaces.Discrete(grid_size),  # Bunny X
            spaces.Discrete(grid_size),  # Bunny Y
            spaces.Discrete(4),          # Discretized direction to carrot
            spaces.Discrete(len(self.distance_bins) + 1)  # Discretized distance to carrot
        ])
        self.reset()

    def reset(self):
        self.step_counter = 0
        return super().reset()

    def step(self, bunny_action):
        self.step_counter += 1
        new_bunny_pos = self.update_position(self.bunny_pos, bunny_action)
        self.bunny_pos = new_bunny_pos
        self.wolf_pos = self.update_position(self.wolf_pos, random.randint(0, 8))
        reward, done = self.calculate_reward(new_bunny_pos, self.wolf_pos)
        bunny_state = self.calculate_states(self.wolf_pos, new_bunny_pos)
        return bunny_state, reward, done

    def calculate_reward(self, bunny_pos, wolf_pos):
#         if (wolf_pos == bunny_pos):
#             return -50, True
        if bunny_pos == self.location_pos:
            return 100, True

         # Calculate the distance band to determine proximity
        distance_band = discretize_distance(np.linalg.norm(np.array(bunny_pos) - np.array(self.location_pos)), self.distance_bins)
        # The default reward is negative and becomes more negative the further away the bunny is
        # Assuming the first band is the closest, we reverse the distance band order for penalty calculation
        reward = -(len(self.distance_bins) - distance_band) * 2
        # Add a time penalty that gets worse the longer the wolf takes to catch the bunny
        time_penalty = -0.01 * self.step_counter  # Adjust the 0.01 factor to scale the time penalty
        reward += time_penalty
        return reward, False  # Bunny not caught yet
