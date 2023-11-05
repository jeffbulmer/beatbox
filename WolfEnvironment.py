import math
import numpy as np
from gym import spaces
from Environment import Environment
import random
from utils import (
    discretize_distance,
    discretize_angle,
    angle_between_points
)

class WolfEnvironment(Environment):
    def __init__(self, grid_size=20):
        super(WolfEnvironment, self).__init__(grid_size)
        self.num_angle_bins = 32
        self.distance_bins = [1, 2, 3, 5, 10, 20, 40]
        self.action_space_size = 9
        self.location_pos = (9,9)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Tuple([
            spaces.Discrete(grid_size),  # Wolf X
            spaces.Discrete(grid_size),  # Wolf Y
            spaces.Discrete(self.num_angle_bins),          # Discretized direction to bunny
            spaces.Discrete(len(self.distance_bins) + 1)  # Discretized distance to bunny
        ])
        self.step_counter = 0

    def reset(self):
        self.step_counter = 0
        return super().reset()

    def step(self, wolf_action):
        self.step_counter += 1
        new_wolf_pos = self.update_position(self.wolf_pos, wolf_action)
        wolf_state = self.calculate_states(new_wolf_pos, self.bunny_pos)
        self.wolf_pos = new_wolf_pos
        reward, done = self.calculate_reward(new_wolf_pos, self.bunny_pos)
        self.bunny_pos = self.update_position(self.bunny_pos, random.randint(0, 8))
        return wolf_state, reward, done

    def calculate_reward(self, wolf_pos, bunny_pos):
        if np.array_equal(wolf_pos, bunny_pos):
            return 100, True  # The wolf catches the bunny

        # Calculate the distance band to determine proximity
        distance_band = discretize_distance(np.linalg.norm(np.array(wolf_pos) - np.array(bunny_pos)), self.distance_bins)
        # The default reward is negative and becomes more negative the further away the bunny is
        # Assuming the first band is the closest, we reverse the distance band order for penalty calculation
        reward = -(len(self.distance_bins) - distance_band) * 2
        # Add a time penalty that gets worse the longer the wolf takes to catch the bunny
        time_penalty = -0.01 * self.step_counter  # Adjust the 0.01 factor to scale the time penalty
        reward += time_penalty

        return reward, False  # Bunny not caught yet