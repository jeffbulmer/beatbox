from typing import Tuple
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import math

from utils import (
    discretize_distance,
    discretize_angle,
    angle_between_points
)

class Environment(gym.Env):
    def __init__(self, grid_size=20):
        super(Environment, self).__init__()
        self.num_angle_bins = 32
        self.distance_bins = [1, 2, 3, 5, 10, 20, 40]
        self.grid_size = grid_size
        self.action_space = spaces.Tuple([spaces.Discrete(9), spaces.Discrete(9)])
        self.observation_space = spaces.Tuple([spaces.Discrete(grid_size) for _ in range(6)])
        self.fig, self.ax = None, None
        self.reset()

    def reset(self):
        self.wolf_pos = tuple(np.random.randint(0, self.grid_size, size=2))
        self.bunny_pos = tuple(np.random.randint(0, self.grid_size, size=2))
        self.location_pos = (9, 9)
        while (self.wolf_pos == self.bunny_pos) or (self.bunny_pos == self.location_pos):
                self.bunny_pos = tuple(np.random.randint(0, self.grid_size, size=2))

        return self.calculate_states(self.wolf_pos, self.bunny_pos)

    def update_position(self, pos, action) -> Tuple[int, int]:
        x, y = pos
        if action == 1:
            return x, max(0, y - 1)
        elif action == 2:
            return x, min(self.grid_size - 1, y + 1)
        elif action == 3:
            return max(0, x - 1), y
        elif action == 4:
            return min(self.grid_size - 1, x + 1), y
        elif action == 5:
            return max(0, x - 1), max(0, y - 1)
        elif action == 6:
            return min(self.grid_size - 1, x + 1), max(0, y - 1)
        elif action == 7:
            return max(0, x - 1), min(self.grid_size - 1, y + 1)
        elif action == 8:
            return min(self.grid_size - 1, x + 1), min(self.grid_size - 1, y + 1)
        return x, y

    def calculate_states(self, wolf_pos, bunny_pos):
        # Discretize angles into 4 bins (up, down, left, right)
        num_angle_bins = self.num_angle_bins
        # Define distance bins (e.g., 0-5, 5-10, >10)
        distance_bins = self.distance_bins

        wolf_bunny_angle = angle_between_points(wolf_pos, bunny_pos)
        wolf_bunny_distance = np.linalg.norm(np.array(wolf_pos) - np.array(bunny_pos))
        wolf_carrot_angle = angle_between_points(wolf_pos, self.location_pos)
        wolf_carrot_distance = np.linalg.norm(np.array(wolf_pos) - np.array(self.location_pos))

        bunny_wolf_angle = angle_between_points(bunny_pos, wolf_pos)
        bunny_wolf_distance = np.linalg.norm(np.array(bunny_pos) - np.array(wolf_pos))
        bunny_carrot_angle = angle_between_points(bunny_pos, self.location_pos)
        bunny_carrot_distance = np.linalg.norm(np.array(bunny_pos) - np.array(self.location_pos))

        # Discretize both angles and distances
        wolf_bunny = (discretize_angle(wolf_bunny_angle, num_angle_bins),
                      discretize_distance(wolf_bunny_distance, distance_bins))
        wolf_carrot = (discretize_angle(wolf_carrot_angle, num_angle_bins),
                       discretize_distance(wolf_carrot_distance, distance_bins))
        bunny_wolf = (discretize_angle(bunny_wolf_angle, num_angle_bins),
                      discretize_distance(bunny_wolf_distance, distance_bins))
        bunny_carrot = (discretize_angle(bunny_carrot_angle, num_angle_bins),
                        discretize_distance(bunny_carrot_distance, distance_bins))

        return (wolf_bunny, wolf_carrot), (bunny_wolf, bunny_carrot)

    def step(self, actions):
        # wolf, bunny
        # action is (0-9, 0-9)
        new_wolf_pos = self.update_position(self.wolf_pos, actions[0])
        new_bunny_pos = self.update_position(self.bunny_pos, actions[1])

        wolf_state, bunny_state = self.calculate_states(new_wolf_pos, new_bunny_pos)

        self.wolf_pos = new_wolf_pos
        rewards, done = self.calculate_rewards(new_wolf_pos, new_bunny_pos)
        self.bunny_pos = new_bunny_pos
        states = (wolf_state, bunny_state)
        return states, rewards, done

    def calculate_rewards(self, wolf_pos, bunny_pos):
        if bunny_pos == self.location_pos:
            return (-100, 100), True
        elif wolf_pos == bunny_pos:
            return (100, -100), True
        return (-1, -1), False

    def render(self, mode='human'):
        if mode == 'human':
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots(figsize=(6, 6))

            grid = np.ones((self.grid_size, self.grid_size, 3))
            grid[self.wolf_pos[1], self.wolf_pos[0]] = [1, 0, 0]
            grid[self.bunny_pos[1], self.bunny_pos[0]] = [0, 0, 1]
            grid[self.location_pos[1], self.location_pos[0]] = [1, 0.5, 0]

            self.ax.clear()
            self.ax.imshow(grid)

            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Wolf'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Bunny'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=(1, 0.5, 0), markersize=10, label='Destination')
            ]
            self.ax.legend(handles=legend_elements, loc='upper right')

            plt.pause(0.1)
            plt.draw()
        elif mode == 'rgb_array':
            pass
