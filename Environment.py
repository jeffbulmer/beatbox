import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

class Environment(gym.Env):
    def __init__(self, grid_size=20):
        super(Environment, self).__init__()

        self.grid_size = grid_size
        self.action_space = spaces.Tuple([spaces.Discrete(9), spaces.Discrete(9)])
        self.observation_space = spaces.Tuple([spaces.Discrete(grid_size) for _ in range(6)])
        self.fig, self.ax = None, None
        self.prev_distance_wolf_bunny = None
        self.prev_distance_bunny_target = None
        self.reset()

    def reset(self):
        self.wolf_pos = np.random.randint(0, self.grid_size, size=2)
        self.bunny_pos = np.random.randint(0, self.grid_size, size=2)
        self.location_pos = (10, 10)
        self.prev_distance_wolf_bunny = self.manhattan_distance(self.wolf_pos, self.bunny_pos)
        self.prev_distance_bunny_target = self.manhattan_distance(self.bunny_pos, self.location_pos)

        while (self.wolf_pos == self.bunny_pos).all() or (self.bunny_pos == self.location_pos).all():
            self.bunny_pos = np.random.randint(0, self.grid_size, size=2)

        return tuple(self.wolf_pos) + tuple(self.bunny_pos) + tuple(self.location_pos)

    def update_position(self, pos, action):
        x, y = pos
        if action == 1:
            return x, max(0, y-1)
        elif action == 2:
            return x, min(self.grid_size-1, y+1)
        elif action == 3:
            return max(0, x-1), y
        elif action == 4:
            return min(self.grid_size-1, x+1), y
        elif action == 5:
            return max(0, x-1), max(0, y-1)
        elif action == 6:
            return min(self.grid_size-1, x+1), max(0, y-1)
        elif action == 7:
            return max(0, x-1), min(self.grid_size-1, y+1)
        elif action == 8:
            return min(self.grid_size-1, x+1), min(self.grid_size-1, y+1)
        return x, y

    def manhattan_distance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def step(self, actions):
        wolf_action, bunny_action = actions

        new_wolf_pos = self.update_position(self.wolf_pos, wolf_action)
        new_bunny_pos = self.update_position(self.bunny_pos, bunny_action)
        current_distance_wolf_bunny = self.manhattan_distance(new_wolf_pos, new_bunny_pos)
        current_distance_bunny_target = self.manhattan_distance(new_bunny_pos, self.location_pos)

        # Default rewards
        wolf_reward, bunny_reward = -1, -1

        # Check if wolf is closer to the bunny than before
        if current_distance_wolf_bunny < self.prev_distance_wolf_bunny:
            wolf_reward += 5  # You can adjust the value of the reward as needed

        # Check if bunny is closer to the carrot than before
        if current_distance_bunny_target < self.prev_distance_bunny_target:
            bunny_reward += 5  # You can adjust the value of the reward as needed

        # Update positions
        self.wolf_pos = new_wolf_pos
        self.bunny_pos = new_bunny_pos

        # Check end-game conditions
        if self.wolf_pos == self.bunny_pos:
            reward = (-100, 100)
            done = True
        elif self.bunny_pos == self.location_pos:
            reward = (100, -100)
            done = True
        else:
            reward = (wolf_reward, bunny_reward)
            done = False

        self.prev_distance_wolf_bunny = current_distance_wolf_bunny
        self.prev_distance_bunny_target = current_distance_bunny_target

        return tuple(new_wolf_pos) + tuple(new_bunny_pos) + tuple(self.location_pos), reward, done, {}


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
