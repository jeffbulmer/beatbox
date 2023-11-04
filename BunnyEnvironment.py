import math
from gym import spaces
from Environment import Environment
import numpy as np
def discretize_angle(angle, num_bins):
    """Discretizes the angle into one of several bins."""
    bin_size = 2 * math.pi / num_bins
    # Normalize angle between 0 and 2*pi
    normalized_angle = angle % (2 * math.pi)
    # Find the bin index
    bin_index = int(normalized_angle // bin_size)
    return bin_index


def discretize_distance(distance, bins):
    """Discretizes the distance into one of several bins."""
    for i, bin_edge in enumerate(bins):
        if distance <= bin_edge:
            return i
    return len(bins)


def angle_between_points(point1, point2):
    # Unpack points
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the differences
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the angle
    angle = math.atan2(dy, dx)

    return angle



class BunnyEnvironment(Environment):
    def __init__(self, grid_size=20):
        super(BunnyEnvironment, self).__init__(grid_size)
        self.num_angle_bins = 4
        # Define distance bins (e.g., 0-5, 5-10, >10)
        self.distance_bins = [5, 10]
        # Bunny can only move, there is no wolf in this environment
        self.action_space = spaces.Discrete(9)  # Only bunny actions
        self.observation_space = spaces.Tuple([
            spaces.Discrete(grid_size),  # Bunny X
            spaces.Discrete(grid_size),  # Bunny Y
            spaces.Discrete(4),          # Discretized direction to carrot
            spaces.Discrete(len(self.distance_bins) + 1)  # Discretized distance to carrot
        ])

    def step(self, bunny_action):
        new_bunny_pos = self.update_position(self.bunny_pos, bunny_action)

        bunny_state = self.calculate_state(new_bunny_pos)

        self.bunny_pos = new_bunny_pos
        reward, done = self.calculate_reward(new_bunny_pos)
        return bunny_state, reward, done

    def calculate_reward(self, bunny_pos):
        if bunny_pos == self.location_pos:
            return 100, True
        return -1, False

    def calculate_state(self, bunny_pos):
        bunny_carrot_angle = angle_between_points(bunny_pos, self.location_pos)
        bunny_carrot_distance = np.linalg.norm(np.array(bunny_pos) - np.array(self.location_pos))

        bunny_carrot = (discretize_angle(bunny_carrot_angle, self.num_angle_bins),
                        discretize_distance(bunny_carrot_distance, self.distance_bins))

        return bunny_carrot

# Now you have two environments: WolfEnvironment for the wolf training and BunnyEnvironment for the bunny training.
