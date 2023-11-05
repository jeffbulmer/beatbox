import sys
from collections import defaultdict
import math

def update_progress_bar(counter_value, total_eps, bar_length=50):
    """
    Updates the progress bar in the console.

    Args:
    - counter_value: the current episode count.
    - total_eps: the total number of episodes.
    - bar_length: the character length of the bar (default is 50 characters).
    """

    # Calculate the percent completion
    percent_complete = counter_value / total_eps

    # Calculate the number of filled in characters on the bar
    num_filled = int(bar_length * percent_complete)

    # Create the bar string
    bar = '█' * num_filled + '-' * (bar_length - num_filled)

    # Print the progress bar - the '\r' character returns the cursor to the start of the line
    sys.stdout.write(f'\r[{bar}] {percent_complete * 100:.2f}% Complete ({counter_value}/{total_eps})')

    # Flush the stdout buffer to ensure the bar is displayed
    sys.stdout.flush()


def combine_q_tables(results):
    combined_q_table = defaultdict(float)
    for q_table in results:
        for key, value in q_table.items():
            combined_q_table[key] += value
    for key in combined_q_table:
        combined_q_table[key] /= len(results)
    return combined_q_table


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