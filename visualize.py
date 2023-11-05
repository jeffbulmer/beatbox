import numpy as np
import matplotlib.pyplot as plt
import pickle
from Agent import Agent
import argparse
parser = argparse.ArgumentParser(description="Q-learning agents: Wolf and Bunny")
parser.add_argument('--brain', type=str, default='wolf_brain.npy', help="brain")
args = parser.parse_args()
# Load the agent from file
agent = Agent(action_space_size=9)
agent.load(args.brain)
q_table = agent.q_table

# Check if Q-table is not empty
if not q_table:
    raise ValueError("The Q-table is empty. Cannot visualize an empty Q-table.")

# Extract all unique states and actions from the Q-table keys
states = set()
actions = set()
for (state, action), _ in q_table.items():
    states.add(state)
    actions.add(action)

# Create mappings from states and actions to indices
state_to_index = {state: index for index, state in enumerate(sorted(states))}
action_to_index = {action: index for index, action in enumerate(sorted(actions))}

# Initialize an array for the Q-table
state_space_size = len(state_to_index)
action_space_size = len(action_to_index)
q_table_array = np.full((state_space_size, action_space_size), np.nan)

# Populate the array
for (state, action), value in q_table.items():
    state_index = state_to_index[state]
    action_index = action_to_index[action]
    q_table_array[state_index, action_index] = value

# Replace nan values with zeros or the global minimum, depending on how you want to handle them
q_table_array = np.nan_to_num(q_table_array, nan=np.nanmin(q_table_array))

# Normalize the Q-table values to make it easier to visualize
min_q_value = np.min(q_table_array[np.nonzero(q_table_array)])
normalized_q_table = (q_table_array - min_q_value) / (np.max(q_table_array) - min_q_value)

def plot_q_table_heatmap(q_table, title):
    # Plotting the heatmap
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.imshow(q_table, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.show()

plot_q_table_heatmap(normalized_q_table, 'Q-Table Heatmap for Wolf and Bunny')
