
import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent

def plot_q_values_for_fixed_positions(q_table, fixed_positions, agent_name, ax):
    """
    Plot Q-values for a fixed bunny and destination position for the given agent.
    """
    max_q_values = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            key = ((i, j),) + tuple(fixed_positions)
            if key in q_table:
                max_q_values[i, j] = np.max(q_table[key])

    # Plot heatmap
    im = ax.imshow(max_q_values, cmap='viridis', origin='upper', interpolation='nearest')

    # Add a colorbar for the heatmap scale reference
    plt.colorbar(im, ax=ax, orientation='vertical')

    # Mark fixed positions
    ax.plot(fixed_positions[1][1], fixed_positions[1][0], 'rX', markersize=10) # destination
    ax.plot(fixed_positions[0][1], fixed_positions[0][0], 'bX', markersize=10) # bunny

    ax.set_title(f'{agent_name} Q-values')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

if __name__ == "__main__":
    wolf_agent = Agent(action_space_size=9)
    bunny_agent = Agent(action_space_size=9)
    wolf_agent.load('wolf_brain.npy')
    bunny_agent.load('bunny_brain.npy')

    wolf_agent_q_table = wolf_agent.q_table
    bunny_agent_q_table = bunny_agent.q_table

    bunny_position = (10, 10)  # Fixed bunny position
    destination_position = (15, 15)  # Fixed escape location

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_q_values_for_fixed_positions(wolf_agent_q_table, [bunny_position, destination_position], "Wolf", ax1)
    plot_q_values_for_fixed_positions(bunny_agent_q_table, [bunny_position, destination_position], "Bunny", ax2)

    plt.tight_layout()
    plt.show()
