from Agent import Agent
from tabulate import tabulate
import numpy as np


def tabulate_q_values(q_table, position):
    """ Tabulate Q-values for a specific position. """
    headers = ["Action", "Q-value"]
    actions = [
        "Up", "Down", "Left", "Right",
        "Diagonal up-left", "Diagonal up-right",
        "Diagonal down-left", "Diagonal down-right",
        "Stay"
    ]
    if position in q_table:
        table = list(zip(actions, q_table[position]))
        return tabulate(table, headers=headers)
    else:
        return f"No Q-values found for position {position}"

def default_q_values():
    """Return default Q-values."""
    return [0] * 9


wolf_agent = Agent(action_space_size=9)
wolf_agent.load("wolf_brain.npy")
wolf_agent_q_table = wolf_agent.q_table
print(np.all(wolf_agent.q_table == 0))