# train_bunny.py
from Agent import Agent
from BunnyEnvironment import BunnyEnvironment  # A separate environment for the bunny
import argparse
import multiprocessing
from collections import defaultdict
import sys
from utils import (
    update_progress_bar,
    combine_q_tables
)

def train_bunny_agent(bunny_agent, start_episode, end_episode, counter, eps):
    env = BunnyEnvironment(grid_size=20)
    increment = eps / 100
    for episode in range(start_episode, end_episode):
        state = env.reset()
        done = False

        while not done:
            action = bunny_agent.get_action(state)
            next_state, reward, done = env.step(action)
            bunny_agent.update(state, action, reward, next_state[1])
            state = next_state

        counter.value += 1
        if counter.value % increment == 0:
            update_progress_bar(counter.value, eps)

    return bunny_agent.q_table

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train Bunny Agent with Multiprocessing")
    parser.add_argument('--eps', type=int, default=10000, help="Number of episodes for training")
    args = parser.parse_args()

    # Determine the number of processes to use
    num_processes = multiprocessing.cpu_count()
    episodes_per_process = args.eps // num_processes

    # Initialize a shared counter for multiprocessing
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    # Initialize the environment to get the action space size
    env = BunnyEnvironment(grid_size=20)
    action_space_size = 9
    bunny_agent = Agent(action_space_size=action_space_size, eps = args.eps)
    bunny_agent.load('bunny_brain.npy')
    # Use multiprocessing pool to train agents in parallel
    with multiprocessing.Pool(num_processes) as pool:
        ranges = [(i * episodes_per_process, (i+1) * episodes_per_process) for i in range(num_processes)]
        args_for_train_bunny_agent = [(bunny_agent, start, end, counter, args.eps) for start, end in ranges]
        results = pool.starmap(train_bunny_agent, args_for_train_bunny_agent)

    # Combine the Q-tables from all processes
    bunny_combined_q_table = combine_q_tables(results)

    # Create a new agent to hold the combined Q-table
    bunny_agent = Agent(action_space_size=9)
    bunny_agent.q_table = bunny_combined_q_table

    # Save the combined Q-table to file
    bunny_agent.save('bunny_brain.npy')
