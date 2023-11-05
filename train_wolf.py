from Agent import Agent
from WolfEnvironment import WolfEnvironment
import argparse
import multiprocessing
from collections import defaultdict
import time

from utils import (
update_progress_bar,
combine_q_tables
)

def train_wolf_agent(agent, start_episode, end_episode, counter, eps):
    env_local = WolfEnvironment(grid_size=20)
    increment = eps / 100
    for episode in range(start_episode, end_episode):
        state = env_local.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env_local.step(action)
            agent.update(state, action, reward, next_state[0])
            state = next_state

        counter.value += 1
        if counter.value % increment == 0:
            update_progress_bar(counter.value, eps)
    return agent.q_table

# ... [other imports and train_wolf_agent function] ...

if __name__ == '__main__':
    start_time = time.time()
    multiprocessing.freeze_support()

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train Wolf Agent with Multiprocessing")
    parser.add_argument('--eps', type=int, default=10000, help="Number of episodes for training")
    args = parser.parse_args()

    # Determine the number of processes to use
    num_processes = multiprocessing.cpu_count()
    episodes_per_process = args.eps // num_processes

    # Initialize a shared counter for multiprocessing
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    # Initialize the environment to get the action space size
    env = WolfEnvironment(grid_size=20)
    action_space_size = env.action_space_size
    wolf_agent = Agent(action_space_size=action_space_size, eps = args.eps)
    wolf_agent.load('wolf_brain.npy')
    # Use multiprocessing pool to train agents in parallel
    with multiprocessing.Pool(num_processes) as pool:
        ranges = [(i * episodes_per_process, (i+1) * episodes_per_process) for i in range(num_processes)]
        args_for_train_wolf_agent = [(wolf_agent, start, end, counter, args.eps) for start, end in ranges]
        results = pool.starmap(train_wolf_agent, args_for_train_wolf_agent)

    # Combine the Q-tables from all processes
    wolf_combined_q_table = combine_q_tables(results)

    # Create a new agent to hold the combined Q-table
    wolf_agent = Agent(action_space_size=action_space_size)
    wolf_agent.q_table = wolf_combined_q_table

    # Save the combined Q-table to file
    wolf_agent.save('wolf_brain.npy')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The script took {elapsed_time / 60.0} minutes to run.")

