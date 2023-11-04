# train_bunny.py
from Agent import Agent
from BunnyEnvironment import BunnyEnvironment  # Assuming you have a separate environment for bunny
import argparse
import multiprocessing
from collections import defaultdict

def train_bunny(start_episode, end_episode, counter, eps):
    env = BunnyEnvironment(grid_size=20)
    bunny_agent = Agent(action_space_size=env.action_space_size)
    bunny_agent.load('bunny_brain.npy')

    for episode in range(start_episode, end_episode):
        state = env.reset()
        done = False

        while not done:
            action = bunny_agent.get_action(state)
            next_state, reward, done = env.step(action)
            bunny_agent.update(state, action, reward, next_state)
            state = next_state

        counter.value += 1
        if counter.value % 50 == 0:
            print(f"Episode {counter.value}/{eps} completed!")

    bunny_agent.save('bunny_brain.npy')

if __name__ == '__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Train Bunny Agent")
    parser.add_argument('--eps', type=int, default=10000, help="Number of episodes for training")
    args = parser.parse_args()

    num_processes = 4
    episodes_per_process = args.eps // num_processes
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    with multiprocessing.Pool(num_processes) as pool:
        ranges = [(i * episodes_per_process, (i+1) * episodes_per_process) for i in range(num_processes)]
        pool.starmap(train_bunny, [(start, end, counter, args.eps) for start, end in ranges])
