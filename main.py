from Agent import Agent
from Environment import Environment
import argparse
import multiprocessing
from collections import defaultdict


def train_agent(start_episode, end_episode, counter, eps):
    env_local = Environment(grid_size=20)
    wolf_agent_local = Agent(action_space_size=9)
    bunny_agent_local = Agent(action_space_size=9)
    wolf_agent_local.load('wolf_brain.npy')
    bunny_agent_local.load('bunny_brain.npy')

    for episode in range(start_episode, end_episode):
        state = env_local.reset()
        wolf_state = state[:2]
        bunny_state = state[2:4]
        done = False

        while not done:
            wolf_action = wolf_agent_local.get_action(tuple(wolf_state))
            bunny_action = bunny_agent_local.get_action(tuple(bunny_state))
            next_state, reward, done = env_local.step((wolf_action, bunny_action))
            next_wolf_state = next_state[0]
            next_bunny_state = next_state[1]
            wolf_agent_local.update(wolf_state, wolf_action, reward[0], next_wolf_state)
            bunny_agent_local.update(bunny_state, bunny_action, reward[1], next_bunny_state)
            wolf_state, bunny_state = next_wolf_state, next_bunny_state

        counter.value += 1
        if counter.value % 50 == 0:
            print(f"Episode {counter.value}/{eps} completed!")

    return wolf_agent_local.q_table, bunny_agent_local.q_table


if __name__ == '__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Q-learning agents: Wolf and Bunny")
    parser.add_argument('--render', action='store_true', help="Render the environment after training")
    parser.add_argument('--eps', type=int, default=10000, help="Number of episodes for training")
    parser.add_argument('--train', action='store_true', help="Train the agents")

    args = parser.parse_args()

    env = Environment(grid_size=20)
    wolf_agent = Agent(action_space_size=9)
    bunny_agent = Agent(action_space_size=9)
    wolf_agent.load('wolf_brain.npy')
    bunny_agent.load('Old_bunny.npy')

    if args.train:
        num_processes = 4
        episodes_per_process = args.eps // num_processes
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)

        with multiprocessing.Pool(num_processes) as pool:
            ranges = [(i * episodes_per_process, (i+1) * episodes_per_process) for i in range(num_processes)]
            args_for_train_agent = [(start, end, counter, args.eps) for start, end in ranges]
            results = pool.starmap(train_agent, args_for_train_agent)

        # Combine the results
        wolf_combined_q_table = defaultdict(float)
        bunny_combined_q_table = defaultdict(float)

        for wolf_q, bunny_q in results:
            for key, value in wolf_q.items():
                wolf_combined_q_table[key] += value
            for key, value in bunny_q.items():
                bunny_combined_q_table[key] += value

        for key in wolf_combined_q_table:
            wolf_combined_q_table[key] /= num_processes
        for key in bunny_combined_q_table:
            bunny_combined_q_table[key] /= num_processes

        wolf_agent.q_table = wolf_combined_q_table
        bunny_agent.q_table = bunny_combined_q_table

        wolf_agent.save('wolf_brain.npy')
        bunny_agent.save('bunny_brain.npy')
    else:
        print("No Brain")
        wolf_agent.set_exploration_rate(0)
        bunny_agent.set_exploration_rate(0)
    state = env.reset()
    done = False
    if args.render:
        while not done:
            wolf_state = state[:2]
            bunny_state = state[2:4]
            wolf_action = wolf_agent.get_action(tuple(wolf_state))
            bunny_action = bunny_agent.get_action(tuple(bunny_state))
            state, _, done = env.step((wolf_action, bunny_action))
            env.render()
