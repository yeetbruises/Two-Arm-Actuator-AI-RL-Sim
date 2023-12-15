"""
Vineet Saraf
12/13/2023
CPSC 4420
TestRunSARSA.py
"""
import time
import pandas as pd
import gym
import numpy as np
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

# Register the environment
gym.envs.register(
    id='PathEnv',
    entry_point='gym.envs.classic_control:PathEnv',
    kwargs={'path': None}
)
env = gym.make("PathEnv")

# We will use this function to find what bucket our action fits into.
def round_to_nearest_box(discretized_actions, input_tuple):
    min_distance = float('inf')
    nearest_box = None
    nearest_index = None
    for i, action in enumerate(discretized_actions):
        distance = np.linalg.norm(np.array(input_tuple) - np.array(action))
        if distance < min_distance:
            min_distance = distance
            nearest_box = action
            nearest_index = i
    return nearest_box, nearest_index


# Get length of the observation space and action space
n_observations = env.observation_space_n
n_action_space = env.action_space_n

# Create the Q-Table from the dimensions of the obs space and action space
Q_table = np.zeros((n_observations, n_action_space))

"""
HYPERPARAMETERS
"""
n_episodes = 10000                    # Number of episodes to be ran
max_iter_episode = 100                # Number of iterations equals the full length of the path
exploration_probe = 1                 # Exploration probe is always initialized to 1
exploration_decreasing_decay = 0.015  # Control the rate of exploration decaying, consequently increasing exploitation
min_exploration_probe = 0.005         # Lower bound on exploration probe
gamma = 0.99                          # Discounted factor
lr = 0.1                              # Learning rate - Can prioritize short term gains over long term ones.

total_rewards_episode = list()

env.reset()


# Iterate over each episode
for e in range(n_episodes):
    print(e)

    # Reset to the beginning, setting default angle values
    current_state = env.reset()[0]
    done = False

    # sum the rewards that the agent gets from the environment
    total_episode_reward = 0

    action = 0

    # Sample a float from a uniform distribution between 0 and 1
    # If the sampled float is less than exploration probe, sample a random action
    # Else exploit by finding the highest Q Value action in the given state
    if np.random.uniform(0, 1) < exploration_probe:
        action = env.action_space.sample()
    else:
        action = env.discretized_actions[np.argmax(Q_table[current_state[0], :])]

    # Set a hard bound on what angles are allowed, force a recalculation until req. is met.
    is_in_range = lambda num, lower, upper: lower <= num <= upper

    while not is_in_range((env.current_pos[1][0] + action[0]), env.ARM1_LOW, env.ARM1_HIGH) or \
            not is_in_range((env.current_pos[1][1] + action[1]), env.ARM2_LOW, env.ARM2_HIGH):
        if np.random.uniform(0, 1) < exploration_probe:
            action = env.action_space.sample()
        else:
            action = env.discretized_actions[np.argmax(Q_table[current_state[0], :])]
    action2 = 0

    # For each movement along the path/iteration/observation do the following.
    for i in range(max_iter_episode):
        # The environment runs the chosen action and returns
        next_state, reward, done, _ = env.step(action)

        # Sample a float from a uniform distribution between 0 and 1
        # If the sampled float is less than exploration probe, sample a random action
        # Else exploit by finding the highest Q Value action in the given state
        if np.random.uniform(0, 1) < exploration_probe:
            action2 = env.action_space.sample()
        else:
            action2 = env.discretized_actions[np.argmax(Q_table[next_state[0], :])]

        # Set a hard bound on what angles are allowed, force a recalculation until req. is met.
        is_in_range = lambda num, lower, upper: lower <= num <= upper

        while not is_in_range((env.current_pos[1][0] + action2[0]), env.ARM1_LOW, env.ARM1_HIGH) or \
                not is_in_range((env.current_pos[1][1] + action2[1]), env.ARM2_LOW, env.ARM2_HIGH):
            if np.random.uniform(0, 1) < exploration_probe:
                action2 = env.action_space.sample()
            else:
                action2 = env.discretized_actions[np.argmax(Q_table[next_state[0], :])]

        # Find which discretized box the random action fits into by getting its index
        _, q_action_index = round_to_nearest_box(env.discretized_actions, action)
        _, q_action2_index = round_to_nearest_box(env.discretized_actions, action2)

        # Update our Q-table using the Q-learning iteration
        Q_table[current_state[0], q_action_index] = (1 - lr) * Q_table[current_state[0], q_action_index] + \
                                                    lr * (reward + gamma * Q_table[next_state[0], q_action2_index] - Q_table[current_state[0], q_action_index])
        total_episode_reward = total_episode_reward + reward

        # Break the loop if done
        if done: break

        """
        Uncomment to view the simulation in pygame after a certain number of episodes.
        """
        # Render game past a certain number of episodes
        """if e >= 1500:
            env.render(e+1, i, reward)
            time.sleep(0.05)"""

        # Move to the following state
        current_state = next_state
        action = action2

    # Balance of exploration and exploitation should shift, this decays the exploration amount each episode.
    exploration_probe = max(min_exploration_probe, np.exp(-exploration_decreasing_decay * e))
    total_rewards_episode.append(total_episode_reward/max_iter_episode)

    """
    Uncomment to see statistics after a certain episode
    """
    """
    if e == 10000:
        print(sum(total_rewards_episode) / 10000)
        plt.plot([x for x in range(len(total_rewards_episode))], [x for x in total_rewards_episode])
        plt.ylabel('Reward')
        plt.title("SARSA Mean Reward Per Episode (n=10000)")
        plt.xlabel("Episodes")
        plt.axis((0, 10000, -80, 10))
        plt.show() 
    """

    # Export the DataFrame to an CSV file
    df = pd.DataFrame(Q_table)
    df.to_csv('q-table.csv', index=False, header=False)

print("Mean reward per thousand episodes")
for i in range(10):
    print(f"{(i+1)*1000} - Mean Episode Reward: ", np.mean(total_rewards_episode[1000*i:1000*(i+1)]))
