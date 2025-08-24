import numpy as np
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import matplotlib.pyplot as plt

env = FlatObsWrapper(gym.make("MiniGrid-Empty-6x6-v0", render_mode="ansi"))

def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, 3)  # Actions: left(0), right(1), forward(2)
    q_vals = [Q.get((tuple(state), a), 0.0) for a in range(3)]
    return int(np.argmax(q_vals))

def q_learning(env, alpha=0.5, gamma=0.99, num_episodes=1000):
    Q = {}  # Dictionary Q-table: key = (state_tuple, action)
    rewards = []

    epsilon = 1.0
    min_epsilon = 0.1
    decay_rate = 0.995

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = obs
        total_reward = 0
        terminated = truncated = False     

        while not (terminated or truncated):
            action = epsilon_greedy(Q, state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs      # [1, 2, 1, 0.0,  ,  ]

            # Get max Q value for next state
            max_q_next = max([Q.get((tuple(next_state), a), 0.0) for a in range(3)])
            old_q = Q.get((tuple(state), action), 0.0)

            # Q-learning update
            Q[(tuple(state), action)] = (1 - alpha) * old_q + alpha * (reward + gamma * max_q_next)

            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * decay_rate)
        rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}: Reward = {total_reward:.3f}, Epsilon = {epsilon:.3f}")

    return rewards

num_episodes = 400
rewards = q_learning(env, num_episodes=num_episodes)

plt.figure(figsize=(12, 6))
plt.plot(rewards, alpha=0.3, label="Episode Reward")
window = 30
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, num_episodes), moving_avg, color='red', label=f"{window}-ep Moving Avg")
plt.title("Q-learning with FlatObsWrapper", fontsize=14)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

env.close()