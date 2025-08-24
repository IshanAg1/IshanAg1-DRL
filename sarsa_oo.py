import numpy as np
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import matplotlib.pyplot as plt

env =  FlatObsWrapper(gym.make("MiniGrid-Empty-6x6-v0"))


def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, 3)  # Actions: left(0), right(1), forward(2)
    q_vals = [Q.get((tuple(state), a), 0.0) for a in range(3)]
    return int(np.argmax(q_vals))


def sarsa(env, gamma=0.9, alpha=0.5 , noof_episodes = 1000):
    Q = {}    # empty dictionary with key = (state_tuple, action)
    rewards = []

    epsilon = 1.0
    min_epsilon = 0.1
    decay_rate = 0.995
    
    for episode in range(noof_episodes):
        obs, info = env.reset()
        state = obs
        action = epsilon_greedy(Q, state, epsilon)

        temp_reward = 0
        terminated = truncated = False 
        
        while not (terminated or truncated):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs

            next_action = epsilon_greedy(Q, next_state, epsilon)
            old_q = Q.get((tuple(state), action), 0.0)    # 
            next_q = Q.get((tuple(next_state), next_action), 0.0)

            Q[(tuple(state), action)] = (1-alpha)*old_q + alpha*(reward + gamma*next_q)

            state = next_state
            action = next_action
            temp_reward += reward

        epsilon = max(min_epsilon, epsilon * decay_rate)
        rewards.append(temp_reward)
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}: Reward = {temp_reward:.3f}, Epsilon = {epsilon:.3f}")

    
    return rewards


num_episodes = 400
rewards = sarsa(env, noof_episodes=num_episodes)

plt.figure(figsize=(12, 6))
plt.plot(rewards, alpha=0.3, label="Episode Reward")
window = 30
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, num_episodes), moving_avg, color='red', label=f"{window}-ep Moving Avg")
plt.title("SARSA with FlatObsWrapper", fontsize=14)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

env.close()
            
            