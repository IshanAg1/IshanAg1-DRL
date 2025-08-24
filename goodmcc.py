import numpy as np
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_agent_state(obs):
    """Extracts (x, y, direction) from observation"""
    grid = obs["image"]
    agent_pos = np.where((grid[:, :, 0] == 10) & (grid[:, :, 1] == 0) & (grid[:, :, 2] == 0))
    if len(agent_pos[0]) == 0:
        return (0, 0, 0)
    x, y = agent_pos[0][0], agent_pos[1][0]
    direction = obs["direction"]
    return (x, y, direction)

def monte_carlo_control(env, n_episodes=1000, gamma=0.95, 
                        epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.97):
    Q = np.zeros((6, 6, 4, 3))  # (x, y, dir, action)
    returns = {}  # Dictionary to store returns for each (s,a)
    rewards_history = []
    steps_history = []
    epsilon = epsilon_start
    
    for episode in tqdm(range(n_episodes)):
        # Generating episode
        obs, _ = env.reset()
        state = get_agent_state(obs)
        episode_data = []
        steps = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            # epsilon-greedy action selection
            x, y, dir = state
            if np.random.random() < epsilon:
                action = np.random.randint(0, 3)
            else:
                action = np.argmax(Q[x, y, dir, :])
            
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = get_agent_state(obs)
            episode_data.append((state, action, reward))
            state = next_state
            steps += 1
        
        rewards_history.append(sum(r for (_, _, r) in episode_data))
        steps_history.append(steps)
        
        # Update Q-values using Monte Carlo
        G = 0
        visited = set()
        
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = gamma * G + reward
            sa_pair = (state, action)
            
            if sa_pair not in visited:
                visited.add(sa_pair)
                if sa_pair not in returns:
                    returns[sa_pair] = []
                returns[sa_pair].append(G)
                x, y, dir = state
                Q[x, y, dir, action] = np.mean(returns[sa_pair])
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    return Q, rewards_history, steps_history

env = gym.make("MiniGrid-Empty-6x6-v0", render_mode = None)
# env = FullyObsWrapper(env)

Q, rewards, steps = monte_carlo_control(env, n_episodes=300)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(rewards, alpha=0.3, color='blue')
window_size = 50
moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(moving_avg, color='blue', linewidth=2, label='Moving Avg')
plt.title("Reward per Episode - Monte Carlo", fontsize=12)
plt.xlabel("Episode", fontsize=10)
plt.ylabel("Average Reward", fontsize=10)
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(steps, alpha=0.3, color='green')
steps_avg = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
plt.plot(steps_avg, color='green', linewidth=2, label='Moving Avg')
plt.title("Steps per Episode - Monte Carlo", fontsize=12)
plt.xlabel("Episode", fontsize=10)
plt.ylabel("Average Steps", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
env.close()