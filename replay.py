import os
import gym
import time
from gym.envs.registration import register
from stable_baselines3 import PPO

# === STEP 1: Define environment settings (same as in main.py) ===
lander_settings = {
    'Side Engines': True,
    'Clouds': True,
    'Vectorized Nozzle': True,
    'Starting Y-Pos Constant': 1,
    'Initial Force': 'random'
}

# === STEP 2: Register RocketLander environment ===
register(
    id="RocketLander-v0",
    entry_point="environments.rocketlander:RocketLander",
    max_episode_steps=1000,
    reward_threshold=0,
    kwargs={'settings': lander_settings}
)

# === STEP 3: Load latest trained PPO model ===
model_path = "logs/saved_models/ppo_rocket_lander_final.zip"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"❌ Model not found at {model_path}. "
        "Train your model first using main.py."
    )

env_id = "RocketLander-v0"
env = gym.make(env_id)
model = PPO.load(model_path, env=env)

print("\n--- 🚀 Starting Replay of Last 20 Episodes ---\n")

# n=150000
# n=245000
n=350000
# === STEP 4: Run and visualize last 20 episodes ===
for episode in range(n-10, n-1):
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        env.render()  # shows live simulation window
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        time.sleep(1 / 30)  # smooth visualization speed (30 FPS)

    print(f"Episode {episode + 1} | Steps: {step} | Total Reward: {total_reward:.2f}")

env.close()
print("\n✅ Replay finished successfully.\n")
