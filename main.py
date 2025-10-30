
import os
import gym
import time
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import numpy as np

# --- Define the required settings dictionary ---
lander_settings = {
    'Side Engines': True,
    'Clouds': True,
    'Vectorized Nozzle': True,
    'Starting Y-Pos Constant': 1,
    'Initial Force': 'random'
}

# --- Manually register the environment for gym ---
register(
     id="RocketLander-v0",
     entry_point="environments.rocketlander:RocketLander",
     max_episode_steps=1000,
     reward_threshold=0,
     kwargs={'settings': lander_settings} # Pass the settings here
)
# ---------------------------------------------------

# --- Setup Directories ---
log_dir = "logs/"
model_dir = os.path.join(log_dir, "saved_models/")
tensorboard_log_dir = os.path.join(log_dir, "tensorboard/")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)
model_save_path = os.path.join(model_dir, "ppo_rocket_lander_final.zip")

# --- Create the Environment ---
env_id = "RocketLander-v0"
env = make_vec_env(env_id, n_envs=8)


#  HYPER-PARAMTERS for AGGRESSIVE Training  within  200k Time steps
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    # learning_rate=0.0003,      # Increased from 0.0001 for faster updates
    # learning_rate=0.0003,      # Increased from 0.0001 for faster updates
    learning_rate=0.0004,      # Increased from 0.0001 for faster updates
    # n_steps=1024,              # Decreased from 2048 for more frequent updates
    n_steps=2048,
    # batch_size=64,
    # batch_size=128,     # increase the batch size 
    batch_size=256,
    # n_epochs = 4
    # n_epochs=20,               # Increased from 10 to learn more from each batch
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    # clip_range=0.2,
    clip_range=0.3,
    # ent_coef=0.03,
    ent_coef=0.04,

    policy_kwargs=dict(net_arch=[256, 256]), # Using a larger neural network
    tensorboard_log=tensorboard_log_dir
)



# --- Train the Model ---
# n = 150000
# n = 200000
# n = 245000
n = 350000
# n = 580000
# n = 500000
training_timesteps = n
print("--- Starting Model Training ---")
model.learn(total_timesteps=training_timesteps)
print("--- Finished Model Training ---")

# --- Save and Evaluate the Model ---
model.save(model_save_path)
print(f"✅ Model saved to: {model_save_path}")


# n_additional = 200000
# training_timesteps = n_additional


del model
print("Loading previous model")
model = PPO.load(model_save_path, env=env)     # Load model from 300k


# print(f"--- Continuing Training for {training_timesteps} timesteps ---")

# # 2️⃣ Optional: continue logging in same tensorboard folder
# model.tensorboard_log = "logs/tensorboard/PPO_11"  # reuse old folder if needed

# # 3️⃣ Continue training
# model.learn(total_timesteps = training_timesteps, reset_num_timesteps = False)  # <-- keep timesteps count
# # model.learn(total_timesteps=training_timesteps)
# print("--- Finished Continued Training ---")

# # --- Save the Updated Model ---
# model.save(model_save_path)  # Overwrite or save a new path if desired
# print(f"✅ Updated Model saved to: {model_save_path}")


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"\n--- Evaluation Results ---")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# --- Visualize the Trained Agent (gym 0.21.0 API) ---
print("\n--- Visualizing Agent's Performance ---")

# Recording n-100 to n-80 th episode
print("\n--- 🎬 Recording Last 20 Episode Simulations ---")


video_dir = os.path.join(log_dir, "videos")
os.makedirs(video_dir, exist_ok=True)
video_path = os.path.join(video_dir, f"PPO_sim_{int(time.time())}.mp4")
frames = []

vis_env = gym.make(env_id)

for episode in range(n-100, n-98):
    obs = vis_env.reset()
    done = False
    episode_reward = 0
    while not done:
        vis_env.render()

        frame = vis_env.render(mode='rgb_array')  # Capture frame
        frames.append(frame)

        # --- Add this line to keep the window responsive ---
        # vis_env.viewer.window.dispatch_events()

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vis_env.step(action)
        episode_reward += reward
        time.sleep(1/30)

    print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

# Save the video
imageio.mimsave(video_path, frames, fps=30)
vis_env.close()
print(f"\n🎥 Simulation video saved at: {video_path}")
print("✅ All tasks completed successfully.")


'''
# --- Setup video path ---
video_dir = os.path.join(log_dir, "videos")
os.makedirs(video_dir, exist_ok=True)
video_path = os.path.join(video_dir, f"PPO_sim_{int(time.time())}.mp4")

# --- Create a separate environment for visualization ---
vis_env = gym.make(env_id)

# --- Record episodes directly to video ---
with imageio.get_writer(video_path, fps=30) as video:
    for episode in range(n-100, n-93):
        obs = vis_env.reset()
        done = False
        episode_reward = 0

        while not done:
            frame = vis_env.render(mode='rgb_array')  # Capture frame
            video.append_data(frame)  # Write frame directly to video

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vis_env.step(action)
            episode_reward += reward

            time.sleep(1/30)  # Optional: slow down for real-time viewing

        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

vis_env.close()
print(f"\n🎥 Simulation video saved at: {video_path}")
print("✅ All tasks completed successfully.")
'''
