import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import numpy as np

# 1. Create a Vectorized env.
# 'make_vec_env' creates multiple copies of the same environment (4 in this case)
# This allowes PPO to collect experiences in parallel, making training much faster

# 'monitor_kwargs' tells the env to record info (like reward) for us to plot later 
env = make_vec_env("CartPole-v1", n_envs=4)

# 2. Create a separate evaluation environment
# We don't want to evaluate the agent on the same environments it's training on.
eval_env = make_vec_env("CartPole-v1", n_envs=1)


# 3. Set up a callback to evaluate the agent during training 
# This will evaluate the agent every 10,000 steps and save the best model.

eval_callback = EvalCallback(eval_env, best_model_save_path='./best_model/', log_path='./logs/', eval_freq=1000, deterministic=True, render=False)


# 4. Instantiate the PPO Agent
# This creates the brain of our AI
# "MlpPolicy": Tells PPO to use a simple Multi-layer Perceptron (a standard Neural network)
# env: The environment the agent will learn from
# verbose=1: Prints information to the Console, so we can how training is going
# The other parameters are hyper-parameters that control *how* the agent learns

model = PPO(
    "MlpPolicy",
    env,
    verbose = 1,
    learning_rate = 3e-4,
    n_steps=2048,  # Collect 2048 steps *in total acroos all 4 environments* before updating
    batch_size = 64,     # Split the 2048 steps into mini-batches of 64 for training
    n_epochs = 10,    # Pass over the collected data 10 times for each update
    gamma=0.99,   # same as of RL- How much the agent cares about future rewards (0.99 = a lot)
    gae_lambda=0.95,  # Helps reduce variance in estimating "how good an action was"
    clip_range=0.2,   # the Core ofPPO: prevents the agent from changing too radically
    ent_coef=0.0, # Encourages exploration. 0.0 means no extra encouragement is added
    tensorboard_log="./ppo_cartpole_tensorboaard/"   # (Optional)  For advanced graphs
) 


# 5. Train the agent !
# This is where the magic happens. The agent will interact with the environment, 
# Learn from its actions and rewards, and improve its Policy for 10,000 total steps
model.learn(total_timesteps=10000, callback = eval_callback)

# 6. Save the Trained model
model.save("ppo_cartpole_gymnasium")


# 7. Let's plot the training progress!
# The EvalCallback logged the evaluation results. We can load and plot them

# First, let's see the Rewards over time from the training output itself. 
# The 'ep_rew_mean' from the Console output is a good proxy

# (This is a simple way to see the trend. For a more accurate graph, we would parse the true logs.)
print("\nTraining finished! Look at the 'ep_rew_mean' values above.")
print("They should have started Low (around 20-50) and climbed to near 500.\n")


# 8.  Load the best model (solve by the callback) and wath it perform!
print("Loading the Best saved model and demonstrating...")
model = PPO.load("./best_model/best_model")


# Creating a single environment for rendering
demo_env = gym.make("CartPole-v1", render_mode='human')
obs, info = demo_env.reset()
total_reward = 0

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = demo_env.step(action)
    total_reward += reward
    demo_env.render()  # Show the Animation

    if (terminated or truncated):
        obs, info = demo_env.reset()
        print(f"Episode finished! Total reward: {total_reward}")
        total_reward = 0
        # break       # uncomment to stop after one episode

demo_env.close()

