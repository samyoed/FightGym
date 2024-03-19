import gymnasium as gym
import soulsgym
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import os

# Set the log level for soulsgym to reduce output verbosity if desired
import logging
soulsgym.set_log_level(level=logging.ERROR)  # Use ERROR to minimize log output

# Load the model
model_path = "./logs/best_model_reina"  # Adjust the path as necessary
model = PPO.load(model_path)

# Create the environment
env = gym.make("Tekken8", game_speed=1)

# Number of episodes you want to run
num_episodes = 10

for episode in range(num_episodes):
    obs, info = env.reset()  # Reset the environment at the start of each episode
    done = False
    total_rewards = 0  # Keep track of the total rewards in the episode

    while not done:
        action, state = model.predict(obs, deterministic=True)
        obs, rewards, done, info, extra = env.step((int)(action))  # Take the action
        total_rewards += rewards
        print((int)(action))
        
    print(f"Episode: {episode + 1}, Total Rewards: {total_rewards}")

env.close()  # Don't forget to close the environment 