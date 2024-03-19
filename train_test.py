import gymnasium as gym
import soulsgym
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os

import logging

soulsgym.set_log_level(level=logging.DEBUG)

from stable_baselines3 import PPO

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-50:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

# Directory where logs and models are saved
log_dir = "./logs/"

# Create or load the environment
env = gym.make("Tekken8", game_speed=1)
env = Monitor(env, log_dir)
env_id = "Tekken8"

# Load the model; if starting afresh, comment this line and uncomment the model initialization line below
model_path = os.path.join(log_dir, 'best_model')
if os.path.exists(model_path):
    print("Loading existing model")
    model = PPO.load(model_path, env=env)
else:
    # Initialize a new model if there's no model to load
    print("Initializing new model")
    model = PPO("CnnPolicy", env, n_steps=200, verbose=1, tensorboard_log="./board/", learning_rate=0.00001)

# Callback for saving the best model
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# Continue training
print("Continuing training model...")
model.learn(total_timesteps=1000000, callback=callback, log_interval=4, tb_log_name="PPO0003")

# Save the model after training
model.save(env_id)
