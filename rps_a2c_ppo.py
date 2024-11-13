import supersuit as ss
from pettingzoo.classic import rps_v2
from stable_baselines3 import A2C, PPO
from supersuit import pettingzoo_env_to_vec_env_v1
from pettingzoo.utils.conversions import aec_to_parallel
import numpy as np

# Step 1: Initialize the PettingZoo Rock-Paper-Scissors environment
env = rps_v2.env()

# Step 2: Convert the AEC environment to a Parallel environment
parallel_env = aec_to_parallel(env)

# Step 3: Wrap the parallel environment to make it compatible with Stable-Baselines3
vec_env = pettingzoo_env_to_vec_env_v1(parallel_env)

# Step 4: Initialize the models (A2C for player_0, PPO for player_1)
a2c_model = A2C("MlpPolicy", vec_env, verbose=1)
ppo_model = PPO("MlpPolicy", vec_env, verbose=1)

# Training loop for competitive setup
num_episodes = 1000  # Number of episodes for training
for episode in range(num_episodes):
    obs = vec_env.reset()
    # print(obs)
    obs = np.array(obs[0])
    done = False
    
    while not done:
        # A2C Agent Step (player_0)
        action_a2c, _ = a2c_model.predict(obs)
        obs, rewards = vec_env.step(action_a2c)
        
        # PPO Agent Step (player_1)
        action_ppo, _ = ppo_model.predict(obs)
        obs, rewards, dones = vec_env.step(action_ppo)
        
        # Check if the episode is done
        done = all(dones.values())
    
    # Optionally print episode rewards to monitor progress
    print(f"Episode {episode + 1} ended with rewards: A2C - {rewards['player_0']}, PPO - {rewards['player_1']}")

    # Update both models after each episode
    a2c_model.learn(total_timesteps=100)
    ppo_model.learn(total_timesteps=100)

# Save the trained models
a2c_model.save("a2c_rps_competitive")
ppo_model.save("ppo_rps_competitive")
