import supersuit as ss
from pettingzoo.classic import rps_v2
from stable_baselines3 import A2C, PPO
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from pettingzoo.utils.conversions import aec_to_parallel
import numpy as np

env = rps_v2.env()

parallel_env = aec_to_parallel(env)

vec_env = pettingzoo_env_to_vec_env_v1(parallel_env)

vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")

a2c_model = A2C("MlpPolicy", vec_env, verbose=1)
ppo_model = PPO("MlpPolicy", vec_env, verbose=1)

num_episodes = 1000  
for episode in range(num_episodes):
    obs = vec_env.reset() 

    done = False

    while not done:
        action_a2c, _ = a2c_model.predict(obs[0])
        action_ppo, _ = ppo_model.predict(obs[1]) 

        combined_actions = [action_a2c, action_ppo]

        observations, rewards, terminations, truncations = vec_env.step(combined_actions)

        obs = observations

        done = np.any(terminations) or np.any(truncations)

    print(f"Episode {episode + 1} ended with rewards: {rewards}")

    a2c_model.learn(total_timesteps=100)
    ppo_model.learn(total_timesteps=100)

a2c_model.save("a2c_rps_competitive")
ppo_model.save("ppo_rps_competitive")
