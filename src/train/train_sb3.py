from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3 import A2C, PPO
from supersuit import concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1

from catan_env.catan_env import PettingZooCatanEnv

env = PettingZooCatanEnv()

# a2c_model = A2C("MlpPolicy", vec_env, verbose=1)
ppo_model = PPO("MlpPolicy", env, verbose=1)

num_episodes = 10

for episode in range(num_episodes):
    obs = vec_env.reset()
    done = False

    while not done:
        # action_a2c, _ = a2c_model.predict(obs[0])

        print(obs)
        action_ppo, _ = ppo_model.predict(obs[1])

        combined_actions = [action_a2c, action_ppo]
        observations, rewards, terminations, truncations = vec_env.step(
            combined_actions
        )
        obs = observations
        done = any(terminations) or any(truncations)
    print(f"Episode {episode + 1} ended with rewards: {rewards}")
    a2c_model.learn(total_timesteps=100)
    ppo_model.learn(total_timesteps=100)
