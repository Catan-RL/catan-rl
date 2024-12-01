from ray.rllib.env.multi_agent_env import MultiAgentEnv
import time
from typing import Any

# pettingzoo 1.23.0
from catan_env.catan_env import PettingZooCatanEnv

class RLlib_Catan(MultiAgentEnv):

    def __init__(self, env_config: dict[str,Any]):
        env = PettingZooCatanEnv(**env_config)

        self.env = env
        
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        self.observation_space = env.observation_spaces[self.agents[0]]
        self.action_space = env.action_spaces[self.agents[0]]

        self.env_config = env_config

    def reset(self):
        self.env.reset()

        obs = {agent: self.env.observe(agent) for agent in self.agents}
        return obs

    def step(self, action_dict):
        self.env.step(action_dict)

        observations = self.env.observation_spaces
        rewards = self.env._cumulative_rewards
        info = self.env.infos

        truncations = self.env.truncations
        terminations = self.env.terminations

        dones = {
            truncations[agent] or terminations[agent]
            for agent in self.agents
        }

        return observations, rewards, dones, info
        

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            # "episode_limit": 25,
        }
        return env_info
