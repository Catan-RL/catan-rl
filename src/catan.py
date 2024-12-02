from ray.rllib.env.multi_agent_env import MultiAgentEnv
import time
from typing import Any

# pettingzoo 1.23.0
from catan_env.catan_env39 import PettingZooCatanEnv
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

REGISTRY = {}
REGISTRY["catan_scenario"] = PettingZooCatanEnv

policy_mapping_dict = {
    "catan_scenario": {
        "description": "four players competative",
        "team_prefix": ("red_", "blue_", "yellow_", "green_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

class RLlib_Catan(MultiAgentEnv):

    def __init__(self, env_config: dict[str,Any]):

        map = env_config["map_name"]
        env_config.pop("map_name", None)

        self.env = REGISTRY[map](**env_config)

        # env = PettingZooCatanEnv(**env_config)

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
            "episode_limit": 25,
            "policy_mapping_info": policy_mapping_dict,
        }
        return env_info
    
if __name__ == '__main__':
    # register new env
    ENV_REGISTRY["catan"] = PettingZooCatanEnv

    env = marl.make_env(environment_name="catanEnv", map_name="catan")
