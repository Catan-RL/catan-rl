import time
from typing import Any

import gym.spaces
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as GymnasiumDict
from gymnasium.spaces import Discrete

# from gymnasium.spaces import Dict as GymDict
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# pettingzoo 1.23.0
from catan_env.catan_env import PettingZooCatanEnv

REGISTRY = {}
REGISTRY["catan_scenario"] = PettingZooCatanEnv

policy_mapping_dict = {
    "catan_scenario": {
        "description": "four players competative",
        "team_prefix": ("white_", "blue_", "orange_", "red_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


def convert_gymnasium_dict_to_gym_dict(gymnasium_dict):
    """
    Convert a gymnasium.spaces.Dict to a gym.spaces.Dict.
    """
    if not isinstance(gymnasium_dict, GymnasiumDict):
        raise ValueError("Input must be a gymnasium.spaces.Dict")

    # Recursively convert the spaces
    converted_dict = {
        key: convert_space(value)
        for key, value in gymnasium_dict.spaces.items()
    }

    return gym.spaces.Dict(converted_dict)


def convert_space(space):
    """
    Convert individual spaces from gymnasium to gym.
    Handles nested spaces recursively if needed.
    """
    if isinstance(space, GymnasiumDict):
        return convert_gymnasium_dict_to_gym_dict(space)
    elif isinstance(space, Box):
        return gym.spaces.Box(
            low=space.low, high=space.high, dtype=space.dtype
        )
    elif isinstance(space, Discrete):
        return gym.spaces.Discrete(n=space.n)
    else:
        raise NotImplementedError(
            f"Conversion for {type(space)} is not implemented"
        )


class RLlib_Catan(MultiAgentEnv):

    def __init__(self, env_config: dict[str, Any]):

        map = env_config["map_name"]
        env_config.pop("map_name", None)

        print("map", map)
        print("env_config", env_config)
        self.env = REGISTRY[map](**env_config)

        colors = ["white", "blue", "orange", "red"]
        self.agent_map = {
            f"{colors[i]}_0": agent for i, agent in enumerate(self.env.agents)
        }
        self.agents = list(self.agent_map.keys())
        self.num_agents = len(self.agents)

        # includes action mask, which we throw away
        full_obs = self.env.observation_spaces[
            list(self.agent_map.values())[0]
        ]
        full_obs = convert_space(full_obs)

        print("agents", self.agents)

        self.observation_space = gym.spaces.Dict(
            {"obs": full_obs["observation"]}
        )
        self.action_space = convert_space(
            self.env.action_spaces[list(self.agent_map.values())[0]]
        )

        print("observation_space", self.observation_space)

        env_config["map_name"] = map
        self.env_config = env_config

        # print("obs space shape")
        # print(self.observation_space.spaces["obs"].shape)
        # import numpy as np
        # print(np.product(self.observation_space.spaces["obs"].shape))
        # print("action type", type(self.action_space))
        # print(self.action_space)

    def reset(self):
        self.env.reset()

        # obs = {agent: self.env.observe(agent) for agent in self.agents}
        obs = {}
        for agent_id, agent in self.agent_map.items():
            # obs[agent_id] = self.env.observe(agent)["observation"]
            obs[agent_id] = {"obs": self.env.observe(agent)["observation"]}

        return obs

    def step(self, action_dict):

        for agent_id, action in action_dict.items():

            # # get action mask
            # agent = self.agent_map[agent_id]
            # mask = self.env.observe(agent)["action_mask"]
            #
            # print("action", action)
            # print("mask", mask)
            #
            # masked_action = np.bitwise_and(action, mask)
            #
            # print("masked_action", masked_action)
            # print(any(masked_action != action))
            #
            self.env.step(action)

        # observations = self.env.observation_spaces
        observations = {}
        for agent_id, agent in self.agent_map.items():
            # observations[agent_id] = self.env.observe(agent)["observation"]
            observations[agent_id] = {
                "obs": self.env.observe(agent)["observation"]
            }

        rewards = self.env._cumulative_rewards

        infos = {}
        agent_infos = self.env.infos

        for agent_id, agent in self.agent_map.items():
            infos[agent_id] = agent_infos[agent]

        truncations = self.env.truncations
        terminations = self.env.terminations

        dones = {
            agent_id: (truncations[agent] or terminations[agent])
            for agent_id, agent in self.agent_map.items()
        }
        dones["__all__"] = all(dones.values())

        return observations, rewards, dones, infos

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


if __name__ == "__main__":
    # register new env
    ENV_REGISTRY["catan"] = PettingZooCatanEnv

    env = marl.make_env(environment_name="catanEnv", map_name="catan")
    # env[0].render()
    # input()

    algo = marl.algos.mappo(hyperparam_source="test")
    model = marl.build_model(
        env,
        algo,
        {"core_arch": "mlp", "encode_layer": "8-8", "num_outputs": 176},
    )
    algo.fit(
        env,
        model,
        stop={"training_iteration": 100},
        local_mode=True,
        num_gpus=0,
        num_workers=2,
        share_policy="group",
        checkpoint_end=False,
    )

    # # initialize algorithm with appointed hyper-parameters
    # mappo = marl.algos.mappo(hyperparam_source="mpe")
    # # build agent model based on env + algorithms + user preference
    # model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})
    # # start training
    # mappo.fit(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=100, share_policy="group")
