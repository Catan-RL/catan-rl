import copy
import numpy as np
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
from catan_env.game.enums import PlayerId, ActionTypes
from catan_env.game.game import Game
import gymnasium.spaces as spaces


class PettingZooCatanEnv(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, interactive=False, debug_mode=False, policies=None):
        super().__init__()
        self.game = Game(interactive=interactive, debug_mode=debug_mode, policies=policies)
        self.possible_agents = [PlayerId.White, PlayerId.Red, PlayerId.Blue, PlayerId.Orange]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.action_spaces = {agent: self._create_action_space() for agent in self.possible_agents}
        self.observation_spaces = {agent: self._create_observation_space() for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.reset()

    def _create_action_space(self):
        # Define the action space for each agent
        return spaces.Dict({
            'type': spaces.Discrete(len(ActionTypes)),
            'corner': spaces.Discrete(54),
            'edge': spaces.Discrete(72),
            'tile': spaces.Discrete(19),
            'target': spaces.Discrete(4)
        })

    def _create_observation_space(self):
        # Define the observation space for each agent
        return spaces.Dict({
            'resources': spaces.Box(low=0, high=20, shape=(5,), dtype=np.float32),
            'victory_points': spaces.Discrete(11),
            'buildings': spaces.MultiBinary(54),
            'roads': spaces.MultiBinary(72),
            'hidden_cards': spaces.MultiDiscrete([5, 5, 5, 5, 5]),
            'visible_cards': spaces.MultiDiscrete([5, 5, 5, 5, 5])
        })

    def reset(self, seed=None, options=None):
        self.game.reset()
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        if self.game.players_need_to_discard:
            current_player = self.game.players[self.game.players_to_discard[0]]
        elif self.game.must_respond_to_trade:
            current_player = self.game.players[self.game.proposed_trade['target_player']]
        else:
            current_player = self.game.players[agent]
        observation = self._get_observation(current_player)
        return self._flatten_observation(observation)
    
    def _get_observation(self, player):
        obs = {
            'resources': np.array(list(player.resources.values()), dtype=np.float32),
            'victory_points': player.victory_points,
            'buildings': np.array(player.buildings, dtype=np.float32),
            'roads': np.array(player.roads, dtype=np.float32),
            'hidden_cards': np.array(list(player.hidden_cards.values()), dtype=np.float32),
            'visible_cards': np.array(list(player.visible_cards.values()), dtype=np.float32)
        }
        return obs

    def _flatten_observation(self, observation):
        flattened_obs = []
        for key, value in observation.items():
            if isinstance(value, (np.ndarray, list)):
                flattened_obs.extend(value)
            else:
                flattened_obs.append(value)
        return np.array(flattened_obs, dtype=np.float32)

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        translated_action = self._translate_action(action)
        valid_action, error_message = self.game.validate_action(translated_action)
        if not valid_action:
            raise ValueError(f"Invalid action: {error_message}")

        message = self.game.apply_action(translated_action)
        self._update_rewards()
        self._update_dones()
        self._update_infos(message)

        self.agent_selection = self._agent_selector.next()

    def _translate_action(self, action):
        translated_action = {'type': action['type']}
        if 'corner' in action:
            translated_action['corner'] = action['corner']
        elif 'edge' in action:
            translated_action['edge'] = action['edge']
        elif 'tile' in action:
            translated_action['tile'] = action['tile']
        elif 'target' in action:
            translated_action['target'] = action['target']
        return translated_action
    
    def _update_rewards(self):
        for player_id, player in self.game.players.items():
            self.rewards[player_id] = player.victory_points
            self._cumulative_rewards[player_id] = player.victory_points

    def _update_dones(self):
        for player_id, player in self.game.players.items():
            if player.victory_points >= 10:
                self.dones[player_id] = True
                self.terminations[player_id] = True

    def _update_infos(self, message):
        for agent in self.agents:
            self.infos[agent]['log'] = message

    def render(self, mode='human'):
        self.game.render()

    def last(self, observe=True):
        agent = self.agent_selection
        if observe:
            return self.observe(agent), self._cumulative_rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]
        else:
            return None, self._cumulative_rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]
