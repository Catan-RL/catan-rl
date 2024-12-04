from __future__ import annotations

import functools
from collections import Counter, OrderedDict
from typing import Any

import gymnasium.spaces as spaces
import numpy as np
from numpy import typing as npt
from pettingzoo.utils.env import AECEnv

from catan_env.game.components.buildings import Building
from catan_env.game.enums import (
    ActionTypes,
    BuildingType,
    DevelopmentCard,
    PlayerId,
    Resource,
)
from catan_env.game.game import Game

N_CORNERS = 54
N_EDGES = 72
N_TILES = 19

# TODO:
# - set up logging
# - implement rendering
# - debug stuck game with > 100k steps without ending


class PettingZooCatanEnv(AECEnv):
    metadata = {"render.modes": ["human"], "name": "catan"}

    def __init__(
        self,
        interactive: bool = False,
        debug_mode: bool = False,
        policies: None = None,
        num_players: int = 4,
        enable_dev_cards: bool = True,
        max_actions_per_turn: int = 10,
        **kwargs,
    ):
        super().__init__()
        print("Got extra kwards:", kwargs)

        self.enable_dev_cards: bool = enable_dev_cards
        # TODO: replace max actions per turn with max number of turns total
        self.max_actions_per_turn: int = max_actions_per_turn

        self.game: Game = Game(
            interactive=interactive, debug_mode=debug_mode, policies=policies
        )

        self.possible_agents: list[PlayerId] = list(self.game.players.keys())[
            :num_players
        ]
        self.num_max_agents: int = len(self.possible_agents)

        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.game.reset()

        self.agents: list[PlayerId] = self.possible_agents.copy()
        self.agent_selection: PlayerId = self.game.players_go

        # termination is the natural game end for the player
        self.terminations = {agent: False for agent in self.agents}
        # truncation is us forcing the game to end for that player
        self.truncations = {agent: False for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {"log": ""} for agent in self.agents}

        # self.observation_spaces = {
        #     agent: self._get_obs_space_with_mask(agent)
        #     for agent in self.agents
        # }
        # self.observation_spaces = {
        #     agent: self._get_flat_obs_space(agent) for agent in self.agents
        # }
        # self.action_spaces = {
        #     agent: self._get_action_space(agent)
        #     # agent: self._get_flat_action_space(agent)
        #     for agent in self.agents
        # }

        self.observation_spaces = {
            agent: self._get_obs_space_with_multibinary_mask(agent)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: self._get_action_space_as_single_multibinary(agent)
            for agent in self.agents
        }

        self._player_vps: dict[PlayerId, int] = {
            agent: 0 for agent in self.agents
        }

        # print(f"Restting game with {self.num_max_agents} players")
        # print(
        #     "Observation space as follows: each agent has observation array of shape: "
        #     f"{spaces.flatdim(self._get_flat_obs_space(self.agents[0]))}\n"
        #     f"and an action mask and action space of shape: {spaces.flatdim(self._get_flat_action_space(self.agents[0]))}\n"
        # )

    def render(self) -> None:
        self.game.render()

    def close(self) -> None:
        pass

    def step(self, action: npt.NDArray, do_mask: bool = False) -> None:
        # print("Agent order:", self.game.player_order)
        # print(
        #     f"Agent: {self.agent_selection}, Last Agent: {self._is_last_agent()}"
        # )
        # print(f"Step Rewards: {self.rewards}")
        # print(f"Cumulative Rewards: {self._cumulative_rewards}")

        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(None)
            return

        # unflatten the action, if needed
        if isinstance(action, np.ndarray):
            action = self._unflatten_action(action)
        assert isinstance(action, tuple)

        # apply mask if needed
        if do_mask:
            # mask = self.observe(self.agent_selection)["action_mask"]
            # mask = spaces.flatten(
            #     self._get_action_space(self.agent_selection),
            #     tuple(self._get_action_mask(self.agent_selection)),
            # )
            # masked_action = np.bitwise_and(action, mask)
            # print("did mask work?", (action != masked_action).any())

            mask = tuple(self._get_action_mask(self.agent_selection))

            masked_action = []
            for i, head in enumerate(action):
                masked_action.append(np.bitwise_and(head, mask[i]))
            action = tuple(masked_action)

            print(f"masked action for player {self.agent_selection}:")

        # take the action
        translated_action = self._translate_action(action)
        # print("Action:", translated_action)
        # print("Untranslated Action:", action)
        # print(self.game.players)

        # valid_action, error = self.game.validate_action(translated_action)
        valid = self.game.validate_action(translated_action)
        if isinstance(valid, tuple):
            valid_action, error = valid
        else:
            valid_action = valid
            error = None
        if valid_action:
            message = self.game.apply_action(translated_action)
            print("Applied valid action:", translated_action)
        else:
            message = (
                f"Invalid action {translated_action}\nResulted in: {error}"
            )
            print(message)

            # raise RuntimeError(
            #     f"Invalid action {translated_action}\nResulted in: {error}"
            # )

        self.infos[self.agent_selection]["log"] = message

        # compute rewards
        if self._is_last_agent():
            self._update_rewards(translated_action, valid_action)
        else:
            # clear rewards from previous step
            self._clear_rewards()

        self._cumulative_rewards[self.agent_selection] = 0
        self._accumulate_rewards()

        # update vps
        self._player_vps = {
            agent: self.game.players[agent].victory_points
            for agent in self.agents
        }

        if self._is_game_over():
            self.terminations = {agent: True for agent in self.agents}

        # update agent selection, if applicable
        # note that the same agent might keep control for some time
        self._update_agent_selection()

    def _update_agent_selection(self) -> None:

        # if a 7 was rolled, the game goes into 'discard' mode
        if self.game.players_need_to_discard:
            self.agent_selection = self.game.players_to_discard[0]
        else:
            self.agent_selection = self.game.players_go

    def observe(self, agent: PlayerId) -> dict[str, Any]:
        this_agent_space = (
            *self._get_public_player_features(agent),
            self._get_unplayed_dev_cards_features(agent),
        )

        # we must use possible agents to ensure the observation space doesn't
        # shrink while the game is ending and final rewards are being collected, causing an error
        other_agent_spaces = {
            str(self._get_agent_rel_pos(agent, agent_id)): (
                *self._get_public_player_features(agent_id),
                self._get_hidden_dev_card_count_features(agent_id),
            )
            for agent_id in self.possible_agents
            if agent_id != agent
        }
        other_agent_spaces = dict(sorted(other_agent_spaces.items()))
        if len(other_agent_spaces) != self.num_max_agents - 1:
            print(self.agents)
        assert (
            len(other_agent_spaces) == self.num_max_agents - 1
        ), "ERROR: Incorrect number of other agents"

        unflat_obs = OrderedDict(
            {
                "tile_features": self._get_tile_features(),
                "corner_features": self._get_corner_features(agent),
                "edge_features": self._get_edge_features(agent),
                "bank_features": self._get_bank_features(),
                "this_agent": this_agent_space,
                "other_agents": other_agent_spaces,
            }
        )

        if not self._get_obs_space(agent).contains(unflat_obs):
            self.print_obs(unflat_obs)  # type: ignore
            print("ERROR: inner observation is not in observation space")

        flat_obs = spaces.flatten(self._get_obs_space(agent), unflat_obs)

        # action_mask = tuple(self._get_action_mask(agent))
        # action_mask = spaces.flatten(
        #     self._get_action_space(agent), tuple(self._get_action_mask(agent))
        # )

        action_mask = self._get_action_mask_multibinary(agent)
        obs = OrderedDict(
            {
                "observation": flat_obs,
                "action_mask": action_mask,
            }
        )
        # obs = OrderedDict({"observation": flat_obs})

        # if not self.observation_space(agent).contains(obs):
        #     self.print_obs(obs)  # type: ignore
        #     raise ValueError("ERROR: Observation is not in observation space")

        return obs
        # return flat_obs

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: PlayerId) -> spaces.Space:
        return self.action_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: PlayerId) -> spaces.Space:
        return self.observation_spaces[agent]

    def _is_last_agent(self) -> bool:
        # we are only done if we aren't in discard mode and have gone through 1 full cycle
        return (
            not self.game.players_need_to_discard
            and self.agent_selection == self.game.player_order[-1]
        )

    def _get_obs_space_with_mask(self, agent: PlayerId) -> spaces.Space:
        return spaces.Dict(
            {
                "observation": self._get_flat_obs_space(agent),
                "action_mask": self._get_action_space(agent),
                # "action_mask": self._get_flat_action_space(agent),
            }
        )

    def _get_obs_space_with_multibinary_mask(
        self, agent: PlayerId
    ) -> spaces.Space:
        return spaces.Dict(
            {
                "observation": self._get_flat_obs_space(agent),
                "action_mask": self._get_action_space_as_single_multibinary(
                    agent
                ),
            }
        )

    def _get_flat_action_space(self, _agent: PlayerId) -> spaces.Space:
        return spaces.flatten_space(self._get_action_space(_agent))

    def _unflatten_action(self, action: np.ndarray) -> list[np.ndarray]:
        return spaces.unflatten(
            self._get_action_space(self.agent_selection), action
        )

    def _get_action_space(self, _agent: PlayerId) -> spaces.Space:
        """

        There are mutliple actions heads.
        Head 1: Action Type
            1 hot of length: len(ActionTypes.in_use()

        Head 2: Tile
            Corresponds to MoveRobber
            1 hot of length N_TILES

        Head 3: Corner
            Corresponds to PlaceSettlement, UpgradeToCity
            1 hot of length N_CORNERS

        Head 4: Edge
            Corresponds to PlaceRoad
            1 hot of length N_EDGES + 1
            the last, extra value is an empty road, signifying skipping the road placement
            only allowed to skip when in road building mode after playing a road building dev card
            and there are no more valid roads to place

        Head 5: Development Card
            Corresponds to PlayDevelopmentCard
            1 hot of length len(DevelopmentCard)

        Head 6: Resource
            Corresponds to ExchangeResource (port trade), DiscardResource, and some PlayDevelopmentCard
            2 one-hots of shape (2, len(Resource.non_empty()))
            the first is the first resource; these actions: DiscardResource, PlayDevelopmentCard.Monopoly
            the second is only used for actions that require two resources: ExchangeResource, PlayDevelopmentCard.YearOfPlenty

        Head 7: Player
            Corresponds to StealResource
            Note that 0 is this player, 1 is next, 2 is next next, etc.
            1 hot of length num_max_agents
        """
        head_1 = spaces.MultiBinary(len(ActionTypes.in_use()))
        head_2 = spaces.MultiBinary(N_TILES)
        head_3 = spaces.MultiBinary(N_CORNERS)
        head_4 = spaces.MultiBinary(N_EDGES + 1)
        head_5 = spaces.MultiBinary(len(DevelopmentCard))
        head_6 = spaces.MultiBinary((2, len(Resource.non_empty())))
        head_7 = spaces.MultiBinary(self.num_max_agents)

        action_space = spaces.Tuple(
            [head_1, head_2, head_3, head_4, head_5, head_6, head_7]
        )
        return action_space

    def _get_action_space_as_single_multibinary(
        self, _agent: PlayerId
    ) -> spaces.Space:
        """
        take each action head from a normal action space and concat them
        """

        action_space = spaces.MultiBinary(
            len(ActionTypes.in_use())
            + N_TILES
            + N_CORNERS
            + N_EDGES
            + 1
            + len(DevelopmentCard)
            + 2 * len(Resource.non_empty())
            + self.num_max_agents
        )

        return action_space

    def _get_flat_obs_space(self, _agent: PlayerId) -> spaces.Space:
        return spaces.flatten_space(self._get_obs_space(_agent))

    def _get_obs_space(self, _agent: PlayerId) -> spaces.Space:
        """
        tile features (19 * (1 + 11 + 6 + 6 * (3 + num_max_agents))):
            per tile (19 total):
                contains rober - T/F
                value (number of tile) - [2, 12]
                resource type - len(Resource), 6 total including None

                per corner (6 total):
                    building type - len(BuildingType), 3 total including None
                    player - len(PlayerId), num_max_agents total

        bank resources:
            len(Resource) + 1 for # dev cards left if true
            each value is:
                0, 1, 2: n # of resources, 3: 3-5 # of resources, 4: 6-8 # of resources, 5: 8+ # of resources

        public player features:
            does the agent need to discard cards (after robber) - T/F
            has the longest road - T/F
            length of this player's longest road - [0, 15] (number of roads is limited to 15 per player)
            has the largest army - T/F
            size of this player's army - [0, 14] (number of knights is limited to 14 knight cards, maybe lower if needed)
            agent's VPs - [2, 10] (always start with two settlements, so 2 VPs)
            which harbor this player has - len(Resource) + 1 (3:1 harbor)
            current resources:
                len(Resource) -1: from Brick to Wheat (ignore empty)
                IMPORTANT: 0, 1, ..., 7 are the exact number of cards, 8 represents 8+ cards of this resource

            resource production:
                a 2D array of len(Resource) - 1 x 11 (we ignore the empty resource)
                the rows are the resource type, columns are the tile values (roll # that results in that resource)
                the value in each cell is the number of that resource granted for that roll; 1 for a single settlement, 2 for a city, etc. for more settlements/cities
                ASSUMPTION: 11 is max number of one resource per player per role
                    max 5 settlements and 4 cities per player
                    max 4 cities, 1 settlement before winning at 10 VP
                    up to 2 tiles with same resource type can have the same roll value; assume they are adjacent
                    up to 3 cities/settlements can be on a tile
                    place 3 cites on one tile, 1 city and 1 settlement on the other, such that they share a city
                    thus, one roll produces: 4 (cities) * 2 + 1 (double counted city) * 2 + 1 (settlement) * 1 = 11 resources

            counts of played dev cards
                list len(DevelopmentCard), up to 14 knights, 5 VPs, and 2 of each remaining card

        for this agent:
            public features
            counts of unplayed dev cards
                same as played dev cards

        for other agents (num_max_agents - 1):
            public features
            count of hidden dev cards - can be [0, 25]
        """

        # tile_features is an (N_TILES, 3) array where:
        # - Column 0 represents whether the tile contains a robber (0 = No, 1 = Yes).
        # - Column 1 represents the tile value (integer range: 2 to 12).
        # - Column 2 represents the resource type (integer range: min(Resource) to max(Resource)).
        tile_features = spaces.Box(
            low=np.array([[0, 2, min(Resource)]] * N_TILES, dtype=np.int8),
            high=np.array([[1, 12, max(Resource)]] * N_TILES, dtype=np.int8),
            dtype=np.int8,
        )

        # corner_features is an (N_CORNERS, 2) array where:
        # - Column 0 represents building type (0 = None, 1...len(BuildingType)).
        # - Column 1 represents the owner (0 = None, 1...self.num_max_agents).
        corner_features = spaces.Box(
            low=np.array([[0, 0]] * N_CORNERS, dtype=np.int8),
            high=np.array(
                [[len(BuildingType) + 1, self.num_max_agents + 1]] * N_CORNERS,
                dtype=np.int8,
            ),
            dtype=np.int8,
        )

        # edge_features is an (N_EDGES, 1) array where:
        # - Column 0 represents the road owner (0 = None, 1...self.num_max_agents).
        edge_features = spaces.Box(
            low=np.array([[0]] * N_EDGES, dtype=np.int8),
            high=np.array(
                [[self.num_max_agents + 1]] * N_EDGES, dtype=np.int8
            ),
            dtype=np.int8,
        )

        # bank is a (len(Resource) - 1 + int(self.enable_dev_cards),) array where:
        # - Each element represents the quantity of a resource or development card (range: 0 to 5).
        bank = spaces.Box(
            low=0,
            high=5,
            shape=(len(Resource.non_empty()) + int(self.enable_dev_cards), 1),
            dtype=np.int8,
        )

        # public_player_space is a list where:
        # - The first 6 entries represent binary or integer values for player states (e.g., need to discard, longest road).
        # - The next 4 entries are structured as arrays representing harbors, resources, resource production, and played development cards.
        public_player_space: list[spaces.Space] = [
            spaces.MultiBinary(1),  # need to discard
            spaces.MultiBinary(1),  # has longest road
            spaces.Discrete(16),  # len of agent's longest road
            spaces.MultiBinary(1),  # has largest army
            spaces.Discrete(
                self.game.max_dev_cards_by_type[DevelopmentCard.Knight] + 1
            ),  # size of agent's army
            spaces.Discrete(11),  # agent's VPs
            # Harbors: 1 for each resource type (empty is replaced with 3:1 harbor)
            spaces.MultiBinary(len(Resource)),
            # Resources: count of each resource type the player has
            spaces.MultiDiscrete([9] * len(Resource.non_empty())),
            # Resource production: resources x tile roll values (offset [2, 12] to indices [0, 10])
            # resource production
            spaces.Box(
                low=0,  # min number of resources per roll
                high=11,  # max number of resources per roll
                # resources x tile roll values (offset from [2, 12] to indices [0, 10])
                shape=(len(Resource.non_empty()), 11),
                dtype=np.int8,
            ),
            # played dev cards
            spaces.Box(
                low=np.zeros((len(self.game.max_dev_cards_by_type),)),
                high=np.array(list(self.game.max_dev_cards_by_type.values())),
                shape=(len(DevelopmentCard),),
                dtype=np.int8,
            ),
        ]

        unplayed_dev_cards = spaces.Box(
            low=np.zeros((len(self.game.max_dev_cards_by_type),)),
            high=np.array(list(self.game.max_dev_cards_by_type.values())),
            shape=(len(DevelopmentCard),),
            dtype=np.int8,
        )

        this_agent_space = spaces.Tuple(
            public_player_space.copy() + [unplayed_dev_cards]
        )

        other_agent_spaces = spaces.Dict(
            {
                str(rel_pos): spaces.Tuple(
                    public_player_space.copy()
                    +
                    # count of hidden dev cards
                    [spaces.Discrete(len(self.game.development_cards))]
                )
                for rel_pos in range(1, len(self.possible_agents))
            }
        )

        obs = spaces.Dict(
            {
                "tile_features": tile_features,
                "corner_features": corner_features,
                "edge_features": edge_features,
                "bank_features": bank,
                "this_agent": this_agent_space,
                "other_agents": other_agent_spaces,
            }
        )

        return obs

    def _translate_action(self, action: tuple[np.ndarray]) -> dict[str, Any]:
        assert isinstance(action, tuple)

        (
            head_action_type,
            head_tile,
            head_corner,
            head_edge,
            head_dev_card,
            head_resource,
            head_player,
        ) = action

        action_type = ActionTypes(head_action_type.argmax().item())
        translated: dict[str, Any] = {
            "type": action_type,
        }
        if action_type == ActionTypes.MoveRobber:
            translated["tile"] = head_tile.argmax().item()
        elif (
            action_type == ActionTypes.PlaceSettlement
            or action_type == ActionTypes.UpgradeToCity
        ):
            translated["corner"] = head_corner.argmax().item()
        elif action_type == ActionTypes.PlaceRoad:
            edge = head_edge.argmax().item()

            # the last edge is the empty edge, which means skipping the road placement
            if edge == N_EDGES:
                edge = None
            translated["edge"] = edge
        elif action_type == ActionTypes.PlayDevelopmentCard:
            card_type = DevelopmentCard(head_dev_card.argmax().item())
            translated["card"] = card_type

            if card_type == DevelopmentCard.YearOfPlenty:
                translated["resource_1"] = Resource.from_non_empty(
                    head_resource[0, :].argmax().item()
                )
                translated["resource_2"] = Resource.from_non_empty(
                    head_resource[1, :].argmax().item()
                )
            elif card_type == DevelopmentCard.Monopoly:
                translated["resource"] = Resource.from_non_empty(
                    head_resource[0, :].argmax().item()
                )
        elif action_type == ActionTypes.StealResource:
            rel_pos = head_player.argmax().item()
            translated["target"] = self._get_agent_from_rel_pos(
                self.agent_selection, rel_pos
            )
        elif action_type == ActionTypes.DiscardResource:
            translated["resources"] = [
                Resource.from_non_empty(head_resource[0, :].argmax().item())
            ]

        elif action_type == ActionTypes.ExchangeResource:
            translated["desired_resource"] = Resource.from_non_empty(
                head_resource[0, :].argmax().item()
            )
            translated["trading_resource"] = Resource.from_non_empty(
                head_resource[1, :].argmax().item()
            )
            translated["exchange_rate"] = self._get_exchange_rate(
                self.agent_selection, translated["trading_resource"]
            )

        else:
            # nothing special for the rest
            pass

        return translated

    def _is_game_over(self) -> bool:
        return any(
            player.victory_points >= 10
            for player in self.game.players.values()
        )

    def _update_rewards(
        self, action: dict[str, Any], valid_action: bool
    ) -> None:
        step_rewards = {agent: 0.0 for agent in self.agents}

        # check for a winner
        winner: PlayerId | None = None
        for player in self.game.players.values():
            if player.victory_points >= 10:
                winner = player.id
                break

        # reward the winner, punish the losers
        if winner is not None:
            step_rewards[winner] = 500
            for agent in self.agents:
                if agent != winner:
                    step_rewards[agent] = -100

        curr_player = self.game.players[self.agent_selection]
        if valid_action:
            curr_player_reward = -0.05

            # reward for increasing VPs, do not punish as harshly for losing VPs
            if curr_player.victory_points > self._player_vps[curr_player.id]:
                curr_player_reward += 10 * (
                    curr_player.victory_points
                    - self._player_vps[curr_player.id]
                )
            else:
                # TODO: seems to be a bug thinking lost VPs when they shouldn't be
                curr_player_reward -= 5

            if action["type"] == ActionTypes.PlayDevelopmentCard:
                curr_player_reward += 5
            elif action["type"] == ActionTypes.MoveRobber:
                curr_player_reward += 1
            elif action["type"] == ActionTypes.StealResource:
                curr_player_reward += 0.5
            elif action["type"] == ActionTypes.DiscardResource:
                curr_player_reward -= 0.25
            elif action["type"] == ActionTypes.UpgradeToCity:
                curr_player_reward += 3
        else:
            curr_player_reward = -10

        step_rewards[curr_player.id] = curr_player_reward

        self.rewards = step_rewards

    def _get_agent_rel_pos(
        self, curr_agent: PlayerId, target_agent: PlayerId
    ) -> int:
        """Returns an integer representing the relative positional distance from the current agent to the target agent.

        0: the current agent and the target agent are the same
        1: the target agent is the next agent
        2: the target agent is the next next agent
        3: the target agent is the next next next
        etc.


        Args:
            curr_agent: the current agent PlayerId to start from.
            target_agent: the target agent PlayerId to calculate the relative position to.

        Returns:
            An integer representing the relative positional distance, between 0 and len(self.game.player_order) - 1.
        """
        curr_agent_idx = self.game.player_order.index(curr_agent)
        target_idx = self.game.player_order.index(target_agent)

        # TODO: this will break when num_agents != len(self.game.player_order)
        rel_pos = (target_idx - curr_agent_idx) % len(self.game.player_order)

        return rel_pos

    def _get_agent_from_rel_pos(
        self, curr_agent: PlayerId, rel_pos: int
    ) -> PlayerId:
        """Returns the PlayerId of the agent that is the given relative positional distance from the current agent."""

        curr_agent_idx = self.game.player_order.index(curr_agent)
        target_idx = (curr_agent_idx + rel_pos) % len(self.game.player_order)

        return self.game.player_order[target_idx]

    def _get_tile_features(self) -> npt.NDArray[np.int8]:
        tile_features: list[list[float]] = []
        for tile in self.game.board.tiles:
            tile_features.append(
                [
                    float(tile.contains_robber),
                    float(tile.value),
                    float(tile.resource),
                ]
            )

        features_arr = np.array(tile_features, dtype=np.int8)

        if features_arr.shape != (N_TILES, 3):
            raise ValueError(
                f"Tile features have the wrong shape: {features_arr.shape}."
                f"Should be ({N_TILES}, 3)."
            )

        return features_arr

    def _get_corner_features(self, agent: PlayerId) -> npt.NDArray[np.int8]:
        corner_features: list[list[float]] = []
        for corner in self.game.board.corners:
            building: Building | None = corner.building

            building_type: BuildingType | int
            if building is not None:
                building_type = building.type
                owner: PlayerId = building.owner

                # add one since 0 is reserved for no owner
                owner_id = self._get_agent_rel_pos(agent, owner) + 1
            else:
                building_type = 0
                owner_id = 0

            corner_features.append([float(building_type), float(owner_id)])

        features_arr = np.array(corner_features, dtype=np.int8)
        if features_arr.shape != (N_CORNERS, 2):
            raise ValueError(
                f"Corner features have the wrong shape: {features_arr.shape}."
                f"Should be ({N_CORNERS}, 2)."
            )
        return features_arr

    def _get_edge_features(self, agent: PlayerId) -> npt.NDArray[np.int8]:
        edge_features: list[list[float]] = []

        for edge in self.game.board.edges:
            road_owner: PlayerId | None = edge.road

            if road_owner is not None:
                # assign 1 to the player who owns the road, 2 to the next agent, etc.
                owner_id = self._get_agent_rel_pos(agent, road_owner) + 1
            else:
                owner_id = 0

            edge_features.append([float(owner_id)])

        features_arr = np.array(edge_features, dtype=np.int8)
        if features_arr.shape != (N_EDGES, 1):
            raise ValueError(
                f"Edge features have the wrong shape: {features_arr.shape}."
                f"Should be ({N_EDGES}, 1)."
            )
        return features_arr

    def _get_bank_features(self) -> npt.NDArray[np.int8]:
        features_arr: npt.NDArray = np.zeros(
            (len(Resource) - 1 + int(self.enable_dev_cards), 1),
            dtype=np.int8,
        )
        for i, resource in enumerate(Resource):
            # we ignore the empty resource, with idx = 0 for the first real resource
            if resource == Resource.Empty:
                continue

            count = self.game.resource_bank[resource]

            if count >= 0 and count <= 2:
                bucketed = count
            elif count >= 3 and count <= 5:
                bucketed = 3
            elif count >= 6 and count <= 8:
                bucketed = 4
            else:
                bucketed = 5

            features_arr[i] = bucketed

        if self.enable_dev_cards:
            dev_card_count = len(self.game.development_cards)

            if dev_card_count >= 0 and dev_card_count <= 2:
                bucketed = dev_card_count
            elif dev_card_count >= 3 and dev_card_count <= 5:
                bucketed = 3
            elif dev_card_count >= 6 and dev_card_count <= 8:
                bucketed = 4
            else:
                bucketed = 5

            features_arr[-1] = bucketed

        if features_arr.shape != (
            len(Resource) - 1 + int(self.enable_dev_cards),
            1,
        ):
            raise ValueError(
                f"Bank features have the wrong shape: {features_arr.shape}."
                f"Should be (len(Resource) - 1 + int(self.enable_dev_cards), 1)."
            )
        return features_arr

    def _get_public_player_features(self, agent: PlayerId) -> tuple[
        npt.NDArray[np.int8],
        npt.NDArray[np.int8],
        np.int64,
        npt.NDArray[np.int8],
        np.int64,
        np.int64,
        npt.NDArray[np.int8],
        npt.NDArray[np.int8],
        npt.NDArray[np.int8],
        npt.NDArray[np.int8],
    ]:
        player = self.game.players[agent]
        needs_to_discard = player in self.game.players_to_discard
        has_longest_road: bool = (
            self.game.longest_road is not None
            and self.game.longest_road["player"] == player.id
        )
        len_player_longest_road: int = self.game.current_longest_path[
            player.id
        ]
        has_largest_army: bool = (
            self.game.largest_army is not None
            and self.game.largest_army["player"] == player.id
        )
        army_size: int = self.game.current_army_size[player.id]
        vp: int = player.victory_points
        # prevent vp from increasing above 10 when game is over to avoid observation space overflows
        vp = min(vp, 10)

        # swap empty with 3:1 harbor
        harbors = np.zeros((len(Resource),), dtype=np.int8)
        for harbor in player.harbours.values():
            # the 3:1 harbor will be index 0 instead of the empty resource
            if harbor.exchange_value == 3:
                harbors[0] = 1
            else:
                harbors[int(harbor.resource)] = 1

        # resources
        resources = np.zeros((len(Resource.non_empty())), dtype=np.int8)
        for res_idx, resource in enumerate(Resource.non_empty()):
            count = player.resources[resource]

            if count >= 0 and count <= 7:
                bucketed = count
            else:
                bucketed = 8

            resources[res_idx] = bucketed

        resource_production = np.zeros(
            (len(Resource.non_empty()), 11), dtype=np.int8
        )

        for tile in self.game.board.tiles:
            resource = tile.resource
            production = 0

            for corner in tile.corners.values():
                if (
                    corner is not None
                    and corner.building is not None
                    and corner.building.owner == player
                ):
                    if corner.building == BuildingType.Settlement:
                        production += 1
                    elif corner.building == BuildingType.City:
                        production += 2

            resource_production[resource.int_non_empty(), tile.value - 2] = (
                production
            )

        played_dev_cards = np.zeros((len(DevelopmentCard),), dtype=np.int8)

        played_dev_card_counts = Counter(player.visible_cards)
        for i, card in enumerate(DevelopmentCard):
            played_dev_cards[i] = played_dev_card_counts[card]

        features = (
            np.array([needs_to_discard], dtype=np.int8),
            np.array([has_longest_road], dtype=np.int8),
            np.int64(len_player_longest_road),
            np.array([has_largest_army], dtype=np.int8),
            np.int64(army_size),
            np.int64(vp),
            harbors,
            resources,
            resource_production,
            played_dev_cards,
        )

        return features

    def _get_unplayed_dev_cards_features(
        self, agent: PlayerId
    ) -> npt.NDArray[np.int8]:
        player = self.game.players[agent]
        unplayed_dev_cards = np.zeros((len(DevelopmentCard),), dtype=np.int8)
        unplayed_dev_card_counts = Counter(player.hidden_cards)
        for i, card in enumerate(DevelopmentCard):
            unplayed_dev_cards[i] = unplayed_dev_card_counts[card]
        return unplayed_dev_cards

    def _get_hidden_dev_card_count_features(self, agent: PlayerId) -> np.int64:
        player = self.game.players[agent]

        return np.int64(len(player.hidden_cards))

    def _get_action_mask_multibinary(
        self, agent: PlayerId
    ) -> npt.NDArray[np.int8]:
        (
            head_action_type,
            head_tile,
            head_corner,
            head_edge,
            head_dev_card,
            head_resource,
            head_player,
        ) = self._get_action_mask(agent)

        return np.concatenate(
            [
                head_action_type,
                head_tile,
                head_corner,
                head_edge,
                head_dev_card,
                head_resource[0],
                head_resource[1],
                head_player,
            ]
        )

    def _get_action_mask(self, agent: PlayerId) -> list[npt.NDArray[np.int8]]:
        """
        Will have a mask of 1s and 0s for each action head, where 1 is allowed and 0 is not.
        """
        player = self.game.players[agent]

        mask_action_type = np.zeros(len(ActionTypes.in_use()), dtype=np.int8)
        mask_tile = np.zeros(N_TILES, dtype=np.int8)
        mask_corner = np.zeros(N_CORNERS, dtype=np.int8)
        mask_edge = np.zeros(N_EDGES + 1, dtype=np.int8)
        mask_dev_card = np.zeros(len(DevelopmentCard), dtype=np.int8)
        mask_resource = np.zeros((2, len(Resource.non_empty())), dtype=np.int8)
        mask_player = np.zeros(self.num_max_agents, dtype=np.int8)

        # if players need to discard, they can only discard
        if self.game.players_need_to_discard:
            mask_action_type[ActionTypes.DiscardResource] = 1.0
            # do not allow selection of resource that are empty
            for i, resource in enumerate(Resource.non_empty()):
                if player.resources[resource] > 0:
                    # use 0 since first dim is the resource to discard
                    mask_resource[0][i] = 1.0

            return [
                mask_action_type,
                mask_tile,
                mask_corner,
                mask_edge,
                mask_dev_card,
                mask_resource,
                mask_player,
            ]

        # setup game phase
        if self.game.initial_placement_phase:
            # if they haven't placed a settlement, or they are placing their second
            if self.game.initial_settlements_placed[player.id] == 0 or (
                self.game.initial_settlements_placed[player.id] == 1
                and self.game.initial_roads_placed[player.id] == 1
            ):
                mask_action_type[ActionTypes.PlaceSettlement] = 1.0
                mask_corner = self._valid_settlement(agent)
            else:
                mask_action_type[ActionTypes.PlaceRoad] = 1.0
                mask_edge = self._valid_road(agent)

            return [
                mask_action_type,
                mask_tile,
                mask_corner,
                mask_edge,
                mask_dev_card,
                mask_resource,
                mask_player,
            ]

        # rest of action types
        done = False

        if self.game.road_building_active[0]:
            # using road building card, can only build roads
            mask_action_type[ActionTypes.PlaceRoad] = 1.0
            mask_edge = self._valid_road(agent)
            done = True

        elif self.game.just_moved_robber:
            mask_action_type[ActionTypes.StealResource] = 1.0
            mask_player = self._valid_steal(agent)
            done = True

        elif not self.game.dice_rolled_this_turn:
            # can only roll or play dev cards
            mask_action_type[ActionTypes.RollDice] = 1.0

            if (
                len(player.hidden_cards) > 0
                and not self.game.played_development_card_this_turn
            ):
                mask_dev_card, mask_resource = self._valid_dev_cards(agent)
            done = True
        elif self.game.actions_this_turn > self.max_actions_per_turn:
            # exceeded max actions, can only end turn
            mask_action_type[ActionTypes.EndTurn] = 1.0
            done = True

        if done:
            return [
                mask_action_type,
                mask_tile,
                mask_corner,
                mask_edge,
                mask_dev_card,
                mask_resource,
                mask_player,
            ]

        # have already rolled the die, allowed to end turn now
        mask_action_type[ActionTypes.EndTurn] = 1.0

        resources = player.resources

        # can place settlement
        if (
            resources[Resource.Wheat] > 0
            and resources[Resource.Sheep] > 0
            and resources[Resource.Wood] > 0
            and resources[Resource.Brick] > 0
            and self.game.building_bank["settlements"][player.id] > 0
        ):
            valid_corners = self._valid_settlement(agent)

            if valid_corners.any():
                mask_action_type[ActionTypes.PlaceSettlement] = 1.0
                mask_corner = np.bitwise_or(mask_corner, valid_corners)

        # upgrade to city
        if (
            resources[Resource.Wheat] >= 2
            and resources[Resource.Ore] >= 3
            and self.game.building_bank["cities"][player.id] > 0
        ):
            valid_corners = self._valid_city(agent)
            if valid_corners.any():
                mask_action_type[ActionTypes.UpgradeToCity] = 1.0
                mask_corner = np.bitwise_or(mask_corner, valid_corners)

        # place road
        if (
            resources[Resource.Wood] > 0
            and resources[Resource.Brick] > 0
            and self.game.road_bank[player.id] > 0
        ):
            valid_edges = self._valid_road(agent)
            if valid_edges.any():
                mask_action_type[ActionTypes.PlaceRoad] = 1.0
                mask_edge = np.bitwise_or(mask_edge, valid_edges)

        # buy dev card
        if (
            resources[Resource.Wheat] > 0
            and resources[Resource.Sheep] > 0
            and resources[Resource.Ore] > 0
            and len(self.game.development_cards_pile) > 0
        ):
            valid_dev_card, valid_dev_resource = self._valid_dev_cards(agent)

            if valid_dev_card.any():
                mask_action_type[ActionTypes.BuyDevelopmentCard] = 1.0
                mask_dev_card = np.bitwise_or(mask_dev_card, valid_dev_card)
                # returns both dims of valid resources
                mask_resource = np.bitwise_or(
                    mask_resource, valid_dev_resource
                )

        # move robber
        if self.game.can_move_robber:
            mask_action_type[ActionTypes.MoveRobber] = 1.0
            valid_tile = self._valid_robber_move(agent)
            mask_tile = np.bitwise_or(mask_tile, valid_tile)

        # port/bank trade
        valid_res_desired, valid_res_trading = self._valid_exchange(agent)

        if valid_res_desired.any() and valid_res_trading.any():
            mask_action_type[ActionTypes.ExchangeResource] = 1.0
            valid_res = np.vstack((valid_res_desired, valid_res_trading))
            mask_resource = np.bitwise_or(mask_resource, valid_res)

        return [
            mask_action_type,
            mask_tile,
            mask_corner,
            mask_edge,
            mask_dev_card,
            mask_resource,
            mask_player,
        ]

    def _valid_settlement(self, agent: PlayerId) -> npt.NDArray[np.int8]:
        player = self.game.players[agent]
        valid_corners = np.zeros(N_CORNERS, dtype=np.int8)
        for i, corner in enumerate(self.game.board.corners):
            if corner.can_place_settlement(
                player.id, initial_placement=self.game.initial_placement_phase
            ):
                valid_corners[i] = 1.0
        return valid_corners

    def _valid_road(self, agent: PlayerId) -> npt.NDArray[np.int8]:
        player = self.game.players[agent]
        valid_edges = np.zeros(N_EDGES + 1, dtype=np.int8)

        after_second_settlement = False
        second_settlement = None
        at_least_one_valid = False

        if (
            self.game.initial_placement_phase
            and self.game.initial_settlements_placed[player.id] == 2
        ):  # if they are placing second road
            after_second_settlement = True
            second_settlement = self.game.initial_second_settlement_corners[
                player.id
            ]

        for i, edge in enumerate(self.game.board.edges):
            if edge.can_place_road(
                player.id,
                after_second_settlement=after_second_settlement,
                second_settlement=second_settlement,
            ):
                valid_edges[i] = 1.0
                at_least_one_valid = True

        if self.game.road_building_active[0] and not at_least_one_valid:
            # allow skipping road placement
            valid_edges[-1] = 1.0

        return valid_edges

    def _valid_city(self, agent: PlayerId) -> npt.NDArray[np.int8]:
        player = self.game.players[agent]
        valid_corners = np.zeros(N_CORNERS, dtype=np.int8)

        for i, corner in enumerate(self.game.board.corners):
            if (
                corner is not None
                and corner.building is not None
                and corner.building.type == BuildingType.Settlement
                and corner.building.owner == player.id
            ):
                valid_corners[i] = 1.0

        return valid_corners

    def _valid_steal(self, agent: PlayerId) -> npt.NDArray[np.int8]:
        valid_players = np.zeros(self.num_max_agents, dtype=np.int8)

        robber_tile = self.game.board.robber_tile
        for corner in robber_tile.corners.values():
            if (
                corner is not None
                and corner.building is not None
                and corner.building.owner != agent
            ):
                rel_pos = self._get_agent_rel_pos(agent, corner.building.owner)
                valid_players[rel_pos] = 1.0

        assert (
            valid_players[0] == 0.0
        ), "Cannot steal from self; something went wrong in masking."
        return valid_players

    def _valid_dev_cards(
        self, agent: PlayerId
    ) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
        player = self.game.players[agent]

        valid_dev_card_type = np.zeros(len(DevelopmentCard), dtype=np.int8)
        valid_dev_resource = np.ones(
            (2, len(Resource.non_empty())), dtype=np.int8
        )

        for i, card in enumerate(DevelopmentCard):
            count = player.hidden_cards.count(card)
            if (
                count > 0
                and self.game.development_cards_bought_this_turn.count(card)
                < count
            ):
                # must have resources left in bank when playing Year of Plenty
                # everything else allowed
                if (
                    card != DevelopmentCard.YearOfPlenty
                    or sum(self.game.resource_bank.values()) > 0
                ):
                    valid_dev_card_type[i] = 1.0

        for i, resource in enumerate(Resource.non_empty()):

            # if YearOfPlenty, need to have resources in bank
            if (
                valid_dev_card_type[DevelopmentCard.YearOfPlenty] == 1.0
                and self.game.resource_bank[resource] <= 0
            ):
                valid_dev_resource[i] = 0.0

        return valid_dev_card_type, valid_dev_resource

    def _valid_robber_move(self, agent: PlayerId) -> npt.NDArray[np.int8]:
        player = self.game.players[agent]
        valid_tiles = np.zeros(N_TILES, dtype=np.int8)

        for i, tile in enumerate(self.game.board.tiles):
            # must have a building on the tile and not any buildings owned by you
            # to move the robber there
            if any(
                (
                    corner is not None
                    and corner.building is not None
                    and corner.building.owner != player
                )
                for corner in tile.corners.values()
            ):
                valid_tiles[i] = 1.0

        return valid_tiles

    def _valid_exchange(
        self, agent: PlayerId
    ) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
        player = self.game.players[agent]
        valid_desired = np.zeros(len(Resource.non_empty()), dtype=np.int8)
        valid_trading = np.zeros(len(Resource.non_empty()), dtype=np.int8)

        # you are only allowed to request/desire a resource if you have:
        # - 4 of one resource
        # - 3/2 of a resource corresponding to a 3:1 or 2:1 port

        # you can desire anything if the bank has enough
        for i, resource in enumerate(Resource.non_empty()):
            if self.game.resource_bank[resource] > 0:
                valid_desired[i] = 1.0

        for i, resource in enumerate(Resource.non_empty()):
            count = player.resources[resource]
            exchange_rate = self._get_exchange_rate(agent, resource)

            if count >= exchange_rate:
                valid_trading[i] = 1.0

        return valid_desired, valid_trading

    def _get_exchange_rate(self, agent: PlayerId, resource: Resource) -> int:
        player = self.game.players[agent]
        best = 4  # exchange with bank
        for harbor in player.harbours.values():
            if harbor.resource is None:
                best = min(best, 3)
            elif harbor.resource == resource:
                best = min(best, 2)
        return best

    @staticmethod
    def print_obs(obs: spaces.Space | npt.NDArray | tuple | dict) -> None:
        if isinstance(obs, np.ndarray):
            if np.ndim(obs) == 0:  # type: ignore
                print(obs)  # type: ignore
            else:
                print(obs.shape)
        elif isinstance(obs, dict):
            for k, v in obs.items():
                print(k)  # type: ignore
                PettingZooCatanEnv.print_obs(v)  # type: ignore
        elif isinstance(obs, tuple):
            print(len(obs))  # type: ignore
            for x in obs:
                print(x.shape)  # type: ignore
        else:
            print("WEIRD", obs)  # type: ignore
