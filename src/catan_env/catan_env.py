import copy
from collections import Counter
from typing import Any

import gymnasium.spaces as spaces
import numpy as np
from numpy import typing as npt
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AECEnv

from catan_env.game.components.buildings import Building
from catan_env.game.components.player import Player
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


class PettingZooCatanEnv(AECEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        interactive: bool = False,
        debug_mode: bool = False,
        policies: None = None,
        num_players: int = 4,
        enable_dev_cards: bool = True,
    ):
        super().__init__()

        self.enable_dev_cards: bool = enable_dev_cards

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

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {"log": ""} for agent in self.agents}

        self.observation_spaces = {
            agent: self._get_obs_space(agent) for agent in self.agents
        }

        self.action_spaces = {
            agent: self._get_action_space(agent) for agent in self.agents
        }

        self._player_vps: dict[PlayerId, int] = {
            agent: 0 for agent in self.agents
        }

    def step(self, action: np.ndarray) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(None)
            return

        # clear rewards from previous step
        self._clear_rewards()

        # take the action
        translated_action = self._translate_action(action)

        valid_action, error = self.game.validate_action(translated_action)
        if not valid_action:
            raise RuntimeError(f"Invalid action: {error}")

        message = self.game.apply_action(translated_action)
        self.infos[self.agent_selection]["log"] = message

        # compute rewards
        self._compute_rewards(translated_action)
        self._accumulate_rewards()

        # update vps
        self._player_vps = {
            agent: self.game.players[agent].victory_points
            for agent in self.agents
        }

        if self._is_game_over():
            self.terminations = {agent: True for agent in self.agents}
            self.truncations = {agent: True for agent in self.agents}

        # update agent selection
        self.agent_selection = self.game.players_go

    def observe(self, agent: PlayerId) -> dict[str, Any]:
        this_agent_space = self._get_public_player_features(agent)
        this_agent_space.append(self._get_unplayed_dev_cards_features(agent))

        other_agent_spaces = {
            str(agent_id): self._get_public_player_features(agent_id)
            + [self._get_hidden_dev_card_count_features(agent_id)]
            for agent_id in self.agents
            if agent_id != agent
        }

        obs = {
            "tile_features": self._get_tile_features(),
            "corner_features": self._get_corner_features(agent),
            "edge_features": self._get_edge_features(agent),
            "bank_features": self._get_bank_features(),
            "this_agent": this_agent_space,
            "other_agents": other_agent_spaces,
        }
        return obs

    def action_space(self, agent: PlayerId) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: PlayerId) -> spaces.Space:
        return self.observation_spaces[agent]

    def _get_action_space(self, _agent: PlayerId) -> spaces.Space:
        # TODO: figuer this out later
        return spaces.Discrete(len(ActionTypes))

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

        tile_features = spaces.Tuple(
            # TODO: do I need to make a deepcopy of this
            [
                # contains robber: True (1) or False (0)
                spaces.MultiBinary(1),
                spaces.Discrete(11, start=2),  # tile value: [2, 12]
                spaces.Discrete(len(Resource)),  # resource type of tile
            ]
            * N_TILES
        )

        corner_features = spaces.Tuple(
            [
                # building type: settlement, city, or None
                spaces.Discrete(len(BuildingType) + 1),
                spaces.Discrete(self.num_max_agents + 1),  # owner
            ]
            * N_CORNERS
        )

        edge_features = spaces.Tuple(
            [
                # road owner, or None for no road
                spaces.Discrete(self.num_max_agents + 1)
            ]
            * N_EDGES
        )

        # bank; subtract 1 for empty
        bank = spaces.MultiDiscrete(
            [6] * (len(Resource) - 1 + int(self.enable_dev_cards))
        )

        public_player_space: list[spaces.Space] = [
            spaces.MultiBinary(1),  # need to discard
            spaces.MultiBinary(1),  # has longest road
            spaces.Discrete(15),  # len of agent's longest road
            spaces.MultiBinary(1),  # has largest army
            spaces.Discrete(14),  # size of agent's army
            spaces.Discrete(9, start=2),  # agent's VPs
            # harbors - replace empty with 3:1
            spaces.MultiBinary(len(Resource)),
            spaces.MultiDiscrete([9] * len(Resource.non_empty())),  # resources
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
                str(agent_id): spaces.Tuple(
                    public_player_space.copy()
                    +
                    # count of hidden dev cards
                    [spaces.Discrete(len(self.game.development_cards))]
                )
                for agent_id in self.possible_agents
            }
        )

        obs = {
            "tile_features": tile_features,
            "corner_features": corner_features,
            "edge_features": edge_features,
            "bank_features": bank,
            # TODO: should I key this with this agent's id?
            "this_agent": this_agent_space,
            "other_agents": other_agent_spaces,
        }

        return spaces.Dict(obs)

    def _translate_action(self, action: np.ndarray) -> dict[str, Any]:
        action_type = action[0]
        translated = {
            "type": action_type,
        }

        match action_type:
            case ActionTypes.PlaceSettlement | ActionTypes.UpgradeToCity:
                translated["corner"] = action[1]
            case ActionTypes.PlaceRoad:
                # TODO: figure out why there is a dummy edge needed
                translated["edge"] = (
                    action[2] if action[1] != N_EDGES else None
                )
            case ActionTypes.PlayDevelopmentCard:
                card_type = DevelopmentCard(action[4])
                translated["card"] = card_type

                match card_type:
                    # TODO: figure this portion too, bc the old wrapper does some weird head out stuff
                    case DevelopmentCard.YearOfPlenty:
                        translated["resource_1"] = action[5]
                        translated["resource_2"] = action[6]
                    case DevelopmentCard.Monopoly:
                        translated["resource"] = action[5]
                    case _:
                        pass

            case ActionTypes.MoveRobber:
                translated["tile"] = action[3]
            case ActionTypes.StealResource:
                target_player = action[6]
                # TODO: figure out the translation from action to PlayerId
                translated["target"] = target_player
            case ActionTypes.DiscardResource:
                # TODO: figure out the head out stuff here too
                translated["resource"] = action[7]

            # nothing special for the rest:
            # ActionTypes.BuyDevelopmentCard
            # ActionTypes.RollDice
            # ActionTypes.EndTurn
            case _:
                pass

        return translated

    def _is_game_over(self) -> bool:
        return any(
            player.victory_points >= 10
            for player in self.game.players.values()
        )

    def _compute_rewards(self, action: dict[str, Any]) -> None:
        rewards = {agent: 0.0 for agent in self.agents}

        # check for a winner
        winner: PlayerId | None = None
        for player in self.game.players.values():
            if player.victory_points >= 10:
                winner = player.id
                break

        # reward the winner, punish the losers
        if winner is not None:
            rewards[winner] = 500
            for agent in self.agents:
                if agent != winner:
                    rewards[agent] = -100

        curr_player_reward = 0.0
        curr_player = self.game.players[self.agent_selection]

        # reward for increasing VPs, do not punish as harshly for losing VPs
        if curr_player.victory_points > self._player_vps[curr_player.id]:
            curr_player_reward += 10 * (
                curr_player.victory_points - self._player_vps[curr_player.id]
            )
        else:
            curr_player_reward -= 5

        match action["type"]:
            case ActionTypes.PlayDevelopmentCard:
                curr_player_reward += 5
            case ActionTypes.MoveRobber:
                curr_player_reward += 1
            case ActionTypes.StealResource:
                curr_player_reward += 0.5
            case ActionTypes.DiscardResource:
                curr_player_reward -= 0.25
            case ActionTypes.UpgradeToCity:
                curr_player_reward += 3

        rewards[curr_player.id] = curr_player_reward

        self.rewards = rewards

    def _get_tile_features(self) -> npt.NDArray:
        tile_features: list[list[float]] = []
        for tile in self.game.board.tiles:
            tile_features.append(
                [
                    float(tile.contains_robber),
                    float(tile.value),
                    float(tile.resource),
                ]
            )

        features_arr = np.array(tile_features, dtype=np.float32)

        if features_arr.shape != (N_TILES, 3):
            raise ValueError(
                f"Tile features have the wrong shape: {features_arr.shape}."
                f"Should be ({N_TILES}, 3)."
            )

        return features_arr

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

    def _get_corner_features(self, agent: PlayerId) -> npt.NDArray:
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

        features_arr = np.array(corner_features, dtype=np.float32)
        if features_arr.shape != (N_CORNERS, 2):
            raise ValueError(
                f"Corner features have the wrong shape: {features_arr.shape}."
                f"Should be ({N_CORNERS}, 2)."
            )
        return features_arr

    def _get_edge_features(self, agent: PlayerId) -> npt.NDArray:
        edge_features: list[list[float]] = []

        for edge in self.game.board.edges:
            road_owner: PlayerId | None = edge.road

            if road_owner is not None:
                # assign 1 to the player who owns the road, 2 to the next agent, etc.
                owner_id = self._get_agent_rel_pos(agent, road_owner) + 1
            else:
                owner_id = 0

            edge_features.append([float(owner_id)])

        features_arr = np.array(edge_features, dtype=np.float32)
        if features_arr.shape != (N_EDGES, 1):
            raise ValueError(
                f"Edge features have the wrong shape: {features_arr.shape}."
                f"Should be ({N_EDGES}, 1)."
            )
        return features_arr

    def _get_bank_features(self) -> npt.NDArray:
        features_arr: npt.NDArray = np.zeros(
            (len(Resource) - 1 + int(self.enable_dev_cards), 1),
            dtype=np.float32,
        )
        for i, resource in enumerate(Resource):
            # we ignore the empty resource, with idx = 0 for the first real resource
            if resource == Resource.Empty:
                continue

            count = self.game.resource_bank[resource]

            match count:
                case x if 0 <= x <= 2:
                    bucketed = count
                case x if 3 <= x <= 5:
                    bucketed = 3
                case x if 6 <= x <= 8:
                    bucketed = 4
                case _:
                    bucketed = 5

            features_arr[i] = bucketed

        if self.enable_dev_cards:
            dev_card_count = len(self.game.development_cards)
            match dev_card_count:
                case x if 0 <= x <= 2:
                    bucketed = dev_card_count
                case x if 3 <= x <= 5:
                    bucketed = 3
                case x if 6 <= x <= 8:
                    bucketed = 4
                case _:
                    bucketed = 5

            features_arr[-1] = bucketed

        if features_arr.shape != (
            len(Resource) - 1 + int(self.enable_dev_cards),
            1,
        ):
            raise ValueError(
                f"Bank features have the wrong shape: {features_arr.shape}."
                f"Should be ({len(Resource) - 1 + int(self.enable_dev_cards)}, 1)."
            )
        return features_arr

    def _get_public_player_features(
        self, agent: PlayerId
    ) -> list[npt.NDArray]:
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

        # swap empty with 3:1 harbor
        harbors = np.zeros((len(Resource),), dtype=np.float32)
        for harbor in player.harbours.values():
            # the 3:1 harbor will be index 0 instead of the empty resource
            if harbor.exchange_value == 3:
                harbors[0] = 1
            else:
                harbors[int(harbor.resource)] = 1

        # resources
        resources = np.zeros((len(Resource.non_empty())), dtype=np.float32)
        for res_idx, resource in enumerate(Resource.non_empty()):
            count = player.resources[resource]
            match count:
                case x if 0 <= x <= 7:
                    bucketed = count
                case _:
                    bucketed = 8
            resources[res_idx] = bucketed

        resource_production = np.zeros(
            (len(Resource.non_empty()), 11), dtype=np.float32
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
                    match corner.building:
                        case BuildingType.Settlement:
                            production += 1
                        case BuildingType.City:
                            production += 2

            resource_production[resource.int_non_empty(), tile.value - 2] = (
                production
            )

        played_dev_cards = np.zeros((len(DevelopmentCard),), dtype=np.float32)

        played_dev_card_counts = Counter(player.visible_cards)
        for i, card in enumerate(DevelopmentCard):
            played_dev_cards[i] = played_dev_card_counts[card]

        features = [
            float(needs_to_discard),
            float(has_longest_road),
            float(len_player_longest_road),
            float(has_largest_army),
            float(army_size),
            float(vp),
            harbors,
            resources,
            resource_production,
            played_dev_cards,
        ]

        return features

    def _get_unplayed_dev_cards_features(self, agent: PlayerId) -> npt.NDArray:
        player = self.game.players[agent]
        unplayed_dev_cards = np.zeros(
            (len(DevelopmentCard),), dtype=np.float32
        )
        unplayed_dev_card_counts = Counter(player.hidden_cards)
        for i, card in enumerate(DevelopmentCard):
            unplayed_dev_cards[i] = unplayed_dev_card_counts[card]
        return unplayed_dev_cards

    def _get_hidden_dev_card_count_features(self, agent: PlayerId) -> float:
        player = self.game.players[agent]

        return float(len(player.hidden_cards))
