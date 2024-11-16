import copy

import gymnasium.spaces as spaces
import numpy as np
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AECEnv

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

        self.agents = self.possible_agents.copy()
        self.agent_selection: PlayerId = self.game.players_go

        # termination is the natural game end for the player
        self.terminations = {agent: False for agent in self.agents}
        # truncation is us forcing the game to end for that player
        self.truncations = {agent: False for agent in self.agents}

        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.observation_spaces = {
            agent: self._get_obs_space(agent) for agent in self.agents
        }

    def _is_game_over(self):
        # set terminals to True for all if a player hit 10 VPs
        pass

    def _get_obs_space(self, agent: PlayerId):
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
            spaces.MultiDiscrete([9] * (len(Resource) - 1)),  # resources
            # resource production
            spaces.Box(
                low=0,  # min number of resources per roll
                high=11,  # max number of resources per roll
                # resources x tile roll values (offset from [2, 12] to indices [0, 10])
                shape=(len(Resource) - 1, 11),
                dtype=np.int8,
            ),
            # played dev cards
            spaces.Box(
                low=np.zeros((len(self.game.max_dev_cards_by_type),)),
                high=np.array(self.game.max_dev_cards_by_type.values()),
                shape=(len(DevelopmentCard),),
                dtype=np.int8,
            ),
        ]

        unplayed_dev_cards = spaces.Box(
            low=np.zeros((len(self.game.max_dev_cards_by_type),)),
            high=np.array(self.game.max_dev_cards_by_type.values()),
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
            "this_agent": this_agent_space,
            "other_agents": other_agent_spaces,
        }

        return spaces.Dict(obs)
