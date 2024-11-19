import copy

from catan_env.game.components.buildings import Building
from catan_env.game.enums import DevelopmentCard, PlayerId, Resource


class Player(object):
    def __init__(self, id: PlayerId):
        self.id: PlayerId = id

    def reset(self, player_order):
        self.player_order = player_order
        self.player_lookup = {}
        self.inverse_player_lookup = {}
        for i in range(len(player_order)):
            if player_order[i] == self.id:
                p_ind = i
        for i, label in enumerate(["next", "next_next", "next_next_next"]):
            ind = (p_ind + 1 + i) % 4
            self.player_lookup[self.player_order[ind]] = label
            self.inverse_player_lookup[label] = self.player_order[ind]

        self.buildings: dict[int, Building] = {}
        self.roads = []
        self.resources: dict[Resource, int] = {
            Resource.Brick: 0,
            Resource.Wood: 0,
            Resource.Wheat: 0,
            Resource.Ore: 0,
            Resource.Sheep: 0,
        }
        self.visible_resources = {
            Resource.Brick: self.resources[Resource.Brick],
            Resource.Wood: self.resources[Resource.Wood],
            Resource.Wheat: self.resources[Resource.Wheat],
            Resource.Sheep: self.resources[Resource.Sheep],
            Resource.Ore: self.resources[Resource.Ore],
        }
        self.opponent_max_res = {
            "next": copy.deepcopy(self.visible_resources),
            "next_next": copy.deepcopy(self.visible_resources),
            "next_next_next": copy.deepcopy(self.visible_resources),
        }
        self.opponent_min_res = copy.deepcopy(self.opponent_max_res)
        self.harbours = {}
        self.longest_road: int = 0
        self.hidden_cards: list[DevelopmentCard] = []
        self.visible_cards: list[DevelopmentCard] = []
        self.victory_points: int = 0
