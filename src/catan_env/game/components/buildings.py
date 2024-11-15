from catan_env.game.enums import BuildingType, PlayerId


class Building:
    def __init__(self, type: BuildingType, owner: PlayerId, corner: None):
        self.type = type
        self.owner = owner
        self.corner = corner
