import pygame


class Obstacle(pygame.sprite.Sprite):
    """Base class for all obstacle sprites in the game."""

    def __init__(self, image: pygame.Surface, pos: tuple, *groups):
        super().__init__(*groups)
        self.image = image
        self.rect = self.image.get_rect(topleft=pos)


class Platform(Obstacle):
    """Solid block obstacle. Player can stand on top; hits head from below; dies if hit from side."""
    pass


class Spike(Obstacle):
    """Spike hazard. Player dies immediately upon collision."""
    pass


class Coin(Obstacle):
    """Collectible coin."""
    pass


class Orb(Obstacle):
    """Jump orb. Pressing jump while touching gives extra vertical boost in mid-air."""
    pass


class Trick(Obstacle):
    """Fake block that the player can safely pass through."""
    pass


class End(Obstacle):
    """Level goal marker."""
    pass
