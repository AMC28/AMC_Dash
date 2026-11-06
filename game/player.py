import random
import pygame
from pygame.math import Vector2
from pygame.draw import rect

from game.config import (
    GRAVITY,
    MAX_FALL_SPEED,
    JUMP_AMOUNT,
    ORB_JUMP_AMOUNT,
    ROTATION_SPEED,
    WHITE,
    YELLOW,
    TILE_SIZE,
    WINDOW_HEIGHT
)
from game.obstacles import Platform, Spike, Coin, Orb, Trick, End


class Player(pygame.sprite.Sprite):
    """
    Main player cube character.
    Handles physics (gravity, jump velocity), collisions, rotation animation, and trail particles.
    """

    def __init__(self, image: pygame.Surface, pos: tuple = (150, 150), *groups):
        super().__init__(*groups)
        self.original_image = pygame.transform.smoothscale(image, (TILE_SIZE, TILE_SIZE))
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=pos)

        self.spawn_pos = pos
        self.vel = Vector2(0, 0)
        self.onGround = False
        self.isjump = False
        self.died = False
        self.win = False

        self.jump_amount = JUMP_AMOUNT
        self.angle = 0.0
        self.particles = []
        self.coins_collected = 0

    def reset(self, pos: tuple = None):
        """Reset player state for a new attempt or level."""
        target_pos = pos if pos is not None else self.spawn_pos
        self.rect.center = target_pos
        self.vel = Vector2(0, 0)
        self.onGround = False
        self.isjump = False
        self.died = False
        self.win = False
        self.angle = 0.0
        self.particles.clear()
        self.jump_amount = JUMP_AMOUNT
        self.coins_collected = 0

    def jump(self):
        """Execute a jump upwards."""
        self.vel.y = -self.jump_amount
        self.onGround = False
        self.isjump = True

    def draw_particle_trail(self, alpha_surf: pygame.Surface, color=WHITE):
        """Generates and updates fading square trail particles behind the player."""
        px = self.rect.left - 2
        py = self.rect.bottom - 4

        self.particles.append([
            [px - 5, py - 8],
            [random.uniform(-1.0, 1.5), 0],
            random.uniform(5, 8)
        ])

        for particle in list(self.particles):
            particle[0][0] += particle[1][0]
            particle[0][1] += particle[1][1]
            particle[2] -= 0.5
            particle[1][0] -= 0.4
            p_size = int(particle[2])
            if p_size > 0:
                rect(alpha_surf, color, ([int(particle[0][0]), int(particle[0][1])], [p_size, p_size]))
            else:
                self.particles.remove(particle)

    def update(self, platforms, jump_input: bool = False, fx_surface: pygame.Surface = None):
        """
        Update player position, physics, and collisions for one frame.
        """
        if self.died or self.win:
            return

        # Filter nearby obstacles once per frame for fast and deterministic resolution
        nearby = [
            p for p in platforms
            if not (p.rect.right < self.rect.left - 12 or p.rect.left > self.rect.right + 12)
        ]

        # 1. Jump initiated if on ground
        if jump_input and self.onGround:
            self.jump()

        # 2. Check Yellow Orbs before vertical movement (mid-air boost)
        if jump_input:
            for p in nearby:
                if isinstance(p, Orb) and self.rect.colliderect(p.rect.inflate(8, 8)):
                    if fx_surface is not None:
                        pygame.draw.circle(fx_surface, YELLOW, p.rect.center, 18)
                    self.jump_amount = ORB_JUMP_AMOUNT
                    self.jump()
                    self.jump_amount = JUMP_AMOUNT
                    break

        # 3. Apply gravity when in the air
        if not self.onGround:
            self.vel += GRAVITY
            if self.vel.y > MAX_FALL_SPEED:
                self.vel.y = MAX_FALL_SPEED

        # Store previous vertical bounds before vertical movement
        prev_bottom = self.rect.bottom
        prev_top = self.rect.top

        # 4. Apply vertical velocity
        self.rect.y += int(self.vel.y)
        self.onGround = False

        # 5. Check Orbs, Coins, and End triggers first
        for p in nearby:
            if isinstance(p, Orb) and jump_input and self.rect.colliderect(p.rect.inflate(8, 8)):
                if fx_surface is not None:
                    pygame.draw.circle(fx_surface, YELLOW, p.rect.center, 18)
                self.jump_amount = ORB_JUMP_AMOUNT
                self.jump()
                self.jump_amount = JUMP_AMOUNT
                continue

            if not self.rect.colliderect(p.rect):
                continue

            if isinstance(p, End):
                self.win = True
            elif isinstance(p, Coin):
                self.coins_collected += 1
                p.kill()

        # 6. Resolve solid Platforms and Spikes
        for p in nearby:
            if not self.rect.colliderect(p.rect):
                continue

            if isinstance(p, Spike):
                spike_hitbox = p.rect.inflate(-10, -8)
                if self.rect.colliderect(spike_hitbox):
                    self.died = True

            elif isinstance(p, Platform):
                if self.vel.y >= 0 and prev_bottom <= p.rect.top + 8:
                    # Landing on top of a platform
                    self.rect.bottom = p.rect.top
                    self.vel.y = 0
                    self.onGround = True
                    self.isjump = False
                elif self.vel.y < 0 and prev_top >= p.rect.bottom - 8:
                    # Hitting platform ceiling from underneath
                    self.rect.top = p.rect.bottom
                    self.vel.y = 0
                else:
                    # Horizontal crash into the side of a block
                    self.vel.x = 0
                    self.rect.right = p.rect.left
                    self.died = True

        # 7. Maintain stable ground state (prevent 1px hover oscillation on flat ground)
        if not self.onGround and self.vel.y >= 0 and not self.died:
            foot_rect = self.rect.move(0, 1)
            for p in nearby:
                if isinstance(p, Platform) and foot_rect.colliderect(p.rect):
                    if self.rect.bottom <= p.rect.top + 2:
                        self.rect.bottom = p.rect.top
                        self.vel.y = 0
                        self.onGround = True
                        self.isjump = False
                        break

        # 8. Check out-of-bounds fall
        if self.rect.top > WINDOW_HEIGHT:
            self.died = True

        # 9. Rotation animation
        if not self.onGround or self.isjump:
            self.angle = (self.angle - ROTATION_SPEED) % 360
        else:
            self.angle = round(self.angle / 90.0) * 90.0 % 360

    def draw(self, surface: pygame.Surface):
        """Draw player with smooth rotation around its center."""
        if self.angle != 0:
            rotated = pygame.transform.rotozoom(self.original_image, self.angle, 1.0)
            rot_rect = rotated.get_rect(center=self.rect.center)
            surface.blit(rotated, rot_rect)
        else:
            surface.blit(self.original_image, self.rect)
