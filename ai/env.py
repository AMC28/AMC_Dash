import numpy as np
import pygame

try:
    import gymnasium as gym
    from gymnasium import spaces
    BaseEnv = gym.Env
except ImportError:
    try:
        import gym
        from gym import spaces
        BaseEnv = gym.Env
    except ImportError:
        gym = None
        spaces = None
        BaseEnv = object

from game.engine import GameEngine
from game.config import WINDOW_WIDTH, WINDOW_HEIGHT
from game.obstacles import Platform, Spike, Orb


class AMCDashEnv(BaseEnv):
    """
    Gymnasium / Gym Environment for AMC Dash.
    Supports:
    - obs_mode='state': 8-dimensional radar sensor vector (fast RL training)
    - obs_mode='rgb': (1, 84, 84) grayscale image frame (CNN DQN training)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, level: int = 1, headless: bool = True, obs_mode: str = "state"):
        super().__init__()
        self.level_idx = level - 1
        self.headless = headless
        self.obs_mode = obs_mode

        self.engine = GameEngine(headless=self.headless, sound_enabled=False)
        self.engine.load_level(self.level_idx)
        self._prev_coins = 0
        self._last_radar_targets = {}

        # Action space: 0 = JUMP, 1 = NO-OP
        if spaces:
            self.action_space = spaces.Discrete(2)
            if self.obs_mode == "rgb":
                self.observation_space = spaces.Box(
                    low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
                )
            else:
                self.observation_space = spaces.Box(
                    low=-2.0, high=2.0, shape=(8,), dtype=np.float32
                )

    def _get_visual_observation(self) -> np.ndarray:
        """Capture screen surface, convert to grayscale, and resize to (1, 84, 84)."""
        self.engine.render()
        surf = self.engine.screen
        scaled = pygame.transform.smoothscale(surf, (84, 84))
        rgb_array = pygame.surfarray.array3d(scaled)
        gray = (
            0.299 * rgb_array[:, :, 0] +
            0.587 * rgb_array[:, :, 1] +
            0.114 * rgb_array[:, :, 2]
        ).astype(np.uint8)
        return gray.T.reshape(1, 84, 84)

    def _get_state_observation(self) -> np.ndarray:
        """
        Extract 8-dimensional radar sensor vector and store target coordinates for HUD visualization:
        [0] player_y_norm       : Player vertical position (0..1)
        [1] player_vy_norm      : Player vertical velocity (-1..1)
        [2] on_ground           : 1.0 if on ground else 0.0
        [3] hazard_dx_norm      : Distance to nearest Spike or wall ahead (0..1)
        [4] hazard_dy_norm      : Vertical offset to nearest hazard (-1..1)
        [5] platform_up_dx_norm : Distance to next step-up platform ahead (0..1)
        [6] platform_up_dy_norm : Height of step-up platform relative to player (-1..1)
        [7] orb_dx_norm         : Distance to nearest Jump Orb ahead (0..1)
        """
        p = self.engine.player
        p_right = p.rect.right
        p_bottom = p.rect.bottom
        p_cy = p.rect.centery

        hazard_dx = 300.0
        hazard_dy = 0.0
        plat_dx = 300.0
        plat_dy = 0.0
        orb_dx = 200.0

        hazard_rect = None
        plat_rect = None
        orb_rect = None
        touching_orb = False

        for sprite in self.engine.elements:
            dx = sprite.rect.left - p_right
            if dx < -40 or dx > 300:
                continue

            if isinstance(sprite, Spike):
                if abs(sprite.rect.centery - p_cy) < 120 and dx >= -16 and dx < hazard_dx:
                    hazard_dx = max(0.0, float(dx))
                    hazard_dy = float(sprite.rect.centery - p_cy)
                    hazard_rect = sprite.rect

            elif isinstance(sprite, Platform):
                # Wall blocking horizontal path
                if sprite.rect.top < p_bottom - 4 and sprite.rect.bottom > p.rect.top + 4:
                    if dx >= 0 and dx < hazard_dx:
                        hazard_dx = float(dx)
                        hazard_dy = float(sprite.rect.centery - p_cy)
                        hazard_rect = sprite.rect
                # Elevated step-up platform ahead
                if 16 <= (p_bottom - sprite.rect.top) <= 120:
                    if dx >= 0 and dx < plat_dx:
                        plat_dx = float(dx)
                        plat_dy = float(sprite.rect.top - p_bottom)
                        plat_rect = sprite.rect

            elif isinstance(sprite, Orb):
                if p.rect.colliderect(sprite.rect.inflate(8, 8)):
                    touching_orb = True
                if dx >= -24 and dx < orb_dx:
                    orb_dx = max(0.0, float(dx))
                    orb_rect = sprite.rect

        self._last_radar_targets = {
            "hazard_rect": hazard_rect,
            "plat_rect": plat_rect,
            "orb_rect": orb_rect,
            "touching_orb": touching_orb
        }

        return np.array([
            p_cy / WINDOW_HEIGHT,
            np.clip(p.vel.y / 20.0, -1.0, 1.0),
            1.0 if p.onGround else 0.0,
            np.clip(hazard_dx / 300.0, 0.0, 1.0),
            np.clip(hazard_dy / 150.0, -1.0, 1.0),
            np.clip(plat_dx / 300.0, 0.0, 1.0),
            np.clip(plat_dy / 150.0, -1.0, 1.0),
            np.clip(orb_dx / 200.0, 0.0, 1.0)
        ], dtype=np.float32)

    def _build_info(self, won: bool = False, died: bool = False) -> dict:
        max_dist = max(1, self.engine.level_width - 150)
        progress = min(100.0, (self.engine.distance_traveled / max_dist) * 100.0)
        if won:
            progress = 100.0
        return {
            "won": won,
            "died": died,
            "coins": self.engine.player.coins_collected,
            "distance": self.engine.distance_traveled,
            "progress": progress,
            "player_y": self.engine.player.rect.y,
            "player_vy": int(round(self.engine.player.vel.y * 100)),
            "on_ground": self.engine.player.onGround,
            "touching_orb": self._last_radar_targets.get("touching_orb", False),
            "radar_targets": self._last_radar_targets
        }

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        self.engine.reset_game(new_attempt=True)
        self.engine.state = GameEngine.STATE_PLAYING
        self._prev_coins = 0

        obs = self._get_visual_observation() if self.obs_mode == "rgb" else self._get_state_observation()
        return obs, self._build_info()

    def step(self, action: int):
        """
        Execute one physics step.
        action: 0 = JUMP, 1 = NO-OP
        """
        jump_input = (action == 0)
        step_result = self.engine.step(jump_input=jump_input)

        obs = self._get_visual_observation() if self.obs_mode == "rgb" else self._get_state_observation()

        reward = step_result["reward"]
        new_coins = step_result["coins"]
        if new_coins > self._prev_coins:
            reward += 15.0 * (new_coins - self._prev_coins)
            self._prev_coins = new_coins

        terminated = step_result["done"]
        truncated = False
        info = self._build_info(won=step_result["won"], died=step_result["died"])

        return obs, reward, terminated, truncated, info

    def render(self):
        self.engine.render()

    def close(self):
        pygame.quit()
