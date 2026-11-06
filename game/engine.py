import os
import pygame
from typing import Dict, Any, Tuple

from game.config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    FPS,
    TILE_SIZE,
    WHITE,
    BLACK,
    BLUE,
    GREEN,
    YELLOW,
    PROGRESS_COLORS,
    PLAYER_SPEED,
    LEVEL_FILES,
    IMAGES_DIR,
    MUSIC_DIR
)
from game.player import Player
from game.level import LevelLoader


class GameEngine:
    """
    Main Game Engine managing state, graphics, level progression,
    event loops, HUD, and programmatic stepping (for AI).
    """

    STATE_MENU = "menu"
    STATE_PLAYING = "playing"
    STATE_GAMEOVER = "gameover"
    STATE_VICTORY = "victory"

    def __init__(self, headless: bool = False, sound_enabled: bool = True):
        self.headless = headless
        self.sound_enabled = sound_enabled

        if not pygame.get_init():
            pygame.init()

        if not self.headless:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("AMC Dash: Geometry Dash Python")
        else:
            self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

        self.clock = pygame.time.Clock()
        self.alpha_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

        # Asset initialization
        self.assets: Dict[str, pygame.Surface] = {}
        self.load_assets()

        # Fonts
        if not self.headless:
            self.font_main = pygame.font.SysFont("lucidaconsole", 20)
            self.font_title = pygame.font.SysFont("lucidaconsole", 28, bold=True)
        else:
            self.font_main = None
            self.font_title = None

        # Game state variables
        self.state = self.STATE_MENU
        self.current_level_idx = 0
        self.attempts = 1
        self.coins = 0
        self.progress_px = 0.0
        self.level_width = 0
        self.level_height = 0
        self.distance_traveled = 0

        # Sprite groups
        self.player_sprite = pygame.sprite.GroupSingle()
        self.elements = pygame.sprite.Group()

        # Instantiate Player
        self.player = Player(self.assets["avatar"], (150, 150))
        self.player_sprite.add(self.player)

        # Cache parsed CSV matrices in memory for fast RL resets
        self.level_matrices = [LevelLoader.load_csv(path) for path in LEVEL_FILES]

        # Pre-load initial level
        self.load_level(self.current_level_idx)

    def load_assets(self):
        """Load and scale all image assets."""
        def load_img(name: str, size: Tuple[int, int] = (TILE_SIZE, TILE_SIZE)) -> pygame.Surface:
            path = os.path.join(IMAGES_DIR, name)
            if not os.path.exists(path):
                # Fallback surface
                surf = pygame.Surface(size)
                surf.fill(BLUE)
                return surf
            img = pygame.image.load(path)
            return pygame.transform.smoothscale(img, size)

        self.assets["avatar"] = load_img("avatar.png")
        self.assets["block"] = load_img("block_1.png")
        self.assets["spike"] = load_img("obj-spike.png")
        self.assets["coin"] = load_img("coin.png")
        self.assets["orb"] = load_img("orb-yellow.png")
        self.assets["trick"] = load_img("obj-breakable.png")

        bg_path = os.path.join(IMAGES_DIR, "bg.png")
        if os.path.exists(bg_path):
            self.assets["bg"] = pygame.image.load(bg_path)
            self.assets["bg"] = pygame.transform.scale(self.assets["bg"], (WINDOW_WIDTH, WINDOW_HEIGHT))
        else:
            self.assets["bg"] = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.assets["bg"].fill((20, 20, 40))

        if not self.headless:
            pygame.display.set_icon(self.assets["avatar"])

    def play_music(self, level_idx: int):
        """Play level soundtrack if audio is available."""
        if not self.sound_enabled or self.headless:
            return

        try:
            pygame.mixer.music.stop()
            track = "bossfight-Vextron.mp3" if level_idx == 0 else "Castle-town.mp3"
            music_path = os.path.join(MUSIC_DIR, track)
            if os.path.exists(music_path):
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play(-1)
        except Exception:
            pass

    def load_level(self, level_idx: int):
        """Loads a given level by index."""
        self.current_level_idx = level_idx % len(LEVEL_FILES)
        map_matrix = self.level_matrices[self.current_level_idx]
        self.level_width, self.level_height = LevelLoader.build_level(
            map_matrix, self.assets, self.elements
        )
        self.reset_player()
        self.progress_px = 0.0
        self.distance_traveled = 0

    def reset_player(self):
        """Reset player and obstacle scroll positions for current level."""
        self.player.reset((150, 150))
        self.progress_px = 0.0
        self.distance_traveled = 0

    def reset_game(self, new_attempt: bool = True):
        """Reset after game over or level win."""
        if new_attempt:
            self.attempts += 1
        self.load_level(self.current_level_idx)
        self.state = self.STATE_PLAYING
        self.play_music(self.current_level_idx)

    def step(self, jump_input: bool = False) -> Dict[str, Any]:
        """
        Executes one physics frame. Suitable for both human input and AI step loops.
        Returns: observation status, reward, done flag, info.
        """
        if self.state != self.STATE_PLAYING:
            return {
                "won": self.player.win,
                "died": self.player.died,
                "done": True,
                "reward": 0.0,
                "coins": self.player.coins_collected
            }

        # Move obstacles leftwards
        for sprite in self.elements:
            sprite.rect.x -= PLAYER_SPEED

        self.distance_traveled += PLAYER_SPEED
        self.progress_px = min(float(WINDOW_WIDTH), (self.distance_traveled / max(1, self.level_width - WINDOW_WIDTH)) * WINDOW_WIDTH)

        # Update player physics and collisions
        self.player.update(self.elements, jump_input=jump_input, fx_surface=self.alpha_surf)

        # Determine reward and terminal conditions
        reward = 1.0  # Reward for surviving frame
        done = False

        if self.player.died:
            self.state = self.STATE_GAMEOVER
            reward = -50.0
            done = True
        elif self.player.win:
            self.state = self.STATE_VICTORY
            reward = 100.0
            done = True

        return {
            "won": self.player.win,
            "died": self.player.died,
            "done": done,
            "reward": reward,
            "coins": self.player.coins_collected
        }

    def render_hud(self):
        """Render attempt count, progress bar, and coins."""
        # Trail fade surface
        self.alpha_surf.fill((255, 255, 255, 1), special_flags=pygame.BLEND_RGBA_MULT)
        self.player.draw_particle_trail(self.alpha_surf, WHITE)
        self.screen.blit(self.alpha_surf, (0, 0))

        # Progress bar
        bar_length = WINDOW_WIDTH
        bar_height = 10
        color_idx = min(len(PROGRESS_COLORS) - 1, int((self.progress_px / bar_length) * len(PROGRESS_COLORS)))
        col = PROGRESS_COLORS[color_idx]

        pygame.draw.rect(self.screen, col, (0, 0, int(self.progress_px), bar_height))
        pygame.draw.rect(self.screen, WHITE, (0, 0, bar_length, bar_height), 2)

        # Attempts text
        if self.font_main:
            tries = self.font_main.render(f"Attempt {self.attempts}", True, WHITE)
            self.screen.blit(tries, (WINDOW_WIDTH - 150, 15))

            # Coins display
            coins_txt = self.font_main.render(f"Coins: {self.player.coins_collected}", True, YELLOW)
            self.screen.blit(coins_txt, (20, 15))

    def render_ai_overlay(
        self,
        radar_targets: Dict[str, Any] = None,
        q_vals: list = None,
        action: int = 1,
        ai_enabled: bool = True,
        radar_enabled: bool = True
    ):
        """
        Render real-time AI Radar Raycast lines and Q-Value Telemetry HUD on screen.
        """
        if self.headless or self.state != self.STATE_PLAYING:
            return

        p_center = self.player.rect.center

        # 1. Draw Raycast Sensor Lines & Target Bounding Boxes
        if radar_enabled and radar_targets:
            # Sensor ring around player
            pygame.draw.circle(self.screen, (0, 220, 255), p_center, 22, 1)

            h_rect = radar_targets.get("hazard_rect")
            if h_rect is not None:
                pygame.draw.line(self.screen, (255, 65, 85), p_center, h_rect.center, 2)
                pygame.draw.rect(self.screen, (255, 65, 85), h_rect.inflate(4, 4), 2)

            p_rect = radar_targets.get("plat_rect")
            if p_rect is not None:
                target_pt = (p_rect.left, p_rect.top)
                pygame.draw.line(self.screen, (75, 255, 130), p_center, target_pt, 2)
                pygame.draw.rect(self.screen, (75, 255, 130), (p_rect.left, p_rect.top, p_rect.width, 6), 0)

            o_rect = radar_targets.get("orb_rect")
            if o_rect is not None:
                pygame.draw.line(self.screen, YELLOW, p_center, o_rect.center, 2)
                pygame.draw.circle(self.screen, YELLOW, o_rect.center, 20, 2)

        # 2. Draw Telemetry HUD Panel
        if self.font_main and (ai_enabled or radar_enabled):
            panel_w, panel_h = 480, 74
            panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            panel.fill((12, 16, 28, 195))
            pygame.draw.rect(panel, (60, 180, 255, 210), (0, 0, panel_w, panel_h), 1, border_radius=6)

            mode_str = "AI AUTOPLAY [A]: ON" if ai_enabled else "AI AUTOPLAY [A]: OFF (MANUAL)"
            mode_col = (80, 255, 140) if ai_enabled else (255, 200, 80)
            t1 = self.font_main.render(mode_str, True, mode_col)
            panel.blit(t1, (10, 6))

            if q_vals is not None and len(q_vals) >= 2:
                act_str = "JUMP!" if action == 0 else "STAY"
                act_col = YELLOW if action == 0 else (180, 220, 255)
                q_str = f"Q(Jump):{q_vals[0]:+6.1f} | Q(Stay):{q_vals[1]:+6.1f} -> {act_str}"
                t2 = self.font_main.render(q_str, True, act_col)
                panel.blit(t2, (10, 28))

            rad_str = f"RADAR [R]: {'ON' if radar_enabled else 'OFF'}  |  Prog: {(self.progress_px / WINDOW_WIDTH) * 100:.1f}%"
            t3 = self.font_main.render(rad_str, True, (200, 210, 230))
            panel.blit(t3, (10, 50))

            self.screen.blit(panel, (16, 42))

    def render(self, ai_overlay: Dict[str, Any] = None):
        """Render current state on screen."""
        if self.headless:
            return

        if self.state == self.STATE_MENU:
            self.screen.fill(BLACK)
            if self.font_title and self.font_main:
                title = self.font_title.render("AMC DASH - Geometry Dash & RL AI", True, YELLOW)
                lvl_info = self.font_main.render(f"Pilih Level: [1] Level 1   |   [2] Level 2  (Aktif: Level {self.current_level_idx + 1})", True, WHITE)
                ctrl = self.font_main.render("Kontrol: [SPACE]/[UP] Lompat  |  [A] AI Autoplay  |  [R] AI Radar", True, GREEN)
                press_space = self.font_main.render("Tekan [SPACE] Untuk Memulai!", True, BLUE)

                self.screen.blit(title, (50, 150))
                self.screen.blit(lvl_info, (50, 230))
                self.screen.blit(ctrl, (50, 300))
                self.screen.blit(press_space, (50, 420))

        elif self.state == self.STATE_PLAYING:
            # Draw background
            self.screen.blit(self.assets["bg"], (0, 0))
            # Draw obstacles
            self.elements.draw(self.screen)
            # Draw player
            self.player.draw(self.screen)
            # Draw HUD
            self.render_hud()
            # Draw AI Radar & Telemetry if provided
            if ai_overlay:
                self.render_ai_overlay(
                    radar_targets=ai_overlay.get("radar_targets"),
                    q_vals=ai_overlay.get("q_vals"),
                    action=ai_overlay.get("action", 1),
                    ai_enabled=ai_overlay.get("ai_enabled", True),
                    radar_enabled=ai_overlay.get("radar_enabled", True)
                )

        elif self.state == self.STATE_GAMEOVER:
            self.screen.fill(BLACK)
            if self.font_title and self.font_main:
                over_txt = self.font_title.render("GAME OVER", True, (255, 60, 60))
                retry_txt = self.font_main.render("Tekan [SPACE] Untuk Coba Lagi", True, WHITE)
                esc_txt = self.font_main.render("Tekan [ESC] Untuk Keluar", True, (160, 160, 160))

                self.screen.blit(over_txt, (WINDOW_WIDTH // 2 - 90, 220))
                self.screen.blit(retry_txt, (WINDOW_WIDTH // 2 - 180, 300))
                self.screen.blit(esc_txt, (WINDOW_WIDTH // 2 - 130, 350))

        elif self.state == self.STATE_VICTORY:
            self.screen.fill((20, 40, 20))
            if self.font_title and self.font_main:
                win_txt = self.font_title.render(f"SELAMAT! Level {self.current_level_idx + 1} Selesai!", True, YELLOW)
                coins_txt = self.font_main.render(f"Total Koin Terkumpul: {self.player.coins_collected}", True, WHITE)
                next_txt = self.font_main.render("Tekan [SPACE] Lanjut / Ulangi   |   [ESC] Keluar", True, GREEN)

                self.screen.blit(win_txt, (WINDOW_WIDTH // 2 - 250, 220))
                self.screen.blit(coins_txt, (WINDOW_WIDTH // 2 - 140, 280))
                self.screen.blit(next_txt, (WINDOW_WIDTH // 2 - 240, 350))

        pygame.display.flip()
        self.clock.tick(FPS)
