import os
from pygame.math import Vector2

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MUSIC_DIR = os.path.join(BASE_DIR, "music")

# Display
WINDOW_WIDTH = 950
WINDOW_HEIGHT = 600
FPS = 60
TILE_SIZE = 32

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

PROGRESS_COLORS = [
    (255, 0, 0),      # Red
    (255, 165, 0),    # Orange
    (255, 255, 0),    # Yellow
    (144, 238, 144),  # Light green
    (0, 255, 0)       # Green
]

# Physics & Gameplay
GRAVITY = Vector2(0, 0.86)
MAX_FALL_SPEED = 100
PLAYER_SPEED = 6
JUMP_AMOUNT = 14
ORB_JUMP_AMOUNT = 20
ROTATION_SPEED = 8.1712

# Level files
LEVEL_FILES = [
    os.path.join(BASE_DIR, "level_1.csv"),
    os.path.join(BASE_DIR, "level_2.csv")
]

# Audio files
MUSIC_LEVEL_1 = os.path.join(MUSIC_DIR, "bossfight-Vextron.mp3")
MUSIC_LEVEL_2 = os.path.join(MUSIC_DIR, "Castle-town.mp3")
