import csv
import os
import pygame
from typing import List, Tuple

from game.config import TILE_SIZE
from game.obstacles import Platform, Spike, Coin, Orb, Trick, End


class LevelLoader:
    """
    Handles CSV level parsing and spawning of obstacle sprites.
    """

    @staticmethod
    def load_csv(csv_path: str) -> List[List[str]]:
        """Reads CSV file and returns cleaned 2D matrix of tile tokens."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Level file not found: {csv_path}")

        matrix = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            for row in reader:
                matrix.append([cell.strip() for cell in row])
        return matrix

    @staticmethod
    def build_level(
        map_matrix: List[List[str]],
        assets: dict,
        elements_group: pygame.sprite.Group
    ) -> Tuple[int, int]:
        """
        Populates elements_group with obstacles according to map matrix.
        Returns total (width, height) of the level in pixels.
        """
        elements_group.empty()

        rows = len(map_matrix)
        cols = max(len(row) for row in map_matrix) if rows > 0 else 0
        total_width = cols * TILE_SIZE
        total_height = rows * TILE_SIZE

        for y_idx, row in enumerate(map_matrix):
            for x_idx, cell in enumerate(row):
                pos = (x_idx * TILE_SIZE, y_idx * TILE_SIZE)

                if cell == "0":
                    Platform(assets["block"], pos, elements_group)
                elif cell == "Spike":
                    Spike(assets["spike"], pos, elements_group)
                elif cell == "Coin":
                    Coin(assets["coin"], pos, elements_group)
                elif "Orb" in cell or cell == "-1rb":
                    Orb(assets["orb"], pos, elements_group)
                elif cell == "T":
                    Trick(assets["trick"], pos, elements_group)
                elif cell == "End":
                    End(assets["coin"], pos, elements_group)

        return total_width, total_height
