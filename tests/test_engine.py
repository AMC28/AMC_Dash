import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.engine import GameEngine

def run_tests():
    print("Testing GameEngine initialization (headless)...")
    engine = GameEngine(headless=True, sound_enabled=False)
    print("GameEngine initialized.")

    print(f"Level 1 elements: {len(engine.elements)}")
    assert len(engine.elements) > 0, "Level 1 should have elements"

    engine.state = GameEngine.STATE_PLAYING
    ended = False
    for i in range(150):
        # periodic jump every 40 frames
        res = engine.step(jump_input=(i % 40 == 0))
        if res["done"]:
            print(f"Simulation ended at frame {i}: died={res['died']}, won={res['won']}")
            ended = True
            break

    if not ended:
        print("Simulation ran for 150 frames smoothly!")

    print("Testing Level 2 loading...")
    engine.load_level(1)
    print(f"Level 2 elements: {len(engine.elements)}")
    assert len(engine.elements) > 0, "Level 2 should have elements"

    print("All tests passed successfully!")

if __name__ == "__main__":
    run_tests()
