import sys
import os

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.env import AMCDashEnv

def test_amc_dash_env():
    print("Testing AMCDashEnv with RGB mode...")
    env = AMCDashEnv(level=1, headless=True, obs_mode="rgb")
    obs, info = env.reset()
    print(f"RGB Observation shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape == (1, 84, 84), "Expected (1, 84, 84)"

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    print("RGB mode test passed.")

    print("Testing AMCDashEnv with State vector mode...")
    env_state = AMCDashEnv(level=1, headless=True, obs_mode="state")
    obs_s, info_s = env_state.reset()
    print(f"State Observation shape: {obs_s.shape}, values: {obs_s}")
    assert obs_s.shape == (8,), "Expected shape (8,)"

    print("State vector mode test passed.")

if __name__ == "__main__":
    test_amc_dash_env()
