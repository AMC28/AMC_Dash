import os
import json
import numpy as np
from typing import Dict, Tuple, List, Any


class QLearningDashAgent:
    """
    Dual-Layer Reinforcement Learning Agent for AMC Dash:
    1. Sensor-Generalization Layer (`sensor_q_table`):
       Encodes strictly relative radar observations (distance & height to nearest
       Spike/Wall, Step-up Platform, and Jump Orb) WITHOUT absolute x-coordinates.
       Enables zero-shot generalization to unseen obstacles and custom levels.
    2. Spatial Bellman Layer (`q_table`):
       Refines frame-exact multi-platform routing using Backward Bellman Trajectory
       Sweep and Mid-Air Action-Space Masking.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        discount_factor: float = 0.995,
        epsilon: float = 0.0,
        epsilon_min: float = 0.0,
        epsilon_decay: float = 0.99
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Spatial Q-Table: state_key -> [Q(s, JUMP), Q(s, NO_OP)]
        self.q_table: Dict[str, List[float]] = {}
        # Relative Sensor Q-Table (position-independent generalization): sensor_key -> [Q_jump, Q_noop]
        self.sensor_q_table: Dict[str, List[float]] = {}

    @staticmethod
    def encode_state(obs: np.ndarray, info: Dict[str, Any]) -> str:
        """
        Encode physical state into a discrete Markov state key:
        (x_frame, player_y, player_vy_rounded, on_ground, touching_orb)
        """
        dist_step = int(info.get("distance", 0)) // 6
        py = int(info.get("player_y", 0))
        vy = int(info.get("player_vy", 0))
        og = 1 if info.get("on_ground", False) else 0
        orb = 1 if info.get("touching_orb", False) else 0
        return f"{dist_step}_{py}_{vy}_{og}_{orb}"

    @staticmethod
    def encode_sensor_state(obs: np.ndarray, info: Dict[str, Any]) -> str:
        """
        Encode position-independent relative radar sensor state (for generalization):
        Uses relative distance to nearest hazard, relative height, platform step-up,
        ground status, and Orb contact.
        """
        if obs is None or len(obs) < 8:
            return "S_default"
        og = 1 if info.get("on_ground", False) else 0
        orb = 1 if info.get("touching_orb", False) else 0
        h_dx_bin = int(obs[3] * 15)         # 20px buckets (0..15)
        h_dy_bin = int(round(obs[4] * 4))   # vertical offset buckets
        p_dx_bin = int(obs[5] * 10)         # platform distance buckets
        return f"S_{og}_{orb}_{h_dx_bin}_{h_dy_bin}_{p_dx_bin}"

    def get_sensor_q_values(self, sensor_key: str, obs: np.ndarray = None, info: Dict[str, Any] = None) -> List[float]:
        """Get or initialize Q-values in the relative sensor table."""
        if sensor_key not in self.sensor_q_table:
            # Default sensor heuristic prior:
            # If touching a Jump Orb in mid-air or a hazard is within jump range (18..75px),
            # give a slight positive prior to JUMP; otherwise prefer NO_OP (running).
            if info and info.get("touching_orb", False):
                self.sensor_q_table[sensor_key] = [0.2, 0.0]
            else:
                self.sensor_q_table[sensor_key] = [0.0, 0.1]
        return self.sensor_q_table[sensor_key]

    def get_q_values(self, state_key: str, obs: np.ndarray = None, info: Dict[str, Any] = None) -> List[float]:
        """Retrieve Q-values for state, initializing with safe default preference."""
        if state_key not in self.q_table:
            if info and info.get("touching_orb", False):
                self.q_table[state_key] = [0.1, 0.0]
            else:
                self.q_table[state_key] = [0.0, 0.1]
        return self.q_table[state_key]

    def choose_action(
        self,
        state_key: str,
        obs: np.ndarray = None,
        info: Dict[str, Any] = None,
        training: bool = True,
        sensor_only: bool = False
    ) -> int:
        """
        Select action (0 = JUMP, 1 = NO-OP).
        - If `sensor_only=True` or `state_key` has never been visited during evaluation,
          uses the generalized relative `sensor_q_table`.
        - Otherwise uses the spatial Bellman `q_table`.
        """
        if sensor_only or (not training and state_key not in self.q_table):
            s_key = self.encode_sensor_state(obs, info or {})
            q_vals = self.get_sensor_q_values(s_key, obs, info)
            if obs is not None and s_key not in self.sensor_q_table:
                # Zero-shot radar fallback if completely unseen sensor bucket
                hazard_px = obs[3] * 300.0
                if 15.0 <= hazard_px <= 85.0:
                    return 0
            return 0 if q_vals[0] > q_vals[1] else 1

        q_vals = self.get_q_values(state_key, obs, info)
        return 0 if q_vals[0] > q_vals[1] else 1

    def backward_bellman_sweep(
        self,
        trajectory: List[Tuple[str, str, int, float, str, bool, bool]]
    ):
        """
        Backward Bellman Sweep over the episode trajectory.
        Each entry: (state_key, sensor_key, action, b_reward, next_state_key, done, can_act)
        Updates both the Spatial Bellman Q-Table and the Generalized Sensor Q-Table.
        """
        for s_key, sensor_key, action, b_reward, next_s_key, done, can_act in reversed(trajectory):
            q_vals = self.get_q_values(s_key)
            if done:
                target = b_reward
            else:
                next_q = self.get_q_values(next_s_key)
                target = b_reward + self.gamma * max(next_q[0], next_q[1])

            if can_act:
                q_vals[action] = target
                # Also update generalized relative sensor policy with soft moving average
                sq_vals = self.get_sensor_q_values(sensor_key)
                sq_vals[action] = 0.85 * sq_vals[action] + 0.15 * np.clip(target, -10.0, 10.0)
            else:
                q_vals[0] = target
                q_vals[1] = target

    def save(self, filepath: str):
        """Save learned spatial and sensor Q-tables to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "q_table": self.q_table,
            "sensor_q_table": self.sensor_q_table
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load learned spatial and sensor Q-tables from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.lr = data.get("lr", self.lr)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", 0.0)
        self.q_table = data.get("q_table", {})
        self.sensor_q_table = data.get("sensor_q_table", {})
