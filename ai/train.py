import os
import sys
import json
import time
import tempfile
import argparse

# Set MPLCONFIGDIR to temp directory to avoid matplotlib cache permission warnings
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.env import AMCDashEnv
from ai.agent import QLearningDashAgent


def capture_engine_snapshot(env: AMCDashEnv, obs, info, state_key: str, parent_trajectory: list):
    """
    Save a snapshot of the environment at a safe ground checkpoint
    along with the parent trajectory from x=0 so Bellman sweeps remain unbroken.
    """
    p = env.engine.player
    obs_positions = [(s, s.rect.x, s.rect.y) for s in env.engine.elements]
    return {
        "state_key": state_key,
        "obs": obs.copy(),
        "info": dict(info),
        "parent_trajectory": list(parent_trajectory),
        "distance": env.engine.distance_traveled,
        "progress_px": env.engine.progress_px,
        "player_rect": p.rect.copy(),
        "player_vel": p.vel.copy(),
        "player_onGround": p.onGround,
        "player_isjump": p.isjump,
        "player_angle": p.angle,
        "player_coins": p.coins_collected,
        "obs_positions": obs_positions
    }


def restore_engine_snapshot(env: AMCDashEnv, snap: dict):
    """Restore environment state from a checkpoint snapshot."""
    env.engine.distance_traveled = snap["distance"]
    env.engine.progress_px = snap["progress_px"]
    env.engine.state = env.engine.STATE_PLAYING

    p = env.engine.player
    p.rect = snap["player_rect"].copy()
    p.vel = snap["player_vel"].copy()
    p.onGround = snap["player_onGround"]
    p.isjump = snap["player_isjump"]
    p.angle = snap["player_angle"]
    p.coins_collected = snap["player_coins"]
    p.died = False
    p.win = False

    env.engine.elements.empty()
    for sprite, x, y in snap["obs_positions"]:
        sprite.rect.x = x
        sprite.rect.y = y
        env.engine.elements.add(sprite)


def save_learning_curve_plot(history: list, level: int, output_path: str):
    """Generate and save a dual-panel RL Learning Curve chart for portfolio documentation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = [h["episode"] for h in history]
        progress = [h["progress"] for h in history]
        best_prog = [h["best_progress"] for h in history]
        q_states = [h["q_states"] for h in history]
        s_states = [h["sensor_states"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120)
        fig.patch.set_facecolor("#12141c")

        for ax in (ax1, ax2):
            ax.set_facecolor("#1b1e2b")
            ax.tick_params(colors="#d0d4e4")
            for spine in ax.spines.values():
                spine.set_color("#343952")
            ax.grid(True, linestyle="--", alpha=0.25, color="#8088a8")

        # Panel 1: Progress (%) vs Episode
        ax1.plot(episodes, progress, color="#4cc9f0", alpha=0.55, linewidth=1.3, label="Episode Progress (%)")
        ax1.plot(episodes, best_prog, color="#f72585", linewidth=2.4, label="Best Frontier Progress (%)")
        ax1.axhline(100.0, color="#4ade80", linestyle=":", linewidth=1.5, label="Target (100% Clear)")
        ax1.set_title(f"AMC Dash Level {level} - RL Learning Curve", color="#ffffff", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Episode", color="#d0d4e4")
        ax1.set_ylabel("Level Completion (%)", color="#d0d4e4")
        ax1.set_ylim(0, 105)
        ax1.legend(facecolor="#1b1e2b", edgecolor="#343952", labelcolor="#ffffff", loc="lower right")

        # Panel 2: Knowledge Table Growth
        ax2.plot(episodes, q_states, color="#fca311", linewidth=2.2, label="Spatial States (Q-Table)")
        ax2.plot(episodes, s_states, color="#4ade80", linewidth=2.0, label="Generalized Sensor States")
        ax2.set_title(f"State-Space Exploration (Level {level})", color="#ffffff", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Episode", color="#d0d4e4")
        ax2.set_ylabel("Learned States Count", color="#d0d4e4")
        ax2.legend(facecolor="#1b1e2b", edgecolor="#343952", labelcolor="#ffffff", loc="upper left")

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"📊 Grafik kurva pembelajaran disimpan ke: {output_path}")
    except Exception as e:
        print(f"Info: Gagal membuat plot matplotlib ({e})")


def train_q_agent(args):
    """
    Train the Dual-Layer Q-Learning + Backward Bellman Sweep + Go-Explore Checkpoint Agent.
    """
    print("=" * 70)
    print(f"🚀 MEMULAI PELATIHAN RL AGENT - AMC DASH (LEVEL {args.level})")
    print("   Algoritma : Dual-Layer Q-Learning (Sensor Generalization + Spatial Bellman)")
    print("   Fitur     : Mid-Air Action Masking + Go-Explore Checkpoints")
    print("=" * 70)

    env = AMCDashEnv(level=args.level, headless=not args.render, obs_mode="state")
    agent = QLearningDashAgent(
        learning_rate=1.0,
        discount_factor=0.995
    )

    model_path = os.path.join(args.checkpoint_dir, f"q_agent_level_{args.level}.json")
    if args.resume and os.path.exists(model_path):
        agent.load(model_path)
        print(f"Melanjutkan model dari {model_path} ({len(agent.q_table)} Q-states)")

    best_progress = 0.0
    start_time = time.time()
    total_steps = 0
    checkpoints = []
    history = []

    for ep in range(1, args.episodes + 1):
        # Pop any checkpoints that have been mathematically proven to be dead-ends
        while checkpoints:
            ckpt_key = checkpoints[-1]["state_key"]
            ckpt_q = agent.get_q_values(ckpt_key)
            if max(ckpt_q[0], ckpt_q[1]) < 0.0:
                checkpoints.pop()
            else:
                break

        if checkpoints:
            restore_engine_snapshot(env, checkpoints[-1])
            obs = checkpoints[-1]["obs"].copy()
            info = dict(checkpoints[-1]["info"])
            trajectory = list(checkpoints[-1]["parent_trajectory"])
        else:
            obs, info = env.reset()
            trajectory = []  # (state_key, sensor_key, action, b_reward, next_state_key, done, can_act)

        done = False

        while not done:
            state_key = agent.encode_state(obs, info)
            sensor_key = agent.encode_sensor_state(obs, info)
            on_ground = info["on_ground"]
            touching_orb = info.get("touching_orb", False)
            dist = info["distance"]

            # Save a checkpoint every ~120px on ground if this state still has viable actions
            if on_ground:
                q_curr = agent.get_q_values(state_key, obs, info)
                if max(q_curr[0], q_curr[1]) >= 0.0:
                    last_dist = checkpoints[-1]["distance"] if checkpoints else 0
                    if dist - last_dist >= 120:
                        checkpoints.append(capture_engine_snapshot(env, obs, info, state_key, trajectory))

            # Action physically affects trajectory only when on ground OR touching a Jump Orb
            can_act = on_ground or touching_orb

            action = agent.choose_action(state_key, obs, info=info, training=True)
            next_obs, _, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_state_key = agent.encode_state(next_obs, next_info)

            if next_info["died"]:
                b_reward = -100.0
            elif next_info["won"]:
                b_reward = 1000.0
            else:
                b_reward = 0.0

            trajectory.append((state_key, sensor_key, action, b_reward, next_state_key, done, can_act))

            obs = next_obs
            info = next_info
            total_steps += 1

            if args.render:
                env.render()

        # Perform exact Backward Bellman Sweep over the episode trajectory
        agent.backward_bellman_sweep(trajectory)

        progress = info["progress"]
        if progress > best_progress:
            best_progress = progress
            agent.save(model_path)

        history.append({
            "episode": ep,
            "progress": round(progress, 2),
            "best_progress": round(best_progress, 2),
            "q_states": len(agent.q_table),
            "sensor_states": len(agent.sensor_q_table),
            "total_steps": total_steps
        })

        if ep % 10 == 0 or ep == 1 or info["won"]:
            elapsed = time.time() - start_time
            fps = total_steps / max(0.001, elapsed)
            print(
                f"Episode {ep:3d}/{args.episodes} | "
                f"Progress: {progress:5.1f}% (Best: {best_progress:5.1f}%) | "
                f"Q-States: {len(agent.q_table):4d} | "
                f"Sensor-States: {len(agent.sensor_q_table):3d} | "
                f"Speed: {fps:5.0f} FPS"
            )

        if info["won"]:
            # Verify full start-to-finish run from x=0 without checkpoints!
            v_obs, v_info = env.reset()
            v_done = False
            v_steps = 0
            while not v_done:
                v_key = agent.encode_state(v_obs, v_info)
                v_act = agent.choose_action(v_key, v_obs, info=v_info, training=False)
                v_obs, _, v_term, v_trunc, v_info = env.step(v_act)
                v_done = v_term or v_trunc
                v_steps += 1

            if v_info["won"]:
                agent.save(model_path)
                elapsed = time.time() - start_time
                print("=" * 70)
                print(
                    f"🏆 LEVEL {args.level} BERHASIL DITAMATKAN (100.0% VERIFIED RUN)!\n"
                    f"   Total Episode  : {ep}\n"
                    f"   Total Langkah  : {v_steps} frames (Koin: {v_info['coins']})\n"
                    f"   Spatial States : {len(agent.q_table)} states\n"
                    f"   Sensor States  : {len(agent.sensor_q_table)} generalized patterns\n"
                    f"   Waktu Training : {elapsed:.2f} detik ({total_steps / max(0.01, elapsed):.0f} FPS)\n"
                    f"   File Model     : {model_path}"
                )
                print("=" * 70)
                break
            else:
                fail_dist = v_info["distance"]
                checkpoints = [c for c in checkpoints if c["distance"] < fail_dist - 60]

    # Save training metrics and plot learning curve
    hist_path = os.path.join(args.checkpoint_dir, f"training_history_level_{args.level}.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_path = os.path.join("images", f"learning_curve_level_{args.level}.png")
    save_learning_curve_plot(history, args.level, plot_path)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train Reinforcement Learning Agent on AMC Dash")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2], help="Level to train on (1 or 2)")
    parser.add_argument("--episodes", type=int, default=500, help="Maximum training episodes")
    parser.add_argument("--checkpoint-dir", type=str, default="./train/", help="Directory to save learned model")
    parser.add_argument("--render", action="store_true", help="Render game window during training")
    parser.add_argument("--resume", action="store_true", help="Resume from saved model")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_q_agent(args)


if __name__ == "__main__":
    main()
