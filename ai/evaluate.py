import os
import sys
import time
import argparse
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.env import AMCDashEnv
from ai.agent import QLearningDashAgent


def evaluate_ai():
    parser = argparse.ArgumentParser(description="Evaluate trained RL Agent playing AMC Dash")
    parser.add_argument("--model", type=str, default="", help="Path to trained model (.json)")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2], help="Level to evaluate (1 or 2)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--fps", type=int, default=60, help="Playback speed (FPS)")
    parser.add_argument("--headless", action="store_true", help="Run without opening a display window")
    parser.add_argument("--sensor-only", action="store_true", help="Use only generalized position-independent radar sensors")
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        model_path = os.path.join("train", f"q_agent_level_{args.level}.json")

    if not os.path.exists(model_path):
        print(f"Model belum ditemukan di: {model_path}")
        print(f"Jalankan training terlebih dahulu: python ai/train.py --level {args.level}")
        return

    print(f"Memuat model RL: {model_path}")
    agent = QLearningDashAgent()
    agent.load(model_path)
    print(f"Model berhasil dimuat ({len(agent.q_table)} Spatial Q-states, {len(agent.sensor_q_table)} Sensor states).")

    env = AMCDashEnv(level=args.level, headless=args.headless, obs_mode="state")
    if not args.headless:
        env.engine.sound_enabled = True
        env.engine.play_music(args.level - 1)

    radar_enabled = True

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        print(f"\n--- Evaluasi Episode {ep}/{args.episodes} (Level {args.level}) ---")

        while not done:
            if not args.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return
                        elif event.key == pygame.K_r:
                            radar_enabled = not radar_enabled

            state_key = agent.encode_state(obs, info)
            q_vals = agent.get_q_values(state_key, obs, info)
            action = agent.choose_action(
                state_key, obs, info=info, training=False, sensor_only=args.sensor_only
            )

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if not args.headless:
                ai_overlay = {
                    "radar_targets": info.get("radar_targets"),
                    "q_vals": q_vals,
                    "action": action,
                    "ai_enabled": True,
                    "radar_enabled": radar_enabled
                }
                env.engine.render(ai_overlay=ai_overlay)

        status = "MENANG (100% Complete)! 🏆" if info.get("won") else f"KALAH pada {info.get('progress', 0):.1f}% 💥"
        print(
            f"Episode {ep} Selesai: {status} | "
            f"Steps: {steps} | Koin: {info.get('coins', 0)} | "
            f"Total Reward: {total_reward:.1f}"
        )

        if not args.headless and info.get("won"):
            env.engine.render()
            time.sleep(2.0)

    env.close()


if __name__ == "__main__":
    evaluate_ai()
