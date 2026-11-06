import os
import sys
import argparse
import pygame

from game.engine import GameEngine
from ai.env import AMCDashEnv
from ai.agent import QLearningDashAgent


def parse_args():
    parser = argparse.ArgumentParser(description="AMC Dash - Geometry Dash Clone & RL AI in Python")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2], help="Level to start (1 or 2)")
    parser.add_argument("--no-sound", action="store_true", help="Disable music and sound effects")
    parser.add_argument("--ai", action="store_true", help="Start with AI Autoplay enabled")
    return parser.parse_args()


def load_agent_for_level(level_idx: int) -> QLearningDashAgent:
    agent = QLearningDashAgent()
    model_path = os.path.join("train", f"q_agent_level_{level_idx + 1}.json")
    if os.path.exists(model_path):
        agent.load(model_path)
    return agent


def run_game():
    args = parse_args()
    env = AMCDashEnv(level=args.level, headless=False, obs_mode="state")
    engine = env.engine
    engine.sound_enabled = not args.no_sound
    engine.state = GameEngine.STATE_MENU

    ai_enabled = args.ai
    radar_enabled = True
    agent = load_agent_for_level(engine.current_level_idx)

    obs = env._get_state_observation()
    info = env._build_info()

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if engine.state == GameEngine.STATE_PLAYING:
                        engine.state = GameEngine.STATE_MENU
                    else:
                        running = False
                        break

                elif event.key == pygame.K_a:
                    ai_enabled = not ai_enabled

                elif event.key == pygame.K_r:
                    radar_enabled = not radar_enabled

                if engine.state == GameEngine.STATE_MENU:
                    if event.key == pygame.K_1:
                        engine.load_level(0)
                        agent = load_agent_for_level(0)
                    elif event.key == pygame.K_2:
                        engine.load_level(1)
                        agent = load_agent_for_level(1)
                    elif event.key == pygame.K_SPACE:
                        engine.reset_game(new_attempt=False)
                        obs = env._get_state_observation()
                        info = env._build_info()

                elif engine.state == GameEngine.STATE_GAMEOVER:
                    if event.key == pygame.K_SPACE:
                        engine.reset_game(new_attempt=True)
                        obs = env._get_state_observation()
                        info = env._build_info()

                elif engine.state == GameEngine.STATE_VICTORY:
                    if event.key == pygame.K_SPACE:
                        next_lvl = (engine.current_level_idx + 1) % 2
                        engine.current_level_idx = next_lvl
                        agent = load_agent_for_level(next_lvl)
                        engine.reset_game(new_attempt=False)
                        obs = env._get_state_observation()
                        info = env._build_info()

        if not running:
            break

        ai_overlay = None
        if engine.state == GameEngine.STATE_PLAYING:
            state_key = agent.encode_state(obs, info)
            q_vals = agent.get_q_values(state_key, obs, info)
            ai_action = agent.choose_action(state_key, obs, info=info, training=False)

            if ai_enabled:
                jump_input = (ai_action == 0)
                active_action = ai_action
            else:
                keys = pygame.key.get_pressed()
                jump_input = bool(keys[pygame.K_SPACE] or keys[pygame.K_UP])
                active_action = 0 if jump_input else 1

            engine.step(jump_input)
            obs = env._get_state_observation()
            info = env._build_info(won=engine.player.win, died=engine.player.died)

            ai_overlay = {
                "radar_targets": info.get("radar_targets"),
                "q_vals": q_vals,
                "action": active_action,
                "ai_enabled": ai_enabled,
                "radar_enabled": radar_enabled
            }

        engine.render(ai_overlay=ai_overlay)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run_game()
