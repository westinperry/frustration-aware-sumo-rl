#!/usr/bin/env python3
import os
import sys
import numpy as np

import traci
from stable_baselines3 import PPO

# Add env module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.single_agent_crosswalk_env import SingleAgentCrosswalkEnv

# ─── Configuration ─────────────────────────────────────
SUMO_NET     = "intersection/environment.net.xml"
SUMO_ROUTE   = "intersection/episode_routes.rou.xml"
MODEL_PATH   = "models/static_baseline.zip"
USE_GUI      = False
MAX_STEPS    = 1000
N_EPISODES   = 1

PHASE_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]
PHASE_DURATION = {0: 40, 1: 10, 2: 5, 3: 15, 4: 40, 5: 10, 6: 5, 7: 3}
# ───────────────────────────────────────────────────────

def run_static_baseline_eval(n_episodes=N_EPISODES):
    env = SingleAgentCrosswalkEnv(
        net_file=SUMO_NET,
        route_file=SUMO_ROUTE,
        sumo_binary="sumo-gui" if USE_GUI else "sumo",
        use_gui=USE_GUI,
        max_steps=MAX_STEPS,
        alpha=0.1,
        gamma=0.0,
        ped_weight=.5,
        veh_weight=.5
    )

    model = PPO.load(MODEL_PATH, device="cpu")
    print(f"📦 Loaded model from: {MODEL_PATH}")
    all_rewards = []

    try:
        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            step_count = 0
            phase_idx = 0

            while not done:
                # Ignore model action, use static phase cycling
                static_phase = PHASE_ORDER[phase_idx % len(PHASE_ORDER)]
                env._set_phase_and_step(static_phase)
                phase_idx += 1

                obs = env._get_observation()
                reward = env._compute_reward()
                done = env._check_termination()

                total_reward += reward
                step_count += 1

            print(f"Episode {ep}: Reward = {total_reward:.1f}")
            all_rewards.append(total_reward)

    finally:
        env.close()

    print("\n=== Static Baseline Evaluation Results ===")
    print(f"Mean Reward over {n_episodes} episodes: {np.mean(all_rewards):.1f}")
    print(f"Std  Reward over {n_episodes} episodes: {np.std(all_rewards):.1f}")

if __name__ == "__main__":
    run_static_baseline_eval()
