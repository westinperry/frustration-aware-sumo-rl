#!/usr/bin/env python3
import os
import numpy as np
import sys

# allow importing your env module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.single_agent_crosswalk_env import SingleAgentCrosswalkEnv

# ── Configuration ───────────────────────────────────────────────
SUMO_NET      = "intersection/environment.net.xml"
SUMO_ROUTE    = "intersection/episode_routes.rou.xml"
N_EPISODES    = 1
MAX_STEPS     = 5000
USE_GUI       = False   # set True if you want to watch the GUI
# ────────────────────────────────────────────────────────────────

def run_random_baseline(n_episodes=N_EPISODES):
    env = SingleAgentCrosswalkEnv(
        net_file   = SUMO_NET,
        route_file = SUMO_ROUTE,
        sumo_binary= "sumo-gui",
        use_gui    = USE_GUI,
        max_steps  = MAX_STEPS
    )

    all_rewards = []
    for ep in range(1, n_episodes + 1):
        obs, _    = env.reset()
        done      = False
        total_rew = 0.0
        step_idx  = 0

        while not done:
            # random baseline
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            total_rew += reward
            step_idx += 1

        print(f"Episode {ep:2d}: total reward = {total_rew:.1f}")
        all_rewards.append(total_rew)

    env.close()
    mean_r = np.mean(all_rewards)
    std_r  = np.std(all_rewards)
    print("\n=== Random Baseline Results ===")
    print(f"Avg Reward over {n_episodes} episodes: {mean_r:.1f} ± {std_r:.1f}")

if __name__ == "__main__":
    run_random_baseline()
