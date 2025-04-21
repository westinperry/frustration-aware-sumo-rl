#!/usr/bin/env python3
import os
import sys

# 1) Add project root so `env/` can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO
from env.single_agent_crosswalk_env import SingleAgentCrosswalkEnv

# Paths
SUMO_NET    = "intersection/environment.net.xml"
SUMO_ROUTE  = "intersection/episode_routes.rou.xml"
MODEL_PATH  = "models/ppo_crosswalk"

# Eval config
eval_episodes = 10
max_steps     = 1000

# Create the env
env = SingleAgentCrosswalkEnv(
    net_file   = SUMO_NET,
    route_file = SUMO_ROUTE,
    sumo_binary= "sumo",
    use_gui    = False,
    max_steps  = max_steps
)

# Load the trained agent
model = PPO.load(MODEL_PATH)

all_rewards    = []
all_ped_waits  = []
all_veh_delays = []

for ep in range(eval_episodes):
    # 2) Unpack reset() properly
    obs, info = env.reset()
    done      = False
    ep_reward = 0
    ped_wait_total = 0
    veh_delay_total = 0

    for step in range(max_steps):
        # 3) Only pass the numpy obs (not the (obs,info) tuple)
        action, _ = model.predict(obs, deterministic=True)

        # 4) Unpack the full step return
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward

        # Accumulate metrics if you wish:
        for pid in env.crosswalk_ids:
            # you’d need helper methods in your env to get these;
            # or directly call traci again here
            pass

        if done or truncated:
            break

    all_rewards.append(ep_reward)
    # all_ped_waits.append(ped_wait_total)
    # all_veh_delays.append(veh_delay_total)

print("=== Evaluation Results ===")
print(f"Avg Reward:    {np.mean(all_rewards):.2f}")
# print(f"Avg Ped Wait:  {np.mean(all_ped_waits):.2f}")
# print(f"Avg Veh Delay: {np.mean(all_veh_delays):.2f}")

env.close()
