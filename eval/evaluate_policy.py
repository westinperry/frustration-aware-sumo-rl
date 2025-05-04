#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from collections import Counter
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import traci
import pandas as pd

# so we can import env/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.single_agent_crosswalk_env import SingleAgentCrosswalkEnv

# Config
SUMO_NET = "intersection/environment.net.xml"
SUMO_ROUTE = "intersection/episode_routes.rou.xml"
MAX_STEPS = 1000
EVAL_EPISODES = 1

def print_stats(name, data):
    if data:
        print(f"\n{name} Wait Time Stats:")
        print(f"  Count   : {len(data)} (samples)")
        print(f"  Max     : {np.max(data):.2f} s")
        print(f"  Min     : {np.min(data):.2f} s")
        print(f"  Mean    : {np.mean(data):.2f} s")
        print(f"  Std Dev : {np.std(data):.2f} s")
        print(f"  95th %  : {np.percentile(data, 95):.2f} s")
    else:
        print(f"\n{name} Wait Time Stats: No data collected.")

def evaluate(model_path, use_gui=False, alpha=0.05, gamma=0.05, ped_weight=1.0, veh_weight=1.0):
    env = SingleAgentCrosswalkEnv(
        net_file=SUMO_NET,
        route_file=SUMO_ROUTE,
        sumo_binary="sumo-gui" if use_gui else "sumo",
        use_gui=use_gui,
        max_steps=MAX_STEPS,
        alpha=alpha,
        gamma=gamma,
        ped_weight=ped_weight,
        veh_weight=veh_weight
    )
    env = TimeLimit(env, max_episode_steps=MAX_STEPS)
    env = Monitor(env)

    model = PPO.load(model_path, device="cpu")
    all_rewards = []

    ped_wait_times = []
    veh_wait_times = []
    seen_peds = set()
    seen_vehs = set()
    total_actions = 0
    action_counter = Counter()

    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            total_actions += 1
            action_counter[int(action)] += 1

            # Collect wait times and unique counts
            for person_id in traci.person.getIDList():
                wait_time = traci.person.getWaitingTime(person_id)
                ped_wait_times.append(wait_time)
                seen_peds.add(person_id)

            for veh_id in traci.vehicle.getIDList():
                wait_time = traci.vehicle.getWaitingTime(veh_id)
                veh_wait_times.append(wait_time)
                seen_vehs.add(veh_id)

        print(f"Episode {ep + 1}: total reward = {total_reward:.1f}")
        all_rewards.append(total_reward)

    env.close()
    mean_r = np.mean(all_rewards)
    std_r  = np.std(all_rewards)

    print("\n=== Evaluation Results ===")
    print(f"Mean Reward over {EVAL_EPISODES} episodes: {mean_r:.1f} ± {std_r:.1f}")
    
    # Wait time stats
    print_stats("Pedestrian", ped_wait_times)
    print_stats("Vehicle", veh_wait_times)

    # Unique counts
    ped_count = len(seen_peds)
    veh_count = len(seen_vehs)

    print(f"\nUnique Pedestrians: {ped_count}")
    print(f"Unique Vehicles   : {veh_count}")
    print(f"Total Actions Taken: {total_actions}")

    print("Action Frequency Histogram:")
    for act, count in sorted(action_counter.items()):
        print(f"  Action {act}: {count} times")

    # Save to CSV
    os.makedirs("evaluation_results", exist_ok=True)
    model_name = os.path.basename(model_path).replace('.zip', '')

    # Reward and count stats
    with open(f"evaluation_results/{model_name}_metrics.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"mean_reward,{mean_r}\n")
        f.write(f"std_reward,{std_r}\n")
        f.write(f"unique_peds,{ped_count}\n")
        f.write(f"unique_vehs,{veh_count}\n")
        f.write(f"total_actions,{total_actions}\n")

    # Action histogram
    hist_df = pd.DataFrame(sorted(action_counter.items()), columns=["action", "count"])
    hist_df.to_csv(f"evaluation_results/{model_name}_action_histogram.csv", index=False)

    # Wait time data
    max_len = max(len(ped_wait_times), len(veh_wait_times))
    ped_wait_times_padded = ped_wait_times + [np.nan] * (max_len - len(ped_wait_times))
    veh_wait_times_padded = veh_wait_times + [np.nan] * (max_len - len(veh_wait_times))

    wait_df = pd.DataFrame({
        "ped_wait_times": ped_wait_times_padded,
        "veh_wait_times": veh_wait_times_padded
    })
    wait_df.to_csv(f"evaluation_results/{model_name}_wait_stats.csv", index=False)

    return mean_r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.00)
    parser.add_argument("--ped-weight", type=float, default=.5)
    parser.add_argument("--veh-weight", type=float, default=.5)
    args = parser.parse_args()

    EVAL_EPISODES = args.episodes
    evaluate(
        model_path=args.model,
        use_gui=args.gui,
        alpha=args.alpha,
        gamma=args.gamma,
        ped_weight=args.ped_weight,
        veh_weight=args.veh_weight
    )
