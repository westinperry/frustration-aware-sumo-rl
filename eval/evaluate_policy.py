# evaluate_policy.py
import numpy as np
from stable_baselines3 import PPO
from env.single_agent_crosswalk_env import SingleAgentCrosswalkEnv

SUMO_NET = "intersection/environment.net.xml"
SUMO_ROUTE = "intersection/episode_routes.rou.xml"
MODEL_PATH = "models/ppo_crosswalk"

# Eval config
eval_episodes = 10
max_steps = 1000

env = SingleAgentCrosswalkEnv(
    net_file=SUMO_NET,
    route_file=SUMO_ROUTE,
    sumo_binary="sumo",
    use_gui=False,
    max_steps=max_steps
)

model = PPO.load(MODEL_PATH)

all_rewards = []
all_ped_waits = []
all_veh_delays = []

for ep in range(eval_episodes):
    obs = env.reset()
    done = False
    ep_reward = 0
    ped_wait_total = 0
    veh_delay_total = 0

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        ep_reward += reward

        # Optional: compute peds + vehicle metrics at each step
        for pid in env.crosswalk_ids:
            ped_ids = env._get_person_ids(pid)
            for person_id in ped_ids:
                ped_wait_total += env._get_person_wait(person_id)

        for edge in env.vehicle_edges:
            veh_delay_total += env._get_edge_wait(edge)

        if done:
            break

    all_rewards.append(ep_reward)
    all_ped_waits.append(ped_wait_total)
    all_veh_delays.append(veh_delay_total)

print("=== Evaluation Results ===")
print(f"Avg Reward: {np.mean(all_rewards):.2f}")
print(f"Avg Ped Wait: {np.mean(all_ped_waits):.2f}")
print(f"Avg Veh Delay: {np.mean(all_veh_delays):.2f}")

env.close()