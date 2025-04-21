# test_env.py
from env.single_agent_crosswalk_env import SingleAgentCrosswalkEnv
import numpy as np

env = SingleAgentCrosswalkEnv(
    net_file="intersection/environment.net.xml",
    route_file="intersection/episode_routes.rou.xml",
    sumo_binary="sumo",  # or "sumo-gui" to visualize
    use_gui=False,
    max_steps=100
)

obs = env.reset()
print("Initial observation:", obs)

total_reward = 0
for t in range(100):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    total_reward += reward
    if done:
        break

print(f"Finished test rollout. Total reward: {total_reward:.2f}")
env.close()
