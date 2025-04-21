# train_ppo.py
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.single_agent_crosswalk_env import SingleAgentCrosswalkEnv
from utils.logging_callback import RewardLoggingCallback

SUMO_NET = "intersection/environment.net.xml"
SUMO_ROUTE = "intersection/episode_routes.rou.xml"
SUMO_CONFIG = "intersection/sumo_config.sumocfg"

# Environment init
env = DummyVecEnv([lambda: SingleAgentCrosswalkEnv(
    net_file=SUMO_NET,
    route_file=SUMO_ROUTE,
    sumo_binary="sumo",   # or "sumo-gui" to visualize
    use_gui=False, # or True
    max_steps=1000
)])

# Optional: sanity check
check_env(env.envs[0], warn=True)

# Initialize PPO agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="logs/ppo_crosswalk/",
    device="cpu"
)

# Train
callback = RewardLoggingCallback(log_dir="logs/ppo_crosswalk/")
model.learn(total_timesteps=10000, callback=callback)


# Save
os.makedirs("models", exist_ok=True)
model.save("models/ppo_crosswalk")
