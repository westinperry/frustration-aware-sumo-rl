#!/usr/bin/env python3
import os
import sys
import traceback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.single_agent_crosswalk_env import SingleAgentCrosswalkEnv
from utils.logging_callback import RewardLoggingCallback

SUMO_NET = "intersection/environment.net.xml"
SUMO_ROUTE = "intersection/episode_routes.rou.xml"
CRASH_LOG = "logs/ppo_crosswalk_crash.log"

# Make sure logs dir exists
os.makedirs("logs", exist_ok=True)

# ---- START LOG FILE ----
with open(CRASH_LOG, "w") as f:
    f.write("🚦 SUMO RL Training Log Start\n\n")

def log_exception(msg, exc):
    with open(CRASH_LOG, "a") as f:
        f.write(f"\n❌ {msg}\n")
        traceback.print_exc(file=f)

try:
    # Wrap environment init too
    env = DummyVecEnv([
        lambda: Monitor(SingleAgentCrosswalkEnv(
            net_file=SUMO_NET,
            route_file=SUMO_ROUTE,
            sumo_binary="sumo",
            use_gui=True,
            max_steps=1000
        ))
    ])

    # Sanity check
    check_env(env.envs[0], warn=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="logs/ppo_crosswalk/",
        device="cpu"
    )

    callback = RewardLoggingCallback(log_dir="logs/ppo_crosswalk/")
    model.learn(total_timesteps=1000, callback=callback)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_crosswalk")

except Exception as e:
    log_exception("Training failed", e)
    raise
