#!/usr/bin/env python3
import os, sys, traceback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TimeLimit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.single_agent_crosswalk_env import SingleAgentCrosswalkEnv
from utils.logging_callback import RewardLoggingCallback

# Paths
SUMO_NET   = "intersection/environment.net.xml"
SUMO_ROUTE = "intersection/episode_routes.rou.xml"
LOG_ROOT   = "logs/ablation_crosswalk/"
MODEL_ROOT = "models/"

# Ensure log dirs exist
os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)

# Parameter grid
alpha_values = [0.01, 0.05]
gamma_values = [.1]
ped_weights  = [.35, .4]

exp_id = 0
for alpha in alpha_values:
    for gamma in gamma_values:
        for ped_w in ped_weights:
            veh_w = 1.0 - ped_w
            exp_name = f"exp_{exp_id}_a{alpha}_g{gamma}_pw{ped_w}_vw{veh_w}"
            log_dir = os.path.join(LOG_ROOT, exp_name)
            model_path = os.path.join(MODEL_ROOT, f"{exp_name}.zip")
            crash_log = os.path.join(log_dir, "crash.log")

            os.makedirs(log_dir, exist_ok=True)
            with open(crash_log, "w") as f:
                f.write(f"🚦 SUMO RL Log Start for {exp_name}\n\n")

            def log_exception(msg, exc):
                with open(crash_log, "a") as f:
                    f.write(f"\n❌ {msg}\n")
                    traceback.print_exc(file=f)

            try:
                def make_env():
                    env = SingleAgentCrosswalkEnv(
                        net_file=SUMO_NET,
                        route_file=SUMO_ROUTE,
                        sumo_binary="sumo",
                        use_gui=False,
                        max_steps=1000,
                        alpha=alpha,
                        gamma=gamma,
                        ped_weight=ped_w,
                        veh_weight=veh_w
                    )
                    return Monitor(TimeLimit(env, max_episode_steps=1000),
                                   filename=os.path.join(log_dir, "monitor.csv"))

                env = DummyVecEnv([make_env])
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    verbose=0,
                    device="cpu",
                    tensorboard_log=log_dir,
                    n_steps=1000,
                    batch_size=250,
                    learning_rate=1e-4,
                    gamma=0.99
                )

                callback = RewardLoggingCallback(log_dir=log_dir)
                model.learn(total_timesteps=10_000, callback=callback, progress_bar=True)
                model.save(model_path)
                print(f"✅ Finished {exp_name}")

            except Exception as e:
                log_exception("Training failed", e)
                print(f"❌ Failed {exp_name}")

            exp_id += 1
