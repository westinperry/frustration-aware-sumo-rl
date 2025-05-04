#!/usr/bin/env python3
import os
import sys
import traceback
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

import traci

# Add env module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.single_agent_crosswalk_env import SingleAgentCrosswalkEnv

# ─── Configuration ─────────────────────────────────────
SUMO_NET      = "intersection/environment.net.xml"
SUMO_ROUTE    = "intersection/episode_routes.rou.xml"
MODEL_PATH    = "models/static_baseline.zip"
LOG_DIR       = "logs/static_baseline/"
os.makedirs("models", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MAX_STEPS     = 1000
USE_GUI       = False
PHASE_ORDER   = [0, 1, 2, 3, 4, 5, 6, 7]
PHASE_DURATION = {0: 40, 1: 10, 2: 5, 3: 15, 4: 40, 5: 10, 6: 5, 7: 3}
# ────────────────────────────────────────────────────────

class StaticTLBaselineEnv(Env):
    def __init__(self):
        super().__init__()
        self.env = SingleAgentCrosswalkEnv(
            net_file=SUMO_NET,
            route_file=SUMO_ROUTE,
            sumo_binary="sumo-gui" if USE_GUI else "sumo",
            use_gui=USE_GUI,
            max_steps=MAX_STEPS,
            alpha=0.05,
            gamma=0.05,
            ped_weight=0.5,
            veh_weight=0.5
        )
        self.action_space = Discrete(1)  # dummy single action
        self.observation_space = self.env.observation_space
        self.phase_idx = 0

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset()
        self.phase_idx = 0
        return obs, {}

    def step(self, action):
        phase = PHASE_ORDER[self.phase_idx % len(PHASE_ORDER)]
        self.env._set_phase_and_step(phase)
        self.phase_idx += 1
        obs = self.env._get_observation()
        reward = self.env._compute_reward()
        done = self.env._check_termination()
        return obs, reward, done, False, {}

    def close(self):
        self.env.close()


# ─── Train Static Baseline ───────────────────────────────
def train_static_baseline():
    try:
        def make_env():
            return Monitor(StaticTLBaselineEnv(), filename=os.path.join(LOG_DIR, "monitor.csv"))

        env = DummyVecEnv([make_env])
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device="cpu",
            tensorboard_log=LOG_DIR,
            n_steps=1000,
            batch_size=250,
            learning_rate=1e-4,
            gamma=0.99
        )

        model.learn(total_timesteps=1000)
        model.save(MODEL_PATH)
        print(f"✅ Static baseline model saved to: {MODEL_PATH}")

    except Exception as e:
        print("❌ Training failed")
        traceback.print_exc()

if __name__ == "__main__":
    train_static_baseline()
