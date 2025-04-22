# logging_callback.py
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os

class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_dir: str = "logs", verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                self.episode_rewards.append(ep_rew)
                self.logger.record("rollout/ep_rew_mean", ep_rew)
        return True

    def _on_training_end(self) -> None:
        path = os.path.join(self.log_dir, "episode_rewards.txt")
        np.savetxt(path, self.episode_rewards, fmt="%.4f")
        if self.verbose:
            print(f"[LoggingCallback] Saved episode rewards to {path}")

