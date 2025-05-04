from stable_baselines3.common.callbacks import BaseCallback
import os
import csv

class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.csv_path = os.path.join(self.log_dir, "rewards.csv")

        # Prepare CSV file
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    reward = self.locals["infos"][idx].get("episode", {}).get("r", None)
                    if reward is not None:
                        episode_num = len(self.episode_rewards)
                        self.episode_rewards.append(reward)
                        with open(self.csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([episode_num, reward])
                        if self.verbose > 0:
                            print(f"[Callback] Episode {episode_num} Reward: {reward:.2f}")
        return True
