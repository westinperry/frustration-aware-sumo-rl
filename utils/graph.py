import pandas as pd
import matplotlib.pyplot as plt

# Path to your Monitor CSV
monitor_path = "logs/ppo_crosswalk/monitor.csv"

# 1) Load the data (skip the JSON metadata line)
df = pd.read_csv(monitor_path, comment="#")

# 2) Raw Episode Reward over Wall‑Clock Time
plt.figure()
plt.plot(df["t"], df["r"])
plt.xlabel("Time (s)")
plt.ylabel("Episode Reward")
plt.title("Episode Reward over Time")
plt.tight_layout()
plt.show()

# 3) Compute a 5‑episode rolling average of reward
df["rolling_r"] = df["r"].rolling(window=5, min_periods=1).mean()

# 4) Smoothed Episode Reward vs Episode #
plt.figure()
plt.plot(df.index, df["rolling_r"])
plt.xlabel("Episode #")
plt.ylabel("Reward (5‑ep MA)")
plt.title("Smoothed Episode Reward")
plt.tight_layout()
plt.show()
