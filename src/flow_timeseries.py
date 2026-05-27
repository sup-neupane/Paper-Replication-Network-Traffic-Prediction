import numpy as np #type:ignore
import pandas as pd #type:ignore
import matplotlib.pyplot as plt #type:ignore

# Load data
data4 = np.load("data/raw/PEMS04.npz")["data"]
flow = data4[:, :, 0]  # (16992, 307) — traffic flow only

# Build timestamp index (5-min intervals)
time_index = pd.date_range(start="2018-01-01", periods=16992, freq="5min")

# We plot mean flow across all 307 sensors at each timestamp
# (single line, equivalent to your load_MW column)
mean_flow = flow.mean(axis=1)  # (16992,)

df = pd.DataFrame({"timestamp": time_index, "mean_flow": mean_flow})

# --- Filter to one week ---
start_date = "2018-01-01"
end_date   = "2018-01-07"

start = pd.Timestamp(start_date)
end   = pd.Timestamp(end_date)

week_df = df.query("@start <= timestamp <= @end")

# --- Plot ---
plt.figure(figsize=(14, 5))
plt.plot(week_df["timestamp"], week_df["mean_flow"], color="steelblue")

plt.title(f"Traffic Flow Pattern (Mean across 307 sensors): {start_date} to {end_date}")
plt.xlabel("Time")
plt.ylabel("Mean Vehicle Count (5-min interval)")
plt.tight_layout()
plt.savefig("results/flow_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()