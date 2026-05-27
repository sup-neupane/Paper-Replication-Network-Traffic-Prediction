# exploration/step2_eda_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Load ──────────────────────────────────────────────────────────────────────
data4 = np.load("data/raw/pems04.npz")["data"]   # (16992, 307, 3)
flow  = data4[:, :, 0]                            # (16992, 307)
time_index = pd.date_range(start="2018-01-01", periods=16992, freq="5min")
df = pd.DataFrame(flow, index=time_index,
                  columns=[f"s{i}" for i in range(307)])

fig, axes = plt.subplots(5, 1, figsize=(16, 24))
fig.suptitle("PeMSD4 — Traffic Flow EDA", fontsize=16, fontweight="bold", y=0.98)

# ── Plot 1: Raw time series for a few sensors ─────────────────────────────────
ax = axes[0]
for sid in [0, 50, 150, 250]:
    ax.plot(df.index[:2016], df[f"s{sid}"].iloc[:2016],
            alpha=0.7, linewidth=0.8, label=f"Sensor {sid}")
ax.set_title("Plot 1 — Raw Flow: 4 sensors over 1 week (first 7 days)")
ax.set_xlabel("Time")
ax.set_ylabel("Vehicle Count")
ax.legend(loc="upper right", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
ax.grid(alpha=0.3)

# ── Plot 2: Average flow across all sensors ───────────────────────────────────
ax = axes[1]
mean_flow = df.mean(axis=1)   # average across all 307 sensors at each time step
ax.plot(df.index[:2016], mean_flow.iloc[:2016],
        color="steelblue", linewidth=0.8)
ax.set_title("Plot 2 — Mean Flow Across All 307 Sensors (1 week)")
ax.set_xlabel("Time")
ax.set_ylabel("Mean Vehicle Count")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
ax.grid(alpha=0.3)

# ── Plot 3: Average hourly pattern (daily cycle) ──────────────────────────────
ax = axes[2]
df_copy = df.copy()
df_copy["hour"] = df.index.hour + df.index.minute / 60
hourly = df_copy.groupby("hour").mean().mean(axis=1)  # avg over sensors then hours
ax.plot(hourly.index, hourly.values, color="darkorange", linewidth=2)
ax.fill_between(hourly.index, hourly.values, alpha=0.2, color="darkorange")
ax.set_title("Plot 3 — Average Daily Traffic Pattern (all sensors, all days)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Mean Vehicle Count")
ax.set_xticks(range(0, 25, 2))
ax.grid(alpha=0.3)

# ── Plot 4: Sensor mean distribution (busiest vs quietest) ───────────────────
ax = axes[3]
sensor_means = df.mean(axis=0)   # mean flow per sensor across all time
ax.hist(sensor_means.values, bins=40, color="teal", edgecolor="white", alpha=0.8)
ax.axvline(sensor_means.mean(), color="red", linestyle="--",
           linewidth=1.5, label=f"Mean = {sensor_means.mean():.1f}")
ax.set_title("Plot 4 — Distribution of Per-Sensor Mean Flow (all 307 sensors)")
ax.set_xlabel("Mean Vehicle Count per Sensor")
ax.set_ylabel("Number of Sensors")
ax.legend()
ax.grid(alpha=0.3)

# ── Plot 5: Heatmap — hour of day vs day of week ─────────────────────────────
ax = axes[4]
df_copy["dow"]  = df.index.dayofweek   # 0=Mon, 6=Sun
df_copy["hour_int"] = df.index.hour
# Average flow (across all sensors) per hour-of-day × day-of-week cell
heatmap_data = (df_copy.groupby(["dow", "hour_int"])
                        .mean()
                        .mean(axis=1)
                        .unstack(level=1))     # (7 days, 24 hours)

im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlOrRd",
               origin="upper")
ax.set_title("Plot 5 — Traffic Heatmap: Day of Week × Hour of Day")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Day of Week")
ax.set_yticks(range(7))
ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_xticks(range(0, 24, 2))
plt.colorbar(im, ax=ax, label="Mean Vehicle Count")

plt.tight_layout()
plt.savefig("results/eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to results/eda_plots.png")