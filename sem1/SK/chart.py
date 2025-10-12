import pandas as pd
import matplotlib.pyplot as plt

data = {
    "tick": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "left": [10.33, 10.33, 10.33, 10.33, 10.67, 10.67, 9.67, 9.67, 9.67, 7.00, 5.33],
    "undecided": [
        51.67,
        51.67,
        51.67,
        51.67,
        49.33,
        49.33,
        55.67,
        55.67,
        55.67,
        64.33,
        70.67,
    ],
    "right": [
        24.67,
        24.67,
        24.67,
        24.67,
        26.67,
        26.67,
        25.33,
        25.33,
        25.33,
        20.67,
        17.33,
    ],
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.plot(df["tick"], df["left"], label="Left", marker="o")
plt.plot(df["tick"], df["undecided"], label="Undecided", marker="o")
plt.plot(df["tick"], df["right"], label="Right", marker="o")
plt.xlabel("Time (ticks)")
plt.ylabel("Percentage of Agents")
plt.title("Opinion Distribution Over Time")
plt.legend()
plt.grid()
# plt.show()

polarization_data = {
    "tick": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "polarization-index": [
        0.198,
        0.198,
        0.198,
        0.198,
        0.217,
        0.217,
        0.198,
        0.198,
        0.198,
        0.176,
        0.155,
    ],
}
df_polarization = pd.DataFrame(polarization_data)

plt.figure(figsize=(10, 6))
plt.plot(
    df_polarization["tick"],
    df_polarization["polarization-index"],
    marker="o",
    color="purple",
)
plt.xlabel("Time (ticks)")
plt.ylabel("Polarization Index")
plt.title("Polarization Index Over Time")
plt.grid()
# plt.show()

opinion_changes_data = {
    "tick": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "total-opinion-changes": [154, 154, 154, 154, 165, 165, 171, 171, 171, 175, 175],
}
df_opinion_changes = pd.DataFrame(opinion_changes_data)

plt.figure(figsize=(10, 6))
plt.bar(
    df_opinion_changes["tick"],
    df_opinion_changes["total-opinion-changes"],
    color="orange",
)
plt.xlabel("Time (ticks)")
plt.ylabel("Total Opinion Changes")
plt.title("Total Opinion Changes Over Time")
plt.grid(axis="y")
# plt.show()

fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot opinion distribution
ax1.plot(df["tick"], df["left"], label="Left", marker="o", color="blue")
ax1.plot(df["tick"], df["undecided"], label="Undecided", marker="o", color="green")
ax1.plot(df["tick"], df["right"], label="Right", marker="o", color="red")
ax1.set_xlabel("Time (ticks)")
ax1.set_ylabel("Percentage of Agents")
ax1.legend(loc="upper left")

# Add polarization index on a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(
    df_polarization["tick"],
    df_polarization["polarization-index"],
    label="Polarization Index",
    marker="o",
    color="purple",
)
ax2.set_ylabel("Polarization Index")
ax2.legend(loc="upper right")

plt.title("Combined Metrics Over Time")
plt.grid()
plt.show()
