import pandas as pd
import matplotlib.pyplot as plt

# List of file paths (replace with your actual file paths)
file_paths = [
    "cc1.csv",
    "cc2.csv",
    "cc3.csv",
    "cc4.csv",
    "cc5.csv",
    "cc6.csv",
    "cc7.csv",
    "cc8.csv",
    "cc9.csv",
    "cc10.csv",
]

# Load all CSV files into a list of DataFrames
all_data = [pd.read_csv(file) for file in file_paths]

# Combine all DataFrames into one
combined_data = pd.concat(all_data)

# Group by 'tick' and compute the mean for each column
aggregated_data = combined_data.groupby("tick").mean().reset_index()

# Plot opinion distribution
plt.figure(figsize=(10, 6))
plt.plot(aggregated_data["tick"], aggregated_data["left"], label="Left", marker="o")
plt.plot(
    aggregated_data["tick"], aggregated_data["undecided"], label="Undecided", marker="o"
)
plt.plot(aggregated_data["tick"], aggregated_data["right"], label="Right", marker="o")
plt.xlabel("Time (ticks)")
plt.ylabel("Percentage of Agents")
plt.title("Opinion Distribution Over Time (Mean of 10 Runs)")
plt.legend()
plt.grid()
plt.show()

# Plot polarization index
plt.figure(figsize=(10, 6))
plt.plot(
    aggregated_data["tick"],
    aggregated_data["polarization-index"],
    marker="o",
    color="purple",
)
plt.xlabel("Time (ticks)")
plt.ylabel("Polarization Index")
plt.title("Polarization Index Over Time (Mean of 10 Runs)")
plt.grid()
plt.show()

# Plot total opinion changes
plt.figure(figsize=(10, 6))
plt.bar(
    aggregated_data["tick"], aggregated_data["total-opinion-changes"], color="orange"
)
plt.xlabel("Time (ticks)")
plt.ylabel("Total Opinion Changes")
plt.title("Total Opinion Changes Over Time (Mean of 10 Runs)")
plt.grid(axis="y")
plt.show()

# Compute standard deviation
std_data = combined_data.groupby("tick").std().reset_index()

# Plot with error bars (example for 'left' opinion)
plt.figure(figsize=(10, 6))
plt.errorbar(
    aggregated_data["tick"],
    aggregated_data["left"],
    yerr=std_data["left"],
    label="Left",
    marker="o",
    capsize=5,
)
plt.xlabel("Time (ticks)")
plt.ylabel("Percentage of Agents")
plt.title("Opinion Distribution Over Time (Mean ± Std Dev)")
plt.legend()
plt.grid()
plt.show()
