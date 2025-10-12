# Wizualizacja Danych, Lab 1 - Jan Banot
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration
OUTPUT_DIR = Path(__file__).parent / "plots" / "lab2"
plot_counter = 0


def save_plot():
    """Save the current plot with an incremented number."""
    global plot_counter
    plot_counter += 1

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the figure
    filename = OUTPUT_DIR / f"{plot_counter}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {filename}")
    plt.close()


# Plot 1: Draw a line
x = list(range(0, 51))
y = [i * 3 for i in x]

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.xlabel("x - axis")
plt.ylabel("y - axis")
plt.title("Draw a line.")
plt.xlim(0, 50)
plt.ylim(0, 160)
save_plot()

# Plot 2: Financial data line chart
df = pd.read_csv(Path(__file__).parent / "data" / "fdata.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%m-%d-%y")

plt.figure(figsize=(10, 6))
plt.plot(df["Date"], df["Open"], label="Open", linewidth=2)
plt.plot(df["Date"], df["High"], label="High", linewidth=2)
plt.plot(df["Date"], df["Low"], label="Low", linewidth=2)
plt.plot(df["Date"], df["Close"], label="Close", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Alphabet Inc. w okresie od 3 października 2016 r. do 7 października 2016 r.")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_plot()

# Plot 3: Two lines with different styles
x = np.linspace(10, 30, 100)
line1 = 20 + (x - 20) * 0.9  # Dotted line (ascending)
line2 = 40 - (x - 10) * 1.4  # Dashed line (descending)

plt.figure(figsize=(8, 6))
plt.plot(
    x,
    line1,
    ":",
    color="navy",
    linewidth=2,
    label="line1-dotted",
    marker="o",
    markevery=10,
)
plt.plot(
    x,
    line2,
    "--",
    color="red",
    linewidth=2,
    label="line2-dashed",
    marker="s",
    markevery=10,
)

plt.xlabel("x - axis")
plt.ylabel("y - axis")
plt.title("Dwie linie w różnych stylach")
plt.xlim(10, 30)
plt.ylim(10, 40)
plt.legend()
plt.tight_layout()
save_plot()

# Plot 4: Scatter plot with two-dimensional data
x_blue = np.array([2.5, 3.5, 5, 6, 8])
y_blue = np.array([1, 5, 10, 18, 20])

x_red = np.array([4, 6.5, 7, 9])
y_red = np.array([2, 6, 11, 20])

plt.figure(figsize=(8, 6))
plt.scatter(x_blue, y_blue, color="navy", s=50, marker="*", label="Series 1")
plt.scatter(x_red, y_red, color="red", s=80, marker="o", label="Series 2")

plt.xlabel("x - axis")
plt.ylabel("y - axis")
plt.title("Wykres punktowy dla dowolnych wartości dwuwymiarowych:")
plt.xlim(0, 10)
plt.ylim(0, 30)
plt.tight_layout()
save_plot()

# Plot 5: Multiple lines with different styles and colors
x = np.linspace(0, 5, 50)

# Three different mathematical functions
y1 = x**3  # Cubic - red triangles
y2 = x**2  # Quadratic - blue squares
y3 = x  # Linear - gray dashed

plt.figure(figsize=(10, 6))

# Red triangles with line (cubic growth)
plt.plot(x, y1, color="darkred", linewidth=1.5, marker="^", markersize=6, markevery=3)

# Blue squares with line (quadratic growth)
plt.plot(x, y2, color="blue", linewidth=1.5, marker="s", markersize=5, markevery=3)

# Gray dashed line (linear growth)
plt.plot(x, y3, "--", color="gray", linewidth=1.5)

plt.xlabel("x - axis")
plt.ylabel("y - axis")
plt.title("Wykres składający się z kilku linii w różnych stylach i kolorach.")
plt.xlim(0, 5)
plt.ylim(0, 120)
plt.tight_layout()
save_plot()

# Plot 6: Multiple subplots in one workspace
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Sine wave
x = np.linspace(0, 2 * np.pi, 100)
axs[0, 0].plot(x, np.sin(x), color="blue")
axs[0, 0].set_title("Sine Wave")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("sin(x)")
axs[0, 0].grid(True, alpha=0.3)

# Subplot 2: Cosine wave
axs[0, 1].plot(x, np.cos(x), color="red")
axs[0, 1].set_title("Cosine Wave")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("cos(x)")
axs[0, 1].grid(True, alpha=0.3)

# Subplot 3: Scatter plot
x_scatter = np.random.rand(50) * 10
y_scatter = np.random.rand(50) * 10
axs[1, 0].scatter(x_scatter, y_scatter, color="green", alpha=0.6)
axs[1, 0].set_title("Random Scatter")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
axs[1, 0].grid(True, alpha=0.3)

# Subplot 4: Bar chart
categories = ["A", "B", "C", "D", "E"]
values = [23, 45, 56, 78, 32]
axs[1, 1].bar(categories, values, color="purple")
axs[1, 1].set_title("Bar Chart")
axs[1, 1].set_xlabel("Category")
axs[1, 1].set_ylabel("Value")
axs[1, 1].grid(True, alpha=0.3, axis="y")

plt.suptitle(
    "Kilka wykresów w ramach jednego obszaru roboczego",
    fontsize=14,
)
plt.tight_layout()
save_plot()

# Plot 7: Bar chart - Programming language popularity
languages = ["Java", "Python", "PHP", "JavaScript", "C#", "C++"]
popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]

plt.figure(figsize=(10, 7))
bars = plt.bar(languages, popularity, color="blue", edgecolor="black", linewidth=1.2)

plt.xlabel("Languages", fontsize=12)
plt.ylabel("Popularity", fontsize=12)
plt.title(
    "Popularity of Programming Language\nWorldwide, Oct 2017 compared to a year ago",
    fontsize=13,
)
plt.ylim(0, 25)
plt.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

plt.tight_layout()
save_plot()

# Plot 8: Horizontal bar chart - Programming language popularity
plt.figure(figsize=(10, 7))
bars = plt.barh(languages, popularity, color="blue", edgecolor="black", linewidth=1.2)

plt.ylabel("Languages", fontsize=12)
plt.xlabel("Popularity", fontsize=12)
plt.title(
    "Popularity of Programming Language\nWorldwide, Oct 2017 compared to a year ago (Horizontal)",
    fontsize=13,
)
plt.xlim(0, 25)
plt.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="x")

plt.tight_layout()
save_plot()

# Plot 9: Bar chart with different colors for each bar and custom positions
colors = ["red", "green", "blue", "orange", "purple", "cyan"]

# Custom bar positions with varied spacing
bar_positions = [0, 1.1, 3.5, 4, 6, 8.6]
# Custom width for each bar
bar_widths = [0.3, 1.1, 0.2, 0.4, 0.9, 0.5]

plt.figure(figsize=(10, 7))

# Plot each bar individually with custom width
bars = []
for pos, pop, width, color in zip(bar_positions, popularity, bar_widths, colors):
    bar = plt.bar(pos, pop, width=width, color=color, edgecolor="black", linewidth=1.2)
    bars.append(bar)

# Add values on top of bars
for i, (pos, value, width) in enumerate(zip(bar_positions, popularity, bar_widths)):
    plt.text(
        pos,
        value + 0.5,
        str(value),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.xticks(bar_positions, languages)
plt.xlabel("Languages", fontsize=12)
plt.ylabel("Popularity", fontsize=12)
plt.title(
    "Popularity of Programming Language\nWorldwide, Oct 2017 compared to a year ago (Multicolor with Custom Positions and Bar widths)",
    fontsize=13,
)
plt.ylim(0, 25)
plt.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

plt.tight_layout()
save_plot()

# Plot 10: Grouped bar chart - Scores by person separated by gender
men_means = [22, 30, 35, 35, 26]
women_means = [25, 32, 30, 35, 29]
persons = ["G1", "G2", "G3", "G4", "G5"]

x = np.arange(len(persons))
bar_width = 0.35

plt.figure(figsize=(10, 7))

# Create bars for men and women side by side
bars1 = plt.bar(
    x - bar_width / 2,
    men_means,
    bar_width,
    label="Men",
    color="darkgreen",
    edgecolor="black",
)
bars2 = plt.bar(
    x + bar_width / 2,
    women_means,
    bar_width,
    label="Women",
    color="red",
    edgecolor="black",
)

plt.xlabel("Person", fontsize=12)
plt.ylabel("Scores", fontsize=12)
plt.title("Scores by person", fontsize=13)
plt.xticks(x, persons)
plt.ylim(0, 35)
plt.legend()
plt.tight_layout()
save_plot()

# Plot 11: Grouped bar chart from DataFrame
# Create DataFrame from the example data
data = {
    "a": [2, 4, 6, 8, 10],
    "b": [4, 2, 4, 2, 2],
    "c": [8, 3, 7, 6, 4],
    "d": [5, 4, 4, 4, 3],
    "e": [7, 6, 7, 8, 3],
}
df_bars = pd.DataFrame(data, index=[2, 4, 6, 8, 10])

# Plot grouped bar chart
ax = df_bars.plot(kind="bar", figsize=(10, 7), edgecolor="black", linewidth=1)

# Customize the plot
ax.set_xlabel("Index", fontsize=12)
ax.set_ylabel("Values", fontsize=12)
ax.set_title("Grouped Bar Chart from DataFrame", fontsize=13)
ax.set_ylim(0, 8)
ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="y")
ax.legend(title="Columns", loc="upper right")

plt.tight_layout()
save_plot()

# Plot 12: Stacked horizontal bar chart with percentage labels
people = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
segments = 4

# Multi-dimensional data (8 people x 4 segments)
data = [
    [
        3.40022085,
        7.70632498,
        6.40979005,
        10.51648577,
        7.53300390,
        7.11235870,
        12.77792868,
        3.44773477,
    ],
    [
        11.24811149,
        5.03778215,
        6.65808464,
        12.32220677,
        7.45964195,
        6.79685302,
        7.24578743,
        3.69371847,
    ],
    [
        3.94253354,
        4.74763549,
        11.73529246,
        4.64655430,
        12.99521820,
        4.63832778,
        11.16849999,
        8.56883433,
    ],
    [
        4.24409799,
        12.71746612,
        11.37721690,
        9.00514257,
        10.47084185,
        10.97567589,
        3.98287652,
        8.80552122,
    ],
]

# Convert to numpy array and transpose to get (people x segments)
data_array = np.array(data).T

# Calculate row sums for percentage calculation
row_sums = data_array.sum(axis=1)

# Calculate percentages
data_percentage = (data_array / row_sums[:, np.newaxis]) * 100

# Colors for each segment
colors = ["red", "green", "lightgray", "purple"]

fig, ax = plt.subplots(figsize=(10, 8))

# Starting position for each bar
left = np.zeros(len(people))

# Plot each segment
for i in range(segments):
    bars = ax.barh(
        people,
        data_array[:, i],
        left=left,
        color=colors[i],
        edgecolor="black",
        linewidth=0.8,
    )

    # Add percentage labels
    for j, (bar, percentage) in enumerate(zip(bars, data_percentage[:, i])):
        width = bar.get_width()
        if percentage > 5:  # Only show label if segment is large enough
            ax.text(
                left[j] + width / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{percentage:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    left += data_array[:, i]

ax.set_xlabel("Scores", fontsize=12)
ax.set_ylabel("People", fontsize=12)
ax.set_title("Stacked Bar Chart with Multi-dimensional Data", fontsize=13)
ax.set_xlim(0, 40)
ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="x")

plt.tight_layout()
save_plot()

# Plot 13: Pie chart from tips data
df_tips = pd.read_csv(Path(__file__).parent / "data" / "tips.csv")

# Count occurrences by day
day_counts = df_tips["day"].value_counts()

# Define colors matching the image
colors_pie = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    day_counts,
    labels=day_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors_pie,
    textprops={"fontsize": 12, "weight": "bold"},
)

# Make percentage text white and bold
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(11)

plt.title("Distribution of Tips by Day", fontsize=14, pad=20)
plt.tight_layout()
save_plot()

# Plot 14: Bar chart with different hatch patterns (textures)
# Define different hatch patterns
hatches = ["|", "/", "\\", "+", "-", ".", "*", "x", "o", "O"]

# Create data - 10 bars all with same height
x_pos = np.arange(len(hatches))
heights = [3.0] * len(hatches)

plt.figure(figsize=(10, 7))

# Create bars with different hatch patterns
for i, (pos, height, hatch) in enumerate(zip(x_pos, heights, hatches)):
    plt.bar(
        pos,
        height,
        width=0.8,
        color="white",
        edgecolor="black",
        linewidth=1.5,
        hatch=hatch * 3,  # Multiply to make pattern denser
    )

plt.xlabel("Bar Index", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("Bar Chart with Different Hatch Patterns", fontsize=13)
plt.xticks(x_pos, x_pos)
plt.ylim(0, 3.0)
plt.xlim(-0.5, len(hatches) - 0.5)
plt.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="y")
plt.tight_layout()
save_plot()

# Plot 15: Pie chart - Programming language popularity
# Using the languages and popularity data from earlier plots
languages_pie = ["Java", "Python", "PHP", "JavaScript", "C#", "C++"]
popularity_pie = [22.2, 17.6, 8.8, 8, 7.7, 6.7]

# Define colors for each language
colors_lang = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#feca57", "#ee5a6f", "#a29bfe"]

plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(
    popularity_pie,
    labels=languages_pie,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors_lang,
    textprops={"fontsize": 11, "weight": "bold"},
    explode=(0.05, 0.05, 0, 0, 0, 0),  # Slightly separate Java and Python
)

# Make percentage text white
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(10)

plt.title(
    "Programming Language Popularity\nWorldwide, Oct 2017",
    fontsize=14,
    pad=20,
    fontweight="bold",
)
plt.tight_layout()
save_plot()

# Plot 16: Scatter plot with random X and Y values
# Generate random data points
np.random.seed(42)  # For reproducibility
n_points = 300

# Generate random X and Y values centered around 0
x_random = np.random.randn(n_points) * 0.8
y_random = np.random.randn(n_points) * 0.8

plt.figure(figsize=(8, 6))
plt.scatter(
    x_random,
    y_random,
    color="red",
    s=30,
    alpha=0.7,
    edgecolors="darkred",
    linewidth=0.5,
)

plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.title("Wykres punktowy dla dowolnych losowych wartości X i Y", fontsize=13)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
plt.tight_layout()
save_plot()

# Plot 17: Scatter plot with 3 dimensions (X, Y, and size)
# Generate random data for about 30 points
np.random.seed(123)  # Different seed for variety
n_points_3d = 30

# Random X and Y values between 0 and 1
x_3d = np.random.rand(n_points_3d)
y_3d = np.random.rand(n_points_3d)

# Random sizes (third dimension) - varied marker sizes
sizes_3d = np.random.rand(n_points_3d) * 1000 + 100  # Size between 100 and 1100

# Random colors - create 4 color categories
color_categories = np.random.choice(
    ["darkblue", "cyan", "darkred", "yellow"], n_points_3d
)

plt.figure(figsize=(8, 6))

# Plot each point with its specific color and size
for i in range(n_points_3d):
    plt.scatter(
        x_3d[i],
        y_3d[i],
        s=sizes_3d[i],
        color=color_categories[i],
        alpha=0.6,
        edgecolors="black",
        linewidth=1.5,
    )

plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.title(
    "Wykres punktowy dla dowolnych losowych wartości 3 wymiary, trzeci jako rozmiar punktów",
    fontsize=12,
)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
plt.tight_layout()
save_plot()

# Plot 18: Scatter plot comparing math and science marks
math_marks = [88, 92, 80, 89, 100, 80, 60, 100, 80, 34]
science_marks = [35, 79, 79, 48, 100, 88, 32, 45, 20, 30]
marks_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plt.figure(figsize=(10, 7))

# Plot math marks
plt.scatter(
    marks_range,
    math_marks,
    color="red",
    s=60,
    alpha=0.7,
    label="Math marks",
    edgecolors="darkred",
    linewidth=1,
)

# Plot science marks
plt.scatter(
    marks_range,
    science_marks,
    color="green",
    s=60,
    alpha=0.7,
    label="Science marks",
    edgecolors="darkgreen",
    linewidth=1,
)

plt.xlabel("Marks Range", fontsize=12)
plt.ylabel("Marks Scored", fontsize=12)
plt.title("Scatter Plot", fontsize=14)
plt.xlim(0, 120)
plt.ylim(10, 110)
plt.legend(loc="upper right", fontsize=11)
plt.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
plt.tight_layout()
save_plot()
