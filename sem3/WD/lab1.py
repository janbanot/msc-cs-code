# pyright: basic
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)

# Load the Titanic dataset
df = pd.read_csv(
    "/Users/janbanot/Dev/uni/msc-cs-code/sem3/WD/plots/lab1/data/Titanic.txt"
)

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSurvival rate overall:", df["Survived"].mean())

# 1. Survival Rate by Gender
plt.figure(figsize=(10, 6))
survival_by_sex = df.groupby("Sex")["Survived"].agg(["sum", "count", "mean"])
survival_by_sex["percentage"] = survival_by_sex["mean"] * 100

colors = ["#e74c3c", "#3498db"]
bars = plt.bar(
    survival_by_sex.index,
    survival_by_sex["percentage"],
    color=colors,
    alpha=0.8,
    edgecolor="black",
)
plt.title("Survival Rate by Gender", fontsize=14, fontweight="bold")
plt.xlabel("Gender", fontsize=12)
plt.ylabel("Survival Rate (%)", fontsize=12)
plt.ylim([0, 100])

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    "/Users/janbanot/Dev/uni/msc-cs-code/sem3/WD/plots/lab1/survival_by_gender.png",
    dpi=300,
    bbox_inches="tight",
)
print("Chart 1 saved as 'survival_by_gender.png'")
plt.show()

# 2. Survival Rate by Passenger Class
plt.figure(figsize=(10, 6))
survival_by_class = df.groupby("Pclass")["Survived"].agg(["sum", "count", "mean"])
survival_by_class["percentage"] = survival_by_class["mean"] * 100

colors = ["#2ecc71", "#f39c12", "#e74c3c"]
bars = plt.bar(
    survival_by_class.index.astype(str),
    survival_by_class["percentage"],
    color=colors,
    alpha=0.8,
    edgecolor="black",
)
plt.title("Survival Rate by Passenger Class", fontsize=14, fontweight="bold")
plt.xlabel("Passenger Class", fontsize=12)
plt.ylabel("Survival Rate (%)", fontsize=12)
plt.ylim([0, 100])

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    "/Users/janbanot/Dev/uni/msc-cs-code/sem3/WD/plots/lab1/survival_by_class.png",
    dpi=300,
    bbox_inches="tight",
)
print("Chart 2 saved as 'survival_by_class.png'")
plt.show()

# 3. Survival Rate by Age Group
plt.figure(figsize=(12, 6))
# Create age groups
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[
        "Child (0-12)",
        "Teen (13-18)",
        "Adult (19-35)",
        "Middle-aged (36-60)",
        "Elderly (60+)",
    ],
)

survival_by_age = df.groupby("AgeGroup", observed=True)["Survived"].agg(
    ["sum", "count", "mean"]
)
survival_by_age["percentage"] = survival_by_age["mean"] * 100

colors = ["#3498db", "#9b59b6", "#2ecc71", "#f39c12", "#e74c3c"]
bars = plt.bar(
    range(len(survival_by_age)),
    survival_by_age["percentage"],
    color=colors,
    alpha=0.8,
    edgecolor="black",
)
plt.title("Survival Rate by Age Group", fontsize=14, fontweight="bold")
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Survival Rate (%)", fontsize=12)
plt.xticks(
    range(len(survival_by_age)), survival_by_age.index.tolist(), rotation=45, ha="right"
)
plt.ylim([0, 100])

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    "/Users/janbanot/Dev/uni/msc-cs-code/sem3/WD/plots/lab1/survival_by_age.png",
    dpi=300,
    bbox_inches="tight",
)
print("Chart 3 saved as 'survival_by_age.png'")
plt.show()

# 4. Combined Analysis: Gender & Class
plt.figure(figsize=(12, 6))
survival_gender_class = df.groupby(["Pclass", "Sex"])["Survived"].mean() * 100
survival_gender_class = survival_gender_class.unstack()

x = np.arange(len(survival_gender_class.index))
width = 0.35

bars1 = plt.bar(
    x - width / 2,
    survival_gender_class["female"],
    width,
    label="Female",
    color="#e74c3c",
    alpha=0.8,
    edgecolor="black",
)
bars2 = plt.bar(
    x + width / 2,
    survival_gender_class["male"],
    width,
    label="Male",
    color="#3498db",
    alpha=0.8,
    edgecolor="black",
)

plt.title("Survival Rate by Class and Gender", fontsize=14, fontweight="bold")
plt.xlabel("Passenger Class", fontsize=12)
plt.ylabel("Survival Rate (%)", fontsize=12)
plt.xticks(x, ["1st Class", "2nd Class", "3rd Class"])
plt.legend()
plt.ylim([0, 100])

# Add percentage labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig(
    "/Users/janbanot/Dev/uni/msc-cs-code/sem3/WD/plots/lab1/survival_by_class_gender.png",
    dpi=300,
    bbox_inches="tight",
)
print("Chart 4 saved as 'survival_by_class_gender.png'")
plt.show()

# 5. Age Distribution: Survivors vs Non-survivors
plt.figure(figsize=(12, 6))
survived_ages = df[df["Survived"] == 1]["Age"].dropna()
died_ages = df[df["Survived"] == 0]["Age"].dropna()

plt.hist(
    [died_ages, survived_ages],
    bins=20,
    label=["Did not survive", "Survived"],
    color=["#e74c3c", "#2ecc71"],
    alpha=0.7,
    edgecolor="black",
)
plt.title(
    "Age Distribution: Survivors vs Non-survivors", fontsize=14, fontweight="bold"
)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)
plt.legend()

plt.tight_layout()
plt.savefig(
    "/Users/janbanot/Dev/uni/msc-cs-code/sem3/WD/plots/lab1/age_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
print("Chart 5 saved as 'age_distribution.png'")
plt.show()

# 6. Fare vs Survival
plt.figure(figsize=(10, 6))
survived_fare = df[df["Survived"] == 1]["Fare"].dropna()
died_fare = df[df["Survived"] == 0]["Fare"].dropna()

box_data = [died_fare, survived_fare]
bp = plt.boxplot(
    box_data, tick_labels=["Did not survive", "Survived"], patch_artist=True, widths=0.6
)

# Color the boxes
colors = ["#e74c3c", "#2ecc71"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title("Fare Distribution by Survival", fontsize=14, fontweight="bold")
plt.ylabel("Fare (£)", fontsize=12)
plt.xlabel("Survival Status", fontsize=12)

plt.tight_layout()
plt.savefig(
    "/Users/janbanot/Dev/uni/msc-cs-code/sem3/WD/plots/lab1/fare_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
print("Chart 6 saved as 'fare_distribution.png'")
plt.show()
