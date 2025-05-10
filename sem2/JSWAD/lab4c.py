# JSwAD - Laboratorium 4 - zadanie C - Jan Banot
import matplotlib.pyplot as plt
import numpy as np

data = {
    "1994": 459.27,
    "1995": 615.93,
    "1996": 740.74,
    "1997": 970.43,
    "1998": 1229.43,
    "1999": 1469.25,
    "2000": 1320.28,
}

# 1. Wykres przedstawiający dane indeksu S&P z lat 1994 - 2000
years = list(data.keys())
values = list(data.values())

plt.figure(figsize=(10, 6))
plt.bar(years, values, color="skyblue")

plt.xlabel("Rok")
plt.ylabel("Wartość indeksu S&P")
plt.title("Wartości indeksu S&P z lat 1994 - 2000")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Interpretacja:
# Możnemy zauważyć rosnący trend wartości indeksu S&P w latach 1994 - 1999

# 2. Badanie korelacji między wydatkami na reklamę a sprzedażą
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2]
y = [2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1]

correlation_matrix = np.corrcoef(x, y)
correlation_xy = correlation_matrix[0, 1]

print(
    f"\nWspółczynnik korelacji Pearsona między wydatkami na reklamę a sprzedażą: {correlation_xy:.4f}"
)

# Wykres rozrzutu
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="coral")
plt.xlabel("Wydatki na reklamę")
plt.ylabel("Sprzedaż")
plt.title("Zależność między wydatkami na reklamę a sprzedażą")
plt.grid(True)
plt.tight_layout()
plt.show()

# Interpretacja:
# Wysoka dodatnia wartość współczynnika korelacji (bliska 1) sugeruje silną dodatnią zależność liniową.
# Oznacza to, że wraz ze wzrostem wydatków na reklamę, rośnie również sprzedaż.
# Wykres rozrzutu pokazuje punkty układające się wzdłuż rosnącej linii prostej.

# Wizualizacja szeregów czasowych dla x i y
plt.figure(figsize=(12, 6))
plt.plot(x, label="Wydatki na reklamę (x)", marker="o", linestyle="-")
plt.plot(y, label="Sprzedaż (y)", marker="x", linestyle="--")
plt.xlabel("Indeks czasu")
plt.ylabel("Wartość")
plt.title("Szeregi czasowe dla wydatków na reklamę i sprzedaży")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
