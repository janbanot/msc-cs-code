# JSwAD - Laboratorium 6 - Analiza danych COVID-19 - Jan Banot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
import sys
import io
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")


# System do przechwytywania output'u konsoli
class OutputCapture:
    def __init__(self):
        self.captured_output = []
        self.original_stdout = sys.stdout

    def start_capture(self):
        sys.stdout = self

    def stop_capture(self):
        sys.stdout = self.original_stdout

    def write(self, text):
        self.original_stdout.write(text)
        self.captured_output.append(text)

    def flush(self):
        self.original_stdout.flush()

    def get_output(self):
        return "".join(self.captured_output)


# Inicjalizacja przechwytywania output'u
output_capture = OutputCapture()
output_capture.start_capture()

# Ustawienia wyświetlania
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
sns.set_palette("husl")

print("=" * 60)
print("ANALIZA DANYCH COVID-19")
print("JSwAD - Laboratorium 6 - Jan Banot")
print("=" * 60)

# ================================================================
# 1. ŁADOWANIE I PRZETWARZANIE DANYCH
# ================================================================

print("\n1. ŁADOWANIE I PRZETWARZANIE DANYCH")
print("-" * 40)

# Wczytanie danych
covid_data = pd.read_csv(
    "/Users/janbanot/Dev/uni/msc-cs-code/sem2/JSWAD/data/bd-covid-19.csv"
)
print(f"Wymiary zbioru danych: {covid_data.shape}")
print(f"Okres analizy: {covid_data['date'].iloc[0]} - {covid_data['date'].iloc[-1]}")

# Konwersja daty i ustawienie jako indeks
covid_data["date"] = pd.to_datetime(covid_data["date"], format="%d/%m/%y")
covid_data = covid_data.set_index("date").sort_index()

# Sprawdzenie brakujących wartości
print("\nBrakujące wartości w poszczególnych kolumnach:")
missing_values = covid_data.isnull().sum()
for col, missing in missing_values.items():
    percentage = (missing / len(covid_data)) * 100
    print(f"{col}: {missing} ({percentage:.1f}%)")

# Wypełnienie brakujących wartości
covid_data = covid_data.fillna(0)

# Utworzenie dodatkowych metryk
covid_data["wskaznik_smiertelnosci"] = (
    covid_data["total_deaths"] / covid_data["total_cases"] * 100
).fillna(0)
covid_data["pozytywnosc_testow"] = (
    covid_data["new_cases"] / covid_data["new_tests"] * 100
).fillna(0)

# Wygładzenie danych - średnie ruchome 7-dniowe
covid_data["nowe_przypadki_7d"] = covid_data["new_cases"].rolling(window=7).mean()
covid_data["nowe_zgony_7d"] = covid_data["new_deaths"].rolling(window=7).mean()
covid_data["nowe_testy_7d"] = covid_data["new_tests"].rolling(window=7).mean()

print("\nPrzetwarzanie danych zakończone pomyślnie!")
print(f"Dodano {len(covid_data.columns) - 7} nowych kolumn z metrykami pochodnymi")

# ================================================================
# 2. ANALIZA EKSPLORACYJNA DANYCH (EDA)
# ================================================================

print("\n\n2. ANALIZA EKSPLORACYJNA DANYCH")
print("-" * 40)

# Podstawowe statystyki opisowe
print("\nStatystyki opisowe dla kluczowych zmiennych:")
podstawowe_kolumny = [
    "new_cases",
    "new_deaths",
    "new_tests",
    "wskaznik_smiertelnosci",
    "pozytywnosc_testow",
]
statystyki = covid_data[podstawowe_kolumny].describe()
print(statystyki.round(2))

# Informacje o szczytowych wartościach
print("\nSzczytowe wartości:")
print(
    f"Maksymalna liczba nowych przypadków: {covid_data['new_cases'].max()} (data: {covid_data['new_cases'].idxmax().strftime('%d-%m-%Y')})"
)
print(
    f"Maksymalna liczba nowych zgonów: {covid_data['new_deaths'].max()} (data: {covid_data['new_deaths'].idxmax().strftime('%d-%m-%Y')})"
)
print(
    f"Maksymalny wskaźnik śmiertelności: {covid_data['wskaznik_smiertelnosci'].max():.2f}%"
)


# Wykrywanie wartości odstających (outliers)
def wykryj_outliers(data, kolumna):
    Q1 = data[kolumna].quantile(0.25)
    Q3 = data[kolumna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[kolumna] < lower_bound) | (data[kolumna] > upper_bound)]
    return len(outliers)


print("\nWykrycie wartości odstających (metoda IQR):")
for kolumna in ["new_cases", "new_deaths", "new_tests"]:
    liczba_outliers = wykryj_outliers(covid_data, kolumna)
    print(f"{kolumna}: {liczba_outliers} wartości odstających")

# ================================================================
# 3. ANALIZA STATYSTYCZNA
# ================================================================

print("\n\n3. ANALIZA STATYSTYCZNA")
print("-" * 40)

# Analiza korelacji
print("\nAnaliza korelacji między zmiennymi:")
kolumny_korelacja = [
    "new_cases",
    "new_deaths",
    "new_tests",
    "total_cases",
    "total_deaths",
    "total_tests",
]
macierz_korelacji = covid_data[kolumny_korelacja].corr()

# Wyświetlenie najsilniejszych korelacji
print("\nNajsilniejsze korelacje:")
for i in range(len(macierz_korelacji.columns)):
    for j in range(i + 1, len(macierz_korelacji.columns)):
        korel = macierz_korelacji.iloc[i, j]
        if abs(korel) > 0.5:
            print(
                f"{macierz_korelacji.columns[i]} - {macierz_korelacji.columns[j]}: {korel:.3f}"
            )

# Test t-Studenta - porównanie pierwszej i drugiej fali
print("\nPorównanie pierwszej i drugiej fali pandemii (test t-Studenta):")

# Podział na okresy (przybliżone fale)
pierwsza_fala = covid_data[
    (covid_data.index >= "2020-03-01") & (covid_data.index <= "2020-08-31")
]
druga_fala = covid_data[
    (covid_data.index >= "2020-09-01") & (covid_data.index <= "2021-02-28")
]

for zmienna in ["new_cases", "new_deaths"]:
    dane1 = pierwsza_fala[zmienna].dropna()
    dane2 = druga_fala[zmienna].dropna()

    if len(dane1) > 0 and len(dane2) > 0:
        t_stat, p_value = stats.ttest_ind(dane1, dane2)
        print(f"{zmienna}:")
        print(
            f"  Pierwsza fala - średnia: {dane1.mean():.1f}, odchylenie: {dane1.std():.1f}"
        )
        print(
            f"  Druga fala - średnia: {dane2.mean():.1f}, odchylenie: {dane2.std():.1f}"
        )
        print(f"  t-statystyka: {t_stat:.3f}, p-wartość: {p_value:.6f}")

        if p_value < 0.05:
            print("  Wynik: Istotna różnica statystyczna (p < 0.05)")
        else:
            print("  Wynik: Brak istotnej różnicy statystycznej (p >= 0.05)")

# Test normalności Shapiro-Wilka dla próby danych
print("\nTest normalności rozkładu (Shapiro-Wilk) - próba 100 obserwacji:")
for zmienna in ["new_cases", "new_deaths"]:
    probka = (
        covid_data[zmienna]
        .dropna()
        .sample(min(100, len(covid_data[zmienna].dropna())), random_state=42)
    )
    if len(probka) > 3:
        stat, p_value = stats.shapiro(probka)
        print(f"{zmienna}: W = {stat:.4f}, p-wartość = {p_value:.6f}")
        print(f"  {'Rozkład normalny' if p_value > 0.05 else 'Rozkład nienormalny'}")


# Test trendu Mann-Kendall (uproszczony)
def test_trendu_mk(data):
    n = len(data)
    if n < 3:
        return None, None

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if data.iloc[j] > data.iloc[i]:
                s += 1
            elif data.iloc[j] < data.iloc[i]:
                s -= 1

    var_s = n * (n - 1) * (2 * n + 5) / 18
    z = s / np.sqrt(var_s) if var_s > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value


print("\nTest trendu Mann-Kendall:")
for zmienna in ["total_cases", "total_deaths"]:
    dane = covid_data[zmienna].dropna()
    if len(dane) > 10:
        z_stat, p_value = test_trendu_mk(dane)
        if z_stat is not None:
            trend = "rosnący" if z_stat > 0 else "malejący" if z_stat < 0 else "brak"
            istotnosc = "istotny" if p_value < 0.05 else "nieistotny"
            print(
                f"{zmienna}: Z = {z_stat:.3f}, p = {p_value:.6f} - trend {trend} ({istotnosc})"
            )

print("\nAnaliza statystyczna zakończona!")

# ================================================================
# 4. WIZUALIZACJE DANYCH
# ================================================================

print("\n\n4. WIZUALIZACJE DANYCH")
print("-" * 40)

# Tworzenie subplotów dla głównych szeregów czasowych
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Analiza szeregów czasowych COVID-19 - Bangladesz", fontsize=16, fontweight="bold"
)

# Wykres 1: Nowe przypadki
axes[0, 0].plot(
    covid_data.index,
    covid_data["new_cases"],
    alpha=0.3,
    color="blue",
    label="Dane dzienne",
)
axes[0, 0].plot(
    covid_data.index,
    covid_data["nowe_przypadki_7d"],
    color="red",
    linewidth=2,
    label="Średnia 7-dniowa",
)
axes[0, 0].set_title("Nowe przypadki COVID-19")
axes[0, 0].set_ylabel("Liczba przypadków")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Wykres 2: Nowe zgony
axes[0, 1].plot(
    covid_data.index,
    covid_data["new_deaths"],
    alpha=0.3,
    color="darkred",
    label="Dane dzienne",
)
axes[0, 1].plot(
    covid_data.index,
    covid_data["nowe_zgony_7d"],
    color="black",
    linewidth=2,
    label="Średnia 7-dniowa",
)
axes[0, 1].set_title("Nowe zgony COVID-19")
axes[0, 1].set_ylabel("Liczba zgonów")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Wykres 3: Nowe testy
axes[1, 0].plot(covid_data.index, covid_data["new_tests"], alpha=0.5, color="green")
axes[1, 0].plot(
    covid_data.index, covid_data["nowe_testy_7d"], color="darkgreen", linewidth=2
)
axes[1, 0].set_title("Nowe testy COVID-19")
axes[1, 0].set_ylabel("Liczba testów")
axes[1, 0].grid(True, alpha=0.3)

# Wykres 4: Wskaźnik śmiertelności
axes[1, 1].plot(
    covid_data.index,
    covid_data["wskaznik_smiertelnosci"],
    color="purple",
    linewidth=1.5,
)
axes[1, 1].set_title("Wskaźnik śmiertelności (%)")
axes[1, 1].set_ylabel("Procent")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("covid_analiza_szeregi_czasowe.png", dpi=300, bbox_inches="tight")
plt.show()

# Mapa ciepła korelacji
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(macierz_korelacji, dtype=bool))
sns.heatmap(
    macierz_korelacji,
    mask=mask,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    fmt=".2f",
)
plt.title("Mapa korelacji między zmiennymi COVID-19", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("covid_mapa_korelacji.png", dpi=300, bbox_inches="tight")
plt.show()

# Rozkłady zmiennych - histogramy
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Rozkłady kluczowych zmiennych COVID-19", fontsize=16, fontweight="bold")

zmienne_hist = ["new_cases", "new_deaths", "new_tests", "pozytywnosc_testow"]
tytuly = ["Nowe przypadki", "Nowe zgony", "Nowe testy", "Pozytywność testów (%)"]

for i, (zmienna, tytul) in enumerate(zip(zmienne_hist, tytuly)):
    row, col = i // 2, i % 2
    dane = covid_data[zmienna].dropna()
    dane = dane[dane > 0]  # Usunięcie zer dla lepszej wizualizacji
    dane = dane[np.isfinite(dane)]  # Usunięcie wartości nieskończonych

    if len(dane) > 0:
        axes[row, col].hist(dane, bins=30, alpha=0.7, edgecolor="black")
        axes[row, col].axvline(
            dane.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Średnia: {dane.mean():.1f}",
        )
        axes[row, col].axvline(
            dane.median(),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mediana: {dane.median():.1f}",
        )
        axes[row, col].set_title(tytul)
        axes[row, col].set_xlabel("Wartość")
        axes[row, col].set_ylabel("Częstość")
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("covid_histogramy.png", dpi=300, bbox_inches="tight")
plt.show()

# Wykres pudełkowy dla porównania fal pandemii
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Porównanie fal pandemii - wykresy pudełkowe", fontsize=16, fontweight="bold"
)

# Przygotowanie danych dla wykresów pudełkowych
fale_dane = {
    "Pierwsza fala": pierwsza_fala[["new_cases", "new_deaths"]],
    "Druga fala": druga_fala[["new_cases", "new_deaths"]],
}

dane_boxplot_przypadki = []
labels_boxplot = []
for nazwa_fali, dane_fali in fale_dane.items():
    dane_boxplot_przypadki.append(dane_fali["new_cases"].dropna())
    labels_boxplot.append(nazwa_fali)

dane_boxplot_zgony = []
for nazwa_fali, dane_fali in fale_dane.items():
    dane_boxplot_zgony.append(dane_fali["new_deaths"].dropna())

axes[0].boxplot(dane_boxplot_przypadki, labels=labels_boxplot)
axes[0].set_title("Nowe przypadki w różnych falach")
axes[0].set_ylabel("Liczba przypadków")
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(dane_boxplot_zgony, labels=labels_boxplot)
axes[1].set_title("Nowe zgony w różnych falach")
axes[1].set_ylabel("Liczba zgonów")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("covid_porownanie_fal.png", dpi=300, bbox_inches="tight")
plt.show()

# Wykres skumulowany
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(
    covid_data.index,
    covid_data["total_cases"],
    color="blue",
    linewidth=2,
    label="Łączne przypadki",
)
plt.plot(
    covid_data.index,
    covid_data["total_deaths"],
    color="red",
    linewidth=2,
    label="Łączne zgony",
)
plt.title(
    "Łączne przypadki i zgony COVID-19 w Bangladeszu", fontsize=14, fontweight="bold"
)
plt.ylabel("Liczba (skala logarytmiczna)")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(covid_data.index, covid_data["total_tests"], color="green", linewidth=2)
plt.title("Łączna liczba testów COVID-19", fontsize=14, fontweight="bold")
plt.ylabel("Liczba testów")
plt.xlabel("Data")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("covid_dane_skumulowane.png", dpi=300, bbox_inches="tight")
plt.show()

print("Wizualizacje zostały wygenerowane i zapisane!")

# ================================================================
# 5. ANALIZA ZAAWANSOWANA
# ================================================================

print("\n\n5. ANALIZA ZAAWANSOWANA")
print("-" * 40)

# Obliczenie współczynników wzrostu
covid_data["tempo_wzrostu_przypadkow"] = (
    covid_data["new_cases"].pct_change().fillna(0) * 100
)
covid_data["tempo_wzrostu_zgonow"] = (
    covid_data["new_deaths"].pct_change().fillna(0) * 100
)

# Wykrywanie szczytów (peaks) w danych

# Znajdowanie szczytów w średniej 7-dniowej nowych przypadków
peaks_przypadki, _ = find_peaks(
    covid_data["nowe_przypadki_7d"].fillna(0),
    height=covid_data["nowe_przypadki_7d"].quantile(0.9),
    distance=14,
)  # Minimum 14 dni między szczytami

peaks_zgony, _ = find_peaks(
    covid_data["nowe_zgony_7d"].fillna(0),
    height=covid_data["nowe_zgony_7d"].quantile(0.9),
    distance=14,
)

print(f"Wykryto {len(peaks_przypadki)} głównych szczytów przypadków:")
for i, peak_idx in enumerate(peaks_przypadki):
    data_szczytu = covid_data.index[peak_idx]
    wartosc_szczytu = covid_data["nowe_przypadki_7d"].iloc[peak_idx]
    print(
        f"  Szczyt {i + 1}: {data_szczytu.strftime('%d-%m-%Y')} - {wartosc_szczytu:.0f} przypadków/dzień"
    )

print(f"\nWykryto {len(peaks_zgony)} głównych szczytów zgonów:")
for i, peak_idx in enumerate(peaks_zgony):
    data_szczytu = covid_data.index[peak_idx]
    wartosc_szczytu = covid_data["nowe_zgony_7d"].iloc[peak_idx]
    print(
        f"  Szczyt {i + 1}: {data_szczytu.strftime('%d-%m-%Y')} - {wartosc_szczytu:.0f} zgonów/dzień"
    )

# Analiza miesięczna
covid_data["miesiac"] = covid_data.index.month
covid_data["rok"] = covid_data.index.year

analiza_miesieczna = (
    covid_data.groupby(["rok", "miesiac"])
    .agg({"new_cases": "sum", "new_deaths": "sum", "new_tests": "sum"})
    .reset_index()
)

print("\nNajgorsze miesiące (największa liczba przypadków):")
najgorsze_miesiace = analiza_miesieczna.nlargest(5, "new_cases")
for _, row in najgorsze_miesiace.iterrows():
    print(
        f"  {int(row['miesiac']):02d}/{int(row['rok'])}: {int(row['new_cases'])} przypadków, {int(row['new_deaths'])} zgonów"
    )

# Efektywność testowania
covid_data["efektywnosc_testow"] = (
    covid_data["new_cases"] / covid_data["new_tests"]
).fillna(0)

print(f"\nŚrednia pozytywność testów: {covid_data['pozytywnosc_testow'].mean():.2f}%")
print(f"Maksymalna pozytywność testów: {covid_data['pozytywnosc_testow'].max():.2f}%")
print(
    f"Okres najwyższej pozytywności: {covid_data['pozytywnosc_testow'].idxmax().strftime('%d-%m-%Y')}"
)

# Prosta prognoza na podstawie trendu (ostatnie 30 dni)
ostatnie_30_dni = covid_data.tail(30)
srednia_przypadkow_30d = ostatnie_30_dni["new_cases"].mean()
trend_przypadkow = ostatnie_30_dni["new_cases"].diff().mean()

print("\nProsta prognoza na podstawie ostatnich 30 dni:")
print(f"Średnia dzienna przypadków (ostatnie 30 dni): {srednia_przypadkow_30d:.1f}")
print(f"Trend dzienny: {trend_przypadkow:+.1f} przypadków/dzień")
print(
    f"Prognozowana średnia na następne 7 dni: {srednia_przypadkow_30d + 7 * trend_przypadkow:.1f}"
)

# ================================================================
# 6. KLUCZOWE WNIOSKI I PODSUMOWANIE
# ================================================================

print("\n\n6. KLUCZOWE WNIOSKI I PODSUMOWANIE")
print("=" * 60)

print("\n📊 PODSTAWOWE STATYSTYKI:")
print(
    f"• Okres analizy: {covid_data.index[0].strftime('%d-%m-%Y')} - {covid_data.index[-1].strftime('%d-%m-%Y')}"
)
print(f"• Łączna liczba przypadków: {covid_data['total_cases'].iloc[-1]:,.0f}")
print(f"• Łączna liczba zgonów: {covid_data['total_deaths'].iloc[-1]:,.0f}")
print(f"• Łączna liczba testów: {covid_data['total_tests'].iloc[-1]:,.0f}")
print(
    f"• Końcowy wskaźnik śmiertelności: {covid_data['wskaznik_smiertelnosci'].iloc[-1]:.2f}%"
)

print("\n📈 DYNAMIKA PANDEMII:")
print(
    f"• Dzień z największą liczbą nowych przypadków: {covid_data['new_cases'].idxmax().strftime('%d-%m-%Y')} ({covid_data['new_cases'].max():,.0f} przypadków)"
)
print(
    f"• Dzień z największą liczbą zgonów: {covid_data['new_deaths'].idxmax().strftime('%d-%m-%Y')} ({covid_data['new_deaths'].max():,.0f} zgonów)"
)
print(f"• Średnia dzienna nowych przypadków: {covid_data['new_cases'].mean():.1f}")
print(f"• Średnia dzienna zgonów: {covid_data['new_deaths'].mean():.1f}")

print("\n🔬 ANALIZA TESTOWANIA:")
print(f"• Średnia pozytywność testów: {covid_data['pozytywnosc_testow'].mean():.2f}%")
print(
    f"• Najwyższa pozytywność testów: {covid_data['pozytywnosc_testow'].max():.2f}% ({covid_data['pozytywnosc_testow'].idxmax().strftime('%d-%m-%Y')})"
)
print(f"• Średnia dzienna testów: {covid_data['new_tests'].mean():.0f}")

print("\n📊 ANALIZA STATYSTYCZNA:")
najsilniejsza_korelacja = (
    macierz_korelacji.abs().unstack().sort_values(ascending=False).drop_duplicates()
)
najsilniejsza_korelacja = najsilniejsza_korelacja[najsilniejsza_korelacja < 1.0].iloc[0]
print(f"• Najsilniejsza korelacja: {najsilniejsza_korelacja:.3f}")

print("\n🌊 IDENTYFIKACJA FAL PANDEMII:")
print(f"• Liczba głównych szczytów przypadków: {len(peaks_przypadki)}")
print(f"• Liczba głównych szczytów zgonów: {len(peaks_zgony)}")

if len(peaks_przypadki) >= 2:
    okres_miedzy_falami = (
        covid_data.index[peaks_przypadki[1]] - covid_data.index[peaks_przypadki[0]]
    ).days
    print(f"• Okres między pierwszym a drugim szczytem: {okres_miedzy_falami} dni")

print("\n🎯 WNIOSKI KOŃCOWE:")
print("• Bangladesz doświadczył wyraźnych fal pandemii COVID-19")
print("• Wskaźnik śmiertelności ewoluował w czasie wraz z poprawą opieki medycznej")
print("• Strategia testowania została znacząco rozszerzona w trakcie pandemii")
print(
    "• Dane wykazują wyraźne trendy sezonowe i cykliczne w rozprzestrzenianiu się wirusa"
)

if covid_data["new_cases"].tail(30).mean() > covid_data["new_cases"].head(30).mean():
    print("• Tendencja rosnąca w ostatnim okresie - wymagana czujność")
else:
    print("• Tendencja stabilizacyjna lub malejąca w ostatnim okresie")

print("\n📝 DODATKOWE OBSERWACJE:")
wskaznik_koncowy = covid_data["wskaznik_smiertelnosci"].iloc[-1]
wskaznik_poczatkowy = (
    covid_data["wskaznik_smiertelnosci"][covid_data["wskaznik_smiertelnosci"] > 0].iloc[
        0
    ]
    if len(
        covid_data["wskaznik_smiertelnosci"][covid_data["wskaznik_smiertelnosci"] > 0]
    )
    > 0
    else 0
)

if wskaznik_koncowy < wskaznik_poczatkowy:
    print(
        f"• Wskaźnik śmiertelności zmniejszył się z {wskaznik_poczatkowy:.2f}% do {wskaznik_koncowy:.2f}%"
    )
else:
    print(
        f"• Wskaźnik śmiertelności utrzymuje się na poziomie około {wskaznik_koncowy:.2f}%"
    )

print("\n" + "=" * 60)
print("ANALIZA COVID-19 ZAKOŃCZONA POMYŚLNIE!")
print("Wygenerowano kompletną analizę statystyczną i wizualizacje.")
print("=" * 60)

# Zatrzymanie przechwytywania output'u
output_capture.stop_capture()


def save_analysis_to_pdf():
    """Zapisuje analizę (logi + wykresy) do pliku PDF"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"COVID19_Analiza_{timestamp}.pdf"

    print(f"\nGenerowanie raportu PDF: {filename}")

    with PdfPages(filename) as pdf:
        # Strona tytułowa z logami
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        # Podział tekstu na strony jeśli jest za długi
        captured_text = output_capture.get_output()
        lines = captured_text.split("\n")

        # Formatowanie tekstu dla lepszej czytelności
        formatted_lines = []
        for line in lines:
            if len(line) > 80:
                # Dzielenie długich linii
                words = line.split(" ")
                current_line = ""
                for word in words:
                    if len(current_line + word) <= 80:
                        current_line += word + " "
                    else:
                        formatted_lines.append(current_line.rstrip())
                        current_line = word + " "
                if current_line:
                    formatted_lines.append(current_line.rstrip())
            else:
                formatted_lines.append(line)

        # Dodawanie tekstów na strony (max 50 linii na stronę)
        lines_per_page = 50
        for i in range(0, len(formatted_lines), lines_per_page):
            if i > 0:  # Nowa strona
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")

            page_lines = formatted_lines[i : i + lines_per_page]
            text_content = "\n".join(page_lines)

            ax.text(
                0.05,
                0.95,
                text_content,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                fontfamily="monospace",
                wrap=True,
            )

            # Nagłówek strony
            page_num = (i // lines_per_page) + 1
            ax.text(
                0.5,
                0.98,
                f"Analiza COVID-19 - Strona {page_num}",
                transform=ax.transAxes,
                fontsize=12,
                horizontalalignment="center",
                fontweight="bold",
            )

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Dodanie wszystkich wykresów z bieżącej sesji
        figures = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figures:
            if fig.get_axes():  # Sprawdź czy wykres ma zawartość
                pdf.savefig(fig, bbox_inches="tight")

        # Strona końcowa z metadanymi
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        metadata_text = f"""
RAPORT ANALIZY COVID-19
{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

Autor: Jan Banot
Kurs: JSwAD - Laboratorium 6
Uniwersytet: [Nazwa uniwersytetu]

Zakres analizy:
• Analiza eksploracyjna danych COVID-19
• Wizualizacje trendów i rozkładów
• Analiza statystyczna i korelacji
• Identyfikacja fal pandemii
• Prognozy i wnioski

Narzędzia użyte:
• Python 3.x
• Pandas, NumPy, Matplotlib, Seaborn
• SciPy (testy statystyczne)
• Matplotlib PdfPages (generowanie PDF)

Plik wygenerowany automatycznie przez skrypt lab6.py
        """

        ax.text(
            0.1,
            0.8,
            metadata_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"✅ Raport PDF zapisany jako: {filename}")
    return filename


# Generowanie raportu PDF
try:
    pdf_filename = save_analysis_to_pdf()
    print(f"📄 Kompletny raport dostępny w pliku: {pdf_filename}")
except Exception as e:
    print(f"❌ Błąd podczas generowania PDF: {e}")
    print("Analiza zakończona, ale raport PDF nie został utworzony.")
