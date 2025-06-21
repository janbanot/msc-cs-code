# JSWAD - Laboratorium 8: Predykcja i klasyfikacja - Jan Banot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    classification_report,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

# Ustawienie stylu wykresów
plt.style.use("default")
sns.set_palette("husl")

print("=== ZADANIE 1: PREDYKCJA CENY MIESZKAŃ ===\n")

# ===========================
# 1. ŁADOWANIE I GENEROWANIE DANYCH MIESZKANIOWYCH
# ===========================


# Generowanie przykładowych danych mieszkaniowych (jeśli nie ma pliku)
def generate_housing_data(n_samples=1000):
    np.random.seed(42)

    # Generowanie cech
    area = np.random.normal(100, 30, n_samples)  # m2
    bedrooms = np.random.poisson(3, n_samples)  # liczba sypialni
    age = np.random.uniform(0, 50, n_samples)  # wiek w latach

    # Generowanie ceny na podstawie cech z pewnym szumem
    price = (
        area * 2000
        + bedrooms * 10000
        - age * 500
        + np.random.normal(0, 20000, n_samples)
    )
    price = np.maximum(price, 50000)  # minimalna cena

    df = pd.DataFrame({"Area": area, "Bedrooms": bedrooms, "Age": age, "Price": price})

    return df


# Próba wczytania danych lub generowanie przykładowych
housing_data_path = (
    "/Users/janbanot/Dev/uni/msc-cs-code/sem2/JSWAD/data/housing_data.csv"
)
try:
    df = pd.read_csv(housing_data_path)
    print("Wczytano dane z pliku housing_data.csv")
except FileNotFoundError:
    print(
        "Plik housing_data.csv nie został znaleziony. Generowanie przykładowych danych..."
    )
    df = generate_housing_data()
    df.to_csv(housing_data_path, index=False)
    print("Wygenerowano i zapisano przykładowe dane do housing_data.csv")

print(f"\nKształt danych: {df.shape}")
print("\nPierwsze 5 wierszy:")
print(df.head())

# ===========================
# 2. SPRAWDZENIE DANYCH
# ===========================

print("\n=== INFORMACJE O DANYCH ===")
print(df.info())
print("\n=== STATYSTYKI OPISOWE ===")
print(df.describe())

print("\nBrakujące wartości:")
print(df.isnull().sum())

# Wizualizacja danych
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Analiza eksploracyjna danych mieszkaniowych", fontsize=16)

# Histogramy
df["Area"].hist(bins=30, ax=axes[0, 0], alpha=0.7)
axes[0, 0].set_title("Rozkład powierzchni")
axes[0, 0].set_xlabel("Powierzchnia (m²)")

df["Price"].hist(bins=30, ax=axes[0, 1], alpha=0.7)
axes[0, 1].set_title("Rozkład cen")
axes[0, 1].set_xlabel("Cena")

# Scatter plots
axes[1, 0].scatter(df["Area"], df["Price"], alpha=0.6)
axes[1, 0].set_title("Powierzchnia vs Cena")
axes[1, 0].set_xlabel("Powierzchnia (m²)")
axes[1, 0].set_ylabel("Cena")

axes[1, 1].scatter(df["Age"], df["Price"], alpha=0.6)
axes[1, 1].set_title("Wiek vs Cena")
axes[1, 1].set_xlabel("Wiek (lata)")
axes[1, 1].set_ylabel("Cena")

plt.tight_layout()
plt.show()

# ===========================
# 3. PODZIAŁ DANYCH
# ===========================

# Definiowanie X i y
X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

# Podział na zbiory: treningowy (60%), walidacyjny (20%), testowy (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("\nRozmiary zbiorów:")
print(f"Treningowy: {X_train.shape[0]} próbek")
print(f"Walidacyjny: {X_val.shape[0]} próbek")
print(f"Testowy: {X_test.shape[0]} próbek")

# Normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ===========================
# 4. EKSPERYMENTY Z RÓŻNYMI PARAMETRAMI
# ===========================


def create_model(input_shape, activation="relu", hidden_units=64):
    """Tworzy model regresyjny"""
    model = keras.Sequential(
        [
            layers.Dense(
                hidden_units, activation=activation, input_shape=(input_shape,)
            ),
            layers.Dense(hidden_units, activation=activation),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def train_and_evaluate_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs,
    batch_size,
    activation="relu",
    model_name="Model",
):
    """Trenuje i ocenia model"""
    model = create_model(X_train.shape[1], activation)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Trenowanie
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )

    # Predykcje
    y_pred = model.predict(X_test, verbose=0)

    # Metryki
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "history": history,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "y_pred": y_pred,
        "name": model_name,
    }


print("\n=== EKSPERYMENTY Z RÓŻNYMI PARAMETRAMI ===")

# Eksperymenty z różnymi parametrami
experiments = []

# Różne liczby epok
epochs_list = [10, 20, 50]
batch_sizes = [16, 32, 64]
activations = ["relu", "tanh", "sigmoid"]

print("\n1. Eksperyment z różnymi liczbami epok:")
for epochs in epochs_list:
    result = train_and_evaluate_model(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
        epochs=epochs,
        batch_size=32,
        model_name=f"Epochs_{epochs}",
    )
    experiments.append(result)
    print(
        f"Epoki: {epochs:2d} | MSE: {result['mse']:,.0f} | MAE: {result['mae']:,.0f} | R²: {result['r2']:.3f}"
    )

print("\n2. Eksperyment z różnymi rozmiarami partii:")
for batch_size in batch_sizes:
    result = train_and_evaluate_model(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
        epochs=30,
        batch_size=batch_size,
        model_name=f"Batch_{batch_size}",
    )
    experiments.append(result)
    print(
        f"Batch size: {batch_size:2d} | MSE: {result['mse']:,.0f} | MAE: {result['mae']:,.0f} | R²: {result['r2']:.3f}"
    )

print("\n3. Eksperyment z różnymi funkcjami aktywacji:")
for activation in activations:
    result = train_and_evaluate_model(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
        epochs=30,
        batch_size=32,
        activation=activation,
        model_name=f"Activation_{activation}",
    )
    experiments.append(result)
    print(
        f"Aktywacja: {activation:7s} | MSE: {result['mse']:,.0f} | MAE: {result['mae']:,.0f} | R²: {result['r2']:.3f}"
    )

# ===========================
# 5. WIZUALIZACJA WYNIKÓW
# ===========================

print("\n=== WIZUALIZACJA WYNIKÓW ===")

# Porównanie metryk
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["mse", "mae", "r2"]
metric_names = ["Mean Squared Error", "Mean Absolute Error", "R² Score"]

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    values = [exp[metric] for exp in experiments]
    names = [exp["name"] for exp in experiments]

    axes[i].bar(range(len(values)), values)
    axes[i].set_title(name)
    axes[i].set_xticks(range(len(names)))
    axes[i].set_xticklabels(names, rotation=45, ha="right")

    if metric == "r2":
        axes[i].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Wykres najlepszego modelu
best_model = max(experiments, key=lambda x: x["r2"])
print(f"\nNajlepszy model: {best_model['name']} (R² = {best_model['r2']:.3f})")

# Wykres predykcji vs rzeczywiste wartości
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_model["y_pred"], alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Rzeczywiste ceny")
plt.ylabel("Przewidywane ceny")
plt.title(f"Predykcje vs Rzeczywiste wartości ({best_model['name']})")
plt.show()

# ===========================
# ZADANIE 2: MODEL DLA ZBIORU TITANIC
# ===========================

print("\n\n=== ZADANIE 2: MODEL DLA ZBIORU TITANIC ===\n")


def generate_titanic_data(n_samples=800):
    """Generuje przykładowe dane w stylu Titanic"""
    np.random.seed(42)

    # Generowanie cech
    pclass = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.3, 0.4])
    sex = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # 0=męski, 1=żeński
    age = np.random.normal(30, 15, n_samples)
    age = np.clip(age, 1, 80)
    sibsp = np.random.poisson(0.5, n_samples)
    parch = np.random.poisson(0.3, n_samples)
    fare = np.random.lognormal(3, 1, n_samples)

    # Generowanie survival na podstawie cech (kobiety i wyższa klasa mają większe szanse)
    survival_prob = 0.2 + 0.3 * sex + 0.2 * (4 - pclass) / 3 + 0.1 * (age < 16)
    survival_prob = np.clip(survival_prob, 0, 1)
    survived = np.random.binomial(1, survival_prob, n_samples)

    df = pd.DataFrame(
        {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Survived": survived,
        }
    )

    return df


# Wczytanie danych Titanic
titanic_data_path = (
    "/Users/janbanot/Dev/uni/msc-cs-code/sem2/JSWAD/data/Titanic-Dataset.csv"
)
try:
    titanic_df = pd.read_csv(titanic_data_path)
    print("Wczytano dane Titanic z pliku Titanic-Dataset.csv")

    # Przygotowanie danych
    if "Sex" in titanic_df.columns:
        le = LabelEncoder()
        titanic_df["Sex"] = le.fit_transform(titanic_df["Sex"])

    # Wybór odpowiednich kolumn i usunięcie brakujących wartości
    required_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]
    available_columns = [col for col in required_columns if col in titanic_df.columns]
    titanic_df = titanic_df[available_columns].dropna()

except FileNotFoundError:
    print(
        f"Plik {titanic_data_path} nie został znaleziony. Generowanie przykładowych danych..."
    )
    titanic_df = generate_titanic_data()
    titanic_df.to_csv("titanic_generated.csv", index=False)
    print("Wygenerowano przykładowe dane Titanic")

print(f"\nKształt danych Titanic: {titanic_df.shape}")
print("\nPierwsze 5 wierszy:")
print(titanic_df.head())

# Analiza danych Titanic
print("\n=== ANALIZA DANYCH TITANIC ===")
print(titanic_df.describe())
print(f"\nWskaźnik przeżycia: {titanic_df['Survived'].mean():.3f}")

# Przygotowanie danych Titanic
X_titanic = titanic_df.drop("Survived", axis=1)
y_titanic = titanic_df["Survived"]

# Podział danych
X_train_t, X_temp_t, y_train_t, y_temp_t = train_test_split(
    X_titanic, y_titanic, test_size=0.4, random_state=42
)
X_val_t, X_test_t, y_val_t, y_test_t = train_test_split(
    X_temp_t, y_temp_t, test_size=0.5, random_state=42
)

# Normalizacja
scaler_t = StandardScaler()
X_train_t_scaled = scaler_t.fit_transform(X_train_t)
X_val_t_scaled = scaler_t.transform(X_val_t)
X_test_t_scaled = scaler_t.transform(X_test_t)

print("\nRozmiary zbiorów Titanic:")
print(f"Treningowy: {X_train_t.shape[0]} próbek")
print(f"Walidacyjny: {X_val_t.shape[0]} próbek")
print(f"Testowy: {X_test_t.shape[0]} próbek")


def create_classification_model(input_shape, activation="relu"):
    """Tworzy model klasyfikacyjny dla Titanic"""
    model = keras.Sequential(
        [
            layers.Dense(64, activation=activation, input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation=activation),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),  # Sigmoid dla klasyfikacji binarnej
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Trenowanie modelu Titanic
print("\n=== TRENOWANIE MODELU TITANIC ===")
titanic_model = create_classification_model(X_train_t_scaled.shape[1])

early_stopping_t = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history_t = titanic_model.fit(
    X_train_t_scaled,
    y_train_t,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_t_scaled, y_val_t),
    callbacks=[early_stopping_t],
    verbose=1,
)

# Ocena modelu Titanic
y_pred_t = titanic_model.predict(X_test_t_scaled, verbose=0)
y_pred_t_binary = (y_pred_t > 0.5).astype(int)

accuracy = accuracy_score(y_test_t, y_pred_t_binary)
print(f"\nDokładność modelu Titanic: {accuracy:.3f}")
print("\nRaport klasyfikacji:")
print(classification_report(y_test_t, y_pred_t_binary))

# Wizualizacja wyników trenowania
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Historia trenowania modelu mieszkaniowego (najlepszy)
axes[0, 0].plot(best_model["history"].history["loss"], label="Treningowy")
axes[0, 0].plot(best_model["history"].history["val_loss"], label="Walidacyjny")
axes[0, 0].set_title("Funkcja straty - Model mieszkaniowy")
axes[0, 0].set_xlabel("Epoka")
axes[0, 0].set_ylabel("MSE")
axes[0, 0].legend()

axes[0, 1].plot(best_model["history"].history["mae"], label="Treningowy")
axes[0, 1].plot(best_model["history"].history["val_mae"], label="Walidacyjny")
axes[0, 1].set_title("MAE - Model mieszkaniowy")
axes[0, 1].set_xlabel("Epoka")
axes[0, 1].set_ylabel("MAE")
axes[0, 1].legend()

# Historia trenowania modelu Titanic
axes[1, 0].plot(history_t.history["loss"], label="Treningowy")
axes[1, 0].plot(history_t.history["val_loss"], label="Walidacyjny")
axes[1, 0].set_title("Funkcja straty - Model Titanic")
axes[1, 0].set_xlabel("Epoka")
axes[1, 0].set_ylabel("Binary Crossentropy")
axes[1, 0].legend()

axes[1, 1].plot(history_t.history["accuracy"], label="Treningowy")
axes[1, 1].plot(history_t.history["val_accuracy"], label="Walidacyjny")
axes[1, 1].set_title("Dokładność - Model Titanic")
axes[1, 1].set_xlabel("Epoka")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("\n=== PODSUMOWANIE ===")
print(f"Najlepszy model mieszkaniowy: {best_model['name']}")
print(f"- R² Score: {best_model['r2']:.3f}")
print(f"- MSE: {best_model['mse']:,.0f}")
print(f"- MAE: {best_model['mae']:,.0f}")
print("\nModel Titanic:")
print(f"- Dokładność: {accuracy:.3f}")
print(f"- Liczba epok: {len(history_t.history['loss'])}")
