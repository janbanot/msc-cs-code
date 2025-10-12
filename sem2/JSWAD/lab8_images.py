# JSWAD - Laboratorium 8: Klasyfikacja obrazów z użyciem CNN - Jan Banot

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Sprawdzenie dostępności GPU
print("GPU dostępne:", tf.config.list_physical_devices("GPU"))
print("TensorFlow wersja:", tf.__version__)

# Parametry modelu
img_height = 64
img_width = 64
batch_size = 32
epochs = 20
model_path = "christmas_classifier_model.h5"

# Tworzenie generatorów danych z augmentacją
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalizacja pikseli do [0, 1]
    rotation_range=20,  # Augmentacja: obrót obrazów
    width_shift_range=0.2,  # Augmentacja: przesunięcie poziome
    height_shift_range=0.2,  # Augmentacja: przesunięcie pionowe
    shear_range=0.2,  # Augmentacja: ścinanie
    zoom_range=0.2,  # Augmentacja: przybliżenie
    horizontal_flip=True,  # Augmentacja: odbicie poziome
    brightness_range=[0.8, 1.2],  # Augmentacja: zmiana jasności
    fill_mode="nearest",  # Sposób wypełniania po transformacji
    validation_split=0.2,  # 20% danych na walidację
)

# Generator dla danych testowych (bez augmentacji)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Źródło danych - https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification
train_dir = "/Users/janbanot/Dev/uni/msc-cs-code/sem2/JSWAD/data/is-that-santa/train"
test_dir = "/Users/janbanot/Dev/uni/msc-cs-code/sem2/JSWAD/data/is-that-santa/test"

# Sprawdzenie czy katalogi istnieją
if not os.path.exists(train_dir):
    print(f"UWAGA: Katalog {train_dir} nie istnieje!")
    print("Pobierz dane z Kaggle i umieść je w odpowiedniej strukturze folderów.")
    train_dir = None

if not os.path.exists(test_dir):
    print(f"UWAGA: Katalog {test_dir} nie istnieje!")
    test_dir = None

# Wczytywanie danych treningowych i walidacyjnych
train_dataset = None
validation_dataset = None
test_dataset = None
num_classes = 2  # Domyślna wartość

if train_dir:
    train_dataset = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",  # Dla wielu klas
        subset="training",
    )

    validation_dataset = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )

    # Liczba klas
    num_classes = len(train_dataset.class_indices)
    print(f"Liczba klas: {num_classes}")
    print(f"Klasy: {train_dataset.class_indices}")
else:
    # Przykładowe dane dla demonstracji
    print("Używam przykładowych danych...")

# Wczytywanie danych testowych
if test_dir:
    test_dataset = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,  # Ważne dla ewaluacji
    )


def create_cnn_model(num_classes):
    """
    Tworzy nowy model CNN dla klasyfikacji obrazów
    """
    model = Sequential(
        [
            # Pierwsza warstwa konwolucyjna
            Conv2D(
                32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)
            ),
            MaxPooling2D((2, 2)),
            # Druga warstwa konwolucyjna
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            # Trzecia warstwa konwolucyjna
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            # Spłaszczenie
            Flatten(),
            # Warstwy gęste z dropout
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            # Warstwa wyjściowa
            Dense(num_classes, activation="softmax" if num_classes > 2 else "sigmoid"),
        ]
    )

    return model


def get_or_create_model(model_path, num_classes, force_retrain=False):
    """
    Sprawdza czy model już istnieje. Jeśli tak, wczytuje go.
    Jeśli nie, tworzy nowy model.

    Args:
        model_path (str): Ścieżka do pliku modelu
        num_classes (int): Liczba klas do klasyfikacji
        force_retrain (bool): Wymusza ponowne trenowanie nawet jeśli model istnieje

    Returns:
        tuple: (model, needs_training)
    """
    if os.path.exists(model_path) and not force_retrain:
        print(f"Znaleziono istniejący model: {model_path}")
        try:
            model = load_model(model_path)
            print("Model został pomyślnie wczytany!")
            print("Architektura wczytanego modelu:")
            model.summary()
            return model, False
        except Exception as e:
            print(f"Błąd podczas wczytywania modelu: {e}")
            print("Tworzę nowy model...")
    else:
        if force_retrain:
            print("Wymuszono ponowne trenowanie - tworzę nowy model...")
        else:
            print("Nie znaleziono istniejącego modelu - tworzę nowy...")

    # Tworzenie nowego modelu
    model = create_cnn_model(num_classes)

    # Kompilacja modelu
    if num_classes > 2:
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
    else:
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    print("Architektura nowego modelu:")
    model.summary()

    return model, True


def train_model(model, train_dataset, validation_dataset, epochs, callbacks):
    """
    Trenuje model używając podanych danych

    Args:
        model: Model do wytrenowania
        train_dataset: Zbiór danych treningowych
        validation_dataset: Zbiór danych walidacyjnych
        epochs: Liczba epok
        callbacks: Lista callbacks dla trenowania

    Returns:
        history: Historia trenowania
    """
    print("Rozpoczynam trenowanie modelu...")

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# Funkcja do wizualizacji historii trenowania
def plot_training_history(history):
    """
    Rysuje wykresy dokładności i straty podczas trenowania
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Dokładność
    axes[0].plot(history.history["accuracy"], label="Trenowanie")
    axes[0].plot(history.history["val_accuracy"], label="Walidacja")
    axes[0].set_title("Dokładność modelu")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Dokładność")
    axes[0].legend()

    # Strata
    axes[1].plot(history.history["loss"], label="Trenowanie")
    axes[1].plot(history.history["val_loss"], label="Walidacja")
    axes[1].set_title("Strata modelu")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Strata")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_dataset):
    """
    Ewaluuje model na zbiorze testowym
    """
    if test_dataset is not None:
        print("\nEwaluacja na zbiorze testowym:")
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
        print(f"Strata testowa: {test_loss:.4f}")
        print(f"Dokładność testowa: {test_accuracy:.4f}")
        return test_loss, test_accuracy
    else:
        print("Brak danych testowych do ewaluacji.")
        return None, None


# Funkcja do predykcji pojedynczego obrazu
def predict_image(model, image_path):
    """
    Funkcja do predykcji klasy pojedynczego obrazu
    """
    from tensorflow.keras.preprocessing import image

    # Wczytaj i przygotuj obraz
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predykcja
    prediction = model.predict(img_array, verbose=0)

    if num_classes > 2:
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
    else:
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]

    return predicted_class, confidence


# Główna logika programu
def main():
    # Sprawdź czy chcesz wymusić ponowne trenowanie
    force_retrain = False  # Zmień na True jeśli chcesz wymusić ponowne trenowanie

    # Pobierz lub utwórz model
    model, needs_training = get_or_create_model(model_path, num_classes, force_retrain)

    # Jeśli model wymaga trenowania i mamy dane
    if needs_training and train_dataset is not None:
        # Callbacks dla lepszego trenowania
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=3, min_lr=0.0001
            ),
        ]

        # Trenowanie
        history = train_model(
            model, train_dataset, validation_dataset, epochs, callbacks
        )

        # Wizualizacja historii trenowania
        plot_training_history(history)

        # Zapisanie modelu
        model.save(model_path)
        print(f"\nModel zapisany jako '{model_path}'")

    elif needs_training and train_dataset is None:
        print("Brak danych treningowych. Model nie został wytrenowany.")
        return None
    else:
        print("Używam istniejącego modelu - pomijam trenowanie.")

    # Ewaluacja na zbiorze testowym
    evaluate_model(model, test_dataset)

    return model


if __name__ == "__main__":
    # Uruchom główną funkcję
    model = main()

    # Test predykcji
    if model is not None:
        # Testowanie z obrazem Santa
        test_image_path = "/Users/janbanot/Dev/uni/msc-cs-code/sem2/JSWAD/data/is-that-santa/test/santa/0.Santa.jpg"
        if os.path.exists(test_image_path):
            predicted_class, confidence = predict_image(model, test_image_path)
            print(
                f"\nTest Santa - Przewidywana klasa: {predicted_class}, Pewność: {confidence:.2f}"
            )

        # Testowanie z obrazem nie-Santa
        test_image_path2 = "/Users/janbanot/Dev/uni/msc-cs-code/sem2/JSWAD/data/is-that-santa/test/not-a-santa/0.not-a-santa.jpg"
        if os.path.exists(test_image_path2):
            predicted_class, confidence = predict_image(model, test_image_path2)
            print(
                f"Test nie-Santa - Przewidywana klasa: {predicted_class}, Pewność: {confidence:.2f}"
            )
