# JSwAD - Laboratorium 4 - zadanie B - uczenie maszynowe - Jan Banot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

numeric = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

penguins = sns.load_dataset("penguins").dropna()

# 1. Etykiety numeryczne dla wybranych kolumn
label_encoder = LabelEncoder()
for col in ["species", "island", "sex"]:
    penguins[col] = label_encoder.fit_transform(penguins[col])

print("Encoded DataFrame head:")
print(penguins.head())
print("-" * 30)

# 2. Klasyfikator KNN dla płci pingwinów
X_sex = penguins.drop("sex", axis=1)  # Renamed to X_sex for clarity
y_sex = penguins["sex"]  # Renamed to y_sex for clarity

X_sex_train, X_sex_test, y_sex_train, y_sex_test = train_test_split(
    X_sex, y_sex, test_size=0.3, random_state=42
)

knn_classifier_sex = KNeighborsClassifier(n_neighbors=3)  # Renamed for clarity
knn_classifier_sex.fit(X_sex_train, y_sex_train)
y_pred_sex = knn_classifier_sex.predict(X_sex_test)  # Renamed for clarity
accuracy_sex = accuracy_score(y_sex_test, y_pred_sex)  # Renamed for clarity
f1_sex = f1_score(y_sex_test, y_pred_sex, average="weighted")  # Renamed for clarity

print("KNN Classifier for Penguin Sex (n_neighbors=3):")
print(f"Accuracy: {accuracy_sex:.4f}")
print(f"F1-score: {f1_sex:.4f}")

# 3. Wyznaczenie optymalnego parametru n_neighbors dla płci
best_n_neighbors_sex = 0  # Renamed for clarity
best_accuracy_sex = 0.0  # Renamed for clarity
best_f1_score_sex = 0.0  # Renamed for clarity
print("\nFinding optimal n_neighbors for SEX (1 to 20):")

for n in range(1, 21):
    knn_classifier_loop_sex = KNeighborsClassifier(n_neighbors=n)  # Renamed for clarity
    knn_classifier_loop_sex.fit(X_sex_train, y_sex_train)
    y_pred_loop_sex = knn_classifier_loop_sex.predict(X_sex_test)  # Renamed for clarity

    current_accuracy_sex = accuracy_score(
        y_sex_test, y_pred_loop_sex
    )  # Renamed for clarity
    current_f1_sex = f1_score(
        y_sex_test, y_pred_loop_sex, average="weighted"
    )  # Renamed for clarity

    print(
        f"  n_neighbors = {n}: Accuracy = {current_accuracy_sex:.4f}, F1-score = {current_f1_sex:.4f}"
    )

    if current_accuracy_sex > best_accuracy_sex:
        best_accuracy_sex = current_accuracy_sex
        best_f1_score_sex = current_f1_sex
        best_n_neighbors_sex = n
    elif (
        current_accuracy_sex == best_accuracy_sex and current_f1_sex > best_f1_score_sex
    ):
        best_f1_score_sex = current_f1_sex
        best_n_neighbors_sex = n

print("-" * 30)
print(f"Optimal n_neighbors for SEX: {best_n_neighbors_sex}")
print(f"Best Accuracy for SEX: {best_accuracy_sex:.4f}")
print(
    f"Best F1-score for SEX (at optimal n_neighbors for accuracy): {best_f1_score_sex:.4f}"
)
print("-" * 30)

# 4. Klasyfikator KNN dla gatunku pingwinów - Wyznaczenie optymalnego parametru n_neighbors
print("\nFinding optimal n_neighbors for SPECIES (1 to 20):")
X_species = penguins.drop("species", axis=1)
y_species = penguins["species"]

X_species_train, X_species_test, y_species_train, y_species_test = train_test_split(
    X_species, y_species, test_size=0.3, random_state=42
)

best_n_neighbors_species = 0
best_accuracy_species = 0.0
best_f1_score_species = 0.0

for n in range(1, 21):
    knn_classifier_loop_species = KNeighborsClassifier(n_neighbors=n)
    knn_classifier_loop_species.fit(X_species_train, y_species_train)
    y_pred_loop_species = knn_classifier_loop_species.predict(X_species_test)

    current_accuracy_species = accuracy_score(y_species_test, y_pred_loop_species)
    current_f1_species = f1_score(
        y_species_test, y_pred_loop_species, average="weighted"
    )

    print(
        f"  n_neighbors = {n}: Accuracy = {current_accuracy_species:.4f}, F1-score = {current_f1_species:.4f}"
    )

    if current_accuracy_species > best_accuracy_species:
        best_accuracy_species = current_accuracy_species
        best_f1_score_species = current_f1_species
        best_n_neighbors_species = n
    elif (
        current_accuracy_species == best_accuracy_species
        and current_f1_species > best_f1_score_species
    ):
        best_f1_score_species = current_f1_species
        best_n_neighbors_species = n

print("-" * 30)
print(f"Optimal n_neighbors for SPECIES: {best_n_neighbors_species}")
print(f"Best Accuracy for SPECIES: {best_accuracy_species:.4f}")
print(
    f"Best F1-score for SPECIES (at optimal n_neighbors for accuracy): {best_f1_score_species:.4f}"
)
print("-" * 30)
