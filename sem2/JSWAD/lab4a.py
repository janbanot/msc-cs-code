# JSwAD - Laboratorium 4 - zadanie A - uczenie maszynowe - Jan Banot
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 1. Generowanie zboru danych do klasyfikacji
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,  # 5 cech informacyjnych
    n_redundant=2,  # 2 cechy redundantne
    n_classes=2,  # Klasyfikacja binarna
    random_state=42,  # Dla uzyskania powtarzalności
)

print(f"Shape of features (X): {X.shape}")
print(f"Shape of target (y): {y.shape}")

# 2. Podział zbioru danych na zbiór treningowy i testowy
# Zbiór testowy: 40%, Zbiór treningowy: 60%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,  # Dla uzyskania powtarzalnych podziałów
)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# 3a. Uczenie modelu z wykorzystaniem algorytmu Gaussian Naive Bayes
print("\nTraining Gaussian Naive Bayes model")
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

# 4a. Ewaluacja modelu (algorytm GNB)
print("\nEvaluating Gaussian Naive Bayes model")
y_pred_gnb = gnb_model.predict(X_test)
y_pred_proba_gnb = gnb_model.predict_proba(X_test)[:, 1]

accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
f1_gnb = f1_score(y_test, y_pred_gnb)
roc_auc_gnb = roc_auc_score(y_test, y_pred_proba_gnb)

print(f"GaussianNB - Accuracy: {accuracy_gnb:.4f}")
print(f"GaussianNB - F1-score: {f1_gnb:.4f}")
print(f"GaussianNB - AUC ROC: {roc_auc_gnb:.4f}")

# 3b. Uczenie modelu z wykorzystaniem algorytmu SVC (Support Vector Classifier)
print("\nTraining and Evaluating SVC model with RBF kernel")
C_values = [1e-02, 1e-01, 1e00, 1e01, 1e02]

for C_val in C_values:
    print(f"\nSVC with C={C_val}")
    svc_model = SVC(C=C_val, kernel="rbf", probability=True, random_state=42)
    svc_model.fit(X_train, y_train)

    # 4b. Ewaluacja modelu (algorytm SVC)
    y_pred_svc = svc_model.predict(X_test)
    y_pred_proba_svc = svc_model.predict_proba(X_test)[:, 1]

    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    f1_svc = f1_score(y_test, y_pred_svc)
    roc_auc_svc = roc_auc_score(y_test, y_pred_proba_svc)

    print(f"SVC (C={C_val}) - Accuracy: {accuracy_svc:.4f}")
    print(f"SVC (C={C_val}) - F1-score: {f1_svc:.4f}")
    print(f"SVC (C={C_val}) - AUC ROC: {roc_auc_svc:.4f}")

# 3c. Uczenie modelu z wykorzystaniem algorytmu RandomForestClassifier
print("\nTraining and Evaluating RandomForestClassifier model")
n_estimators_values = [10, 100, 1000]

for n_est in n_estimators_values:
    print(f"\nRandomForestClassifier with n_estimators={n_est}")

    rf_model = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf_model.fit(X_train, y_train)

    # 4c. Ewaluacja modelu (algorytm RandomForestClassifier)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

    print(
        f"RandomForestClassifier (n_estimators={n_est}) - Accuracy: {accuracy_rf:.4f}"
    )
    print(f"RandomForestClassifier (n_estimators={n_est}) - F1-score: {f1_rf:.4f}")
    print(f"RandomForestClassifier (n_estimators={n_est}) - AUC ROC: {roc_auc_rf:.4f}")
