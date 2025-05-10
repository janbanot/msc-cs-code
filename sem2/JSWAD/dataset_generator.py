from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Generate the dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,  # Example: 5 informative features
    n_redundant=2,  # Example: 2 redundant features
    n_classes=2,  # Binary classification
    random_state=42,  # For reproducibility
)

print(f"Shape of features (X): {X.shape}")
print(f"Shape of target (y): {y.shape}")

# 2. Split the dataset into training and test sets
# Test set: 40%, Training set: 60%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,  # For reproducible splits
)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
