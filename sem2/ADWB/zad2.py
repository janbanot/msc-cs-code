# JSwADwBAD - zadanie 2 - algorytm CART - Jan Banot

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Zadanie 13
file_path = "/Users/janbanot/Dev/studia/msc-cs-code/sem2/ADWB/data/decision-tree.csv"
data = pd.read_csv(file_path)

print("Original data head:")
print(data.head())
print("\nData types:")
print(data.dtypes)

data_processed = data.copy().drop(columns=["identyfikator"])

categorical_cols = data_processed.select_dtypes(include=["object"]).columns

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data_processed[col] = le.fit_transform(data_processed[col])
    encoders[col] = le

print("\nProcessed data head (after encoding):")
print(data_processed.head())

X = data_processed.iloc[:, :-1]
y = data_processed.iloc[:, -1]

cart_classifier = DecisionTreeClassifier(random_state=42)

cart_classifier.fit(X, y)

print("\nDecision Tree Classifier trained.")

feature_names = X.columns.tolist()

target_col_name = data.columns[-1]
if target_col_name in encoders:
    class_names = encoders[target_col_name].classes_.astype(str).tolist()
else:
    class_names = y.unique().astype(str).tolist()
    class_names.sort()

plt.figure(figsize=(20, 10))
plot_tree(
    cart_classifier,
    filled=True,
    feature_names=feature_names,
    class_names=class_names,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree (CART Algorithm)")
plt.show()
