"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit

Goal: build a classification model on a small dataset.
Pipeline (IPO): Input (Iris dataset + scaling) -> Process (train/test split + KNN)
-> Output (confusion matrix + F1 score).
"""

from typing import cast
from sklearn.utils import Bunch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pandas as pd

# ---------- INPUT ----------
iris = cast(Bunch, load_iris())
X = iris.data          # 4 features: sepal length/width, petal length/width
y = iris.target        # 3 classes: setosa, versicolor, virginica

print("Dataset loaded:", X.shape[0], "samples,", X.shape[1], "features")
print("Classes:", list(iris.target_names))

# ---------- PROCESS ----------
# Structural integrity: shuffle + split BEFORE fitting and scaling
x_train_data, x_test_data, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Gatekeeper rule: scale features after splitting to prevent data leakage
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_data)
x_test_scaled = scaler.transform(x_test_data)

model = KNeighborsClassifier(n_neighbors=5)   # instantiate
model.fit(x_train_scaled, y_train)             # fit (memorize the map)
predictions = model.predict(x_test_scaled)     # predict (apply logic)

# ---------- OUTPUT ----------
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, predictions)
print(pd.DataFrame(
    cm,
    index=[f"actual_{n}" for n in iris.target_names],
    columns=[f"pred_{n}" for n in iris.target_names],
))

print("\n--- Classification Report ---")
print(classification_report(y_test, predictions, target_names=iris.target_names))

f1 = f1_score(y_test, predictions, average="weighted")
print(f"Weighted F1 Score: {f1:.4f}")

accuracy = model.score(x_test_scaled, y_test)
print(f"Accuracy: {accuracy:.4f}")
