# ---------------------------------------------------------
# IRIS DATASET PLAYGROUND
# Modern Python + pandas + scikit-learn
# ---------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["class"] = iris.target

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Shape ===")
print(df.shape)

print("\n=== Summary Statistics ===")
print(df.describe())

print("\n=== Column Names ===")
print(df.columns)

print("\n=== Data Types ===")
print(df.dtypes)

# ---------------------------------------------------------
# 2. Exploration
# ---------------------------------------------------------

print("\n=== Unique Classes ===")
print(df["class"].unique())

print("\n=== Samples Per Class ===")
print(df["class"].value_counts())

print("\n=== Correlation Matrix ===")
print(df.corr())

# ---------------------------------------------------------
# 3. Visualizations
# ---------------------------------------------------------

print("\nShowing histograms...")
df.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()

print("\nShowing boxplots...")
df.plot(kind="box", figsize=(8, 6))
plt.tight_layout()
plt.show()

print("\nShowing scatter plot...")
df.plot(kind="scatter",
        x="sepal length (cm)",
        y="petal width (cm)",
        title="Sepal Length vs Petal Width")
plt.show()

# ---------------------------------------------------------
# 4. Machine Learning Example
# ---------------------------------------------------------

X = df.iloc[:, :-1]   # features
y = df["class"]       # labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("\n=== KNN Model Accuracy ===")
