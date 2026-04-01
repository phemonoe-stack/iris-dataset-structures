# iris-dataset-structures
Playing around with Datasets &amp; Python

## 🌟 A Tiny “Iris Dataset Quick Reference” You Can Keep Handy

### 🔧 **Load the dataset**
```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["class"] = iris.target
```

---

### 📘 **Core DataFrame operations**
- `df.head()` — first rows  
- `df.tail()` — last rows  
- `df.shape` — dimensions  
- `df.describe()` — stats summary  
- `df.columns` — column names  
- `df.dtypes` — data types  

---

### 🔍 **Exploration**
- `df["class"].unique()` — class labels  
- `df["class"].value_counts()` — samples per class  
- `df.corr()` — correlations  

---

### 📊 **Quick visualizations**
```python
df.hist()
df.plot(kind="box")
df.plot(kind="scatter", x="sepal length (cm)", y="petal width (cm)")
```

---

### 🤖 **Machine learning basics**
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = df.iloc[:, :-1]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)
```

#-----------------------------------------------

Below is a compact but powerful toolbox you can lean on as you learn.

---

# 🌱 Essential Iris Dataset Functions (Modern Python)

These examples assume:

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["class"] = iris.target
```

---

## 📌 1. Inspecting the Dataset

### **View the first rows**
```python
df.head()
```

### **View the last rows**
```python
df.tail()
```

### **Get dataset shape**
```python
df.shape
```

### **Column names**
```python
df.columns
```

### **Quick summary statistics**
```python
df.describe()
```

### **Data types**
```python
df.dtypes
```

---

## 🔍 2. Exploring the Data

### **Unique class labels**
```python
df["class"].unique()
```

### **Count samples per class**
```python
df["class"].value_counts()
```

### **Correlation matrix**
```python
df.corr()
```

---

## 📊 3. Basic Visualizations (using pandas + matplotlib)

```python
import matplotlib.pyplot as plt
```

### **Histogram**
```python
df.hist(figsize=(8,6))
plt.show()
```

### **Scatter plot**
```python
df.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)")
plt.show()
```

### **Boxplots**
```python
df.plot(kind="box", figsize=(8,6))
plt.show()
```

---

## 🤖 4. Preparing Data for Machine Learning

### **Split into train/test**
```python
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### **Train a simple model**
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X_train, y_train)
```

### **Evaluate**
```python
model.score(X_test, y_test)
```

---

## 🧭 5. Converting the scikit‑learn dataset into a DataFrame (the clean way)

This is the modern, canonical approach:

```python
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["class"] = iris.target
```

No URLs, no CSVs — just clean, in‑memory data.

---
