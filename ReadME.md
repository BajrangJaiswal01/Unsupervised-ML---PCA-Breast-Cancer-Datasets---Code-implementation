# PCA — Breast Cancer Dataset Quick Revision

## 1. What is PCA?

**PCA (Principal Component Analysis)** is a dimensionality reduction technique.

Its main purpose is:

> Convert many features into fewer new features (Principal Components) while retaining as much variance as possible.

Example:

```text
30 Features
     ↓ PCA
2 Principal Components
```

PCA does **not** simply delete columns. It creates new features by combining the original features through linear combinations.

---

## 2. Loading the Breast Cancer Dataset

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
```

---

## 3. Separating X and y

```python
X = data.data
y = data.target
```

* `X` → input features
* `y` → target/class

Expected shapes:

```text
X → (569, 30)
y → (569,)
```

Meaning:

* 569 samples
* 30 features
* 569 target values

### Important confusion

After:

```python
X = data.data
```

`X` is already a NumPy array.

Therefore:

```python
X.data
X.feature_names
```

❌ Wrong.

Feature names are available through:

```python
data.feature_names
```

---

## 4. Converting X into a DataFrame

If X and y are already separated:

```python
import pandas as pd

df = pd.DataFrame(X,columns=data.feature_names)
```

Check:

```python
df.shape
```

Output:

```text
(569, 30)
```

---

## 5. Why do we scale the data?

PCA is based on **variance**.

If features have very different scales, large-scale features can have a disproportionately large influence on the variance.

Example:

```text
mean radius → around 10–30
mean area   → around 100–2500
```

So we standardize the features:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)
```

After scaling approximately:

```text
Mean = 0
Standard Deviation = 1
```

### Important confusion

Scaling does **not** automatically modify the original DataFrame.

```python
df
```

→ original values

```python
X_scaled
```

→ scaled values

If you want a scaled DataFrame:

```python
df_scaled = pd.DataFrame(
    X_scaled,
    columns=df.columns
)
```

---

## 6. Applying PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

Before PCA:

```text
569 × 30
```

After PCA:

```text
569 × 2
```

So:

```text
30 Features
     ↓
    PCA
     ↓
PC1 + PC2
```

`X_pca` now contains two new features:

* PC1
* PC2

---

## 7. What are PC1 and PC2?

### PC1

The principal direction that captures the **maximum variance** in the data.

### PC2

The next principal direction that captures the **next highest variance**, and is orthogonal to PC1.

In simple terms:

```text
PC1 → highest variance
PC2 → second highest variance
PC3 → third highest variance
...
```

---

## 8. What does variance mean here?

Variance can be understood as:

> **How spread out the data is.**

Low spread:

```text
50, 51, 50, 52
```

→ low variance

High spread:

```text
20, 40, 60, 80, 100
```

→ high variance

PCA looks for directions in which the data has high variance and creates Principal Components from those directions.

---

## 9. Explained Variance Ratio

```python
pca.explained_variance_ratio_
```

Example:

```text
[0.44, 0.19]
```

Meaning:

```text
PC1 → 44% variance
PC2 → 19% variance
```

Total:

```python
pca.explained_variance_ratio_.sum()
```

Example:

```text
0.63
```

Meaning:

> PC1 + PC2 together represent approximately 63% of the total variance in the standardized dataset.

### Technical note

Saying **"63% information retained"** is a simplified explanation.

Technically:

> The selected principal components represent approximately 63% of the total variance in the standardized dataset.
>
> ##

---

## 10. PCA Visualization

```python
import matplotlib.pyplot as plt

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Breast Cancer Dataset")

plt.show()
```

Graph:

```text
X-axis → PC1
Y-axis → PC2
```

Each dot represents one sample/patient.

---

## 13. What does `c=y` mean?

To visually distinguish the target classes:

```python
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)
```

`c=y` means:

> Use the value of `y` to determine the color of each point.

For the Breast Cancer dataset:

```text
y = 0 → malignant
y = 1 → benign
```

This allows us to visually inspect whether the two classes appear separated or overlapping in the PC1-PC2 space.

### Important

`y` was **not used to create PCA**.

```text
X_scaled
   ↓
  PCA
   ↓
 X_pca
```

`y` is only used for visualization.

Therefore, PCA is an **unsupervised dimensionality reduction technique**.
