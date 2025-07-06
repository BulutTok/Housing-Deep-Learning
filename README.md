
# Ames Housing Price Analysis Using Deep Learning

Comprehensive Jupyter Notebook delivering an end-to-end workflow on the Ames Housing dataset: from data ingestion and cleaning, through exploratory data analysis (EDA) and dimensionality reduction, to building, training, and evaluating a neural network regression model.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Notebook Setup](#notebook-setup)
3. [Data Loading & Overview](#data-loading--overview)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
6. [Preprocessing Pipelines](#preprocessing-pipelines)
7. [Neural Network Model](#neural-network-model)
8. [Model Training & Results](#model-training--results)
9. [Contributing](#contributing)
10. [License & Contact](#license--contact)

---

## Project Overview

This project demonstrates:

* **Data ingestion** from CSV (via Google Colab or local files).
* **EDA** to summarize distributions, missingness, and key relationships.
* **PCA** to reduce dimensionality while retaining ≥ 80% variance.
* **Preprocessing** pipelines for numeric and categorical attributes.
* **Neural network regressor** built with TensorFlow/Keras to predict `SalePrice`.
* **Evaluation** using MSE, RMSE, and R² on a held-out validation set.

---

## Notebook Setup

Execute these cells when running on Google Colab:

```python
from google.colab import drive
# Mount Google Drive if not already
drive.mount('/content/drive')

# Copy CSV files from Drive to the working directory
!cp "/content/drive/MyDrive/Colab Notebooks/Housing/train.csv" .
!cp "/content/drive/MyDrive/Colab Notebooks/Housing/test.csv"  .
```

If running locally, ensure `train.csv` and `test.csv` reside in the project root.

---

## Data Loading & Overview

```python
import pandas as pd
# 1. Load the training data
df = pd.read_csv('train.csv')
print(df.shape)  # (1460, 81)
```

* **Shape**: 1,460 rows × 81 columns.
* **Data types**: mixture of `int64`, `float64`, and `object` for categorical fields.

---

## Exploratory Data Analysis (EDA)

### 1. Summary Statistics

```python
print(df.describe().loc[['mean','std','min','50%','max'], ['LotArea','GrLivArea','SalePrice']])
```

| Statistic | LotArea    | GrLivArea | SalePrice  |
| --------- | ---------- | --------- | ---------- |
| mean      | 10516.83   | 1515.46   | 180,921.20 |
| std       | 9,981.26   | 525.48    | 79,442.50  |
| min       | 1,300.00   | 334.00    | 34,900.00  |
| 50%       | 9,478.50   | 1,484.00  | 163,000.00 |
| max       | 215,245.00 | 4,691.00  | 755,000.00 |

### 2. Missing Values

```python
missing = df.isnull().sum().sort_values(ascending=False)
missing[missing>0].head(10)
```

Top missing columns:

* `PoolQC`: 1453 nulls
* `MiscFeature`: 1406 nulls
* `Alley`: 1369 nulls
* `Fence`: 1179 nulls
* `MasVnrType`: 872 nulls

### 3. Visualizations

* **Histograms**: Distribution of numeric features (e.g., `LotArea`, `OverallQual`).
* **Boxplots**: Outlier detection for `SalePrice`, `GrLivArea`.
* **Correlation Heatmap**: Positive correlation of `OverallQual` (ρ ≈ 0.79) and `GrLivArea` (ρ ≈ 0.71) with `SalePrice`.

---

## Principal Component Analysis (PCA)

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

num_cols = df.select_dtypes(include='number').drop(['Id','SalePrice'], axis=1).columns
pca_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.80, svd_solver='full'))
])
pca_pipeline.fit(df[num_cols])
n_comp = pca_pipeline.named_steps['pca'].n_components_
print(n_comp)  
```

* **Number of components** to explain ≥ 80% variance: **18**.
* **Top 5 explained variance ratios**:

| PC  | ExplainedVarRatio | CumulativeVarRatio |
| --- | ----------------- | ------------------ |
| PC1 | 0.1979            | 0.1979             |
| PC2 | 0.0890            | 0.2869             |
| PC3 | 0.0715            | 0.3583             |
| PC4 | 0.0562            | 0.4145             |
| PC5 | 0.0410            | 0.4555             |

* **First 5 loadings** highlight how features contribute to PCs (e.g., `OverallQual` loads 0.30 on PC1).

---

## Preprocessing Pipelines

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Numeric pipeline: median imputation + standard scaling\ nnum_pipeline = Pipeline([...])
# Categorical pipeline: most-frequent imputation + one-hot encoding\ ncat_pipeline = Pipeline([...])

preprocess = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])
```

This ensures consistent transformation for both training and test sets.

---

## Neural Network Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_squared_error']
)
```

* **Architecture**: 3 hidden layers (128, 128, 64 units)
* **Loss**: MSE
* **Optimizer**: Adam

---

## Model Training & Results

### 1. Full Training

```python
model.fit(X_train, y_train, batch_size=5, epochs=150)
# Final training MSE ≈ 137,242,448
```

### 2. Train/Validation Split

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    batch_size=5,
    epochs=150,
    verbose=2
)
y_pred_val = model.predict(X_val)

mse  = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
r2   = r2_score(y_val, y_pred_val)
print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
```

**Validation Performance:** The model’s **Mean Squared Error (MSE)** on the validation set is **411,008,576**, and its **Root Mean Squared Error (RMSE)** is **20,273**. In practical terms, RMSE of \$20,273 means that, on average, predicted sale prices differ from actual prices by about \$20k—an acceptable margin given the typical price range. The **R² score** of **0.946** indicates that the model captures **94.6%** of the variability in sale prices, reflecting excellent explanatory power and strong generalization to unseen data.

---

## Contributing

We welcome contributions! To contribute:

1. **Fork** this repository to your GitHub account.
2. **Create** a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit** your changes with a clear message:

   ```bash
   git commit -m "Add feature: brief description"
   ```
4. **Push** your branch to GitHub:

   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open** a Pull Request against the `main` branch and describe your changes.

Please ensure code style consistency and include relevant tests or notebook updates when applicable.

---

## License & Contact

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Author:** Bulut Tok
**Contact:** [buluttok2013@gmail.com]

Feel free to open issues or pull requests on GitHub for questions, suggestions, or bug reports.
