# Datasets

This project uses two real-world datasets provided through [scikit-learn](https://scikit-learn.org/). **No manual download is required** — the datasets are fetched automatically at runtime via the `sklearn.datasets` API and cached locally by scikit-learn.

## Table of Contents

- [California Housing](#california-housing)
  - [Overview](#overview)
  - [Features](#features)
  - [Descriptive Statistics](#descriptive-statistics)
  - [Key Characteristics](#key-characteristics)
  - [Preprocessing](#preprocessing)
  - [Loading the Data](#loading-the-data)
  - [References](#references)
- [Forest CoverType](#forest-covertype)
  - [Overview](#overview-1)
  - [Features](#features-1)
  - [Class Distribution](#class-distribution)
  - [Key Characteristics](#key-characteristics-1)
  - [Preprocessing](#preprocessing-1)
  - [Loading the Data](#loading-the-data-1)
  - [References](#references-1)

---

## California Housing

### Overview

| | |
|---|---|
| **Task** | Regression |
| **Samples** | 20,640 |
| **Features** | 8 (all numeric, continuous) |
| **Target** | Median house value (in units of $100,000 USD) |
| **Missing Values** | None |
| **Source** | U.S. Census 1990 (Pace & Barry, 1997) |
| **scikit-learn API** | [`sklearn.datasets.fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |

The California Housing dataset was constructed from the 1990 U.S. Census. Each observation corresponds to a **census block group** — the smallest geographical unit used in census statistics, typically containing 600–3,000 people. The dataset covers the entire state of California, with each block group represented by a centroid point for spatial analysis.

### Features

| Feature | Type | Description |
|---|---|---|
| `MedInc` | Float | Median income of households in the block group |
| `HouseAge` | Float | Median age of houses in the block group |
| `AveRooms` | Float | Average number of rooms per household |
| `AveBedrms` | Float | Average number of bedrooms per household |
| `Population` | Float | Total population of the block group |
| `AveOccup` | Float | Average number of occupants per household |
| `Latitude` | Float | Latitude of the block group centroid |
| `Longitude` | Float | Longitude of the block group centroid |
| **`MedHouseVal`** | **Float** | **Median house value (target, in $100k)** |

### Descriptive Statistics

| Statistic | MedInc | HouseAge | AveRooms | AveBedrms | Population | AveOccup | Latitude | Longitude | MedHouseVal |
|---|---|---|---|---|---|---|---|---|---|
| Mean | 3.87 | 28.64 | 5.43 | 1.10 | 1,425.48 | 3.07 | 35.63 | −119.57 | 2.07 |
| Std | 1.90 | 12.59 | 2.47 | 0.47 | 1,132.46 | 10.39 | 2.14 | 2.00 | 1.15 |
| Min | 0.50 | 1.00 | 0.85 | 0.33 | 3.00 | 0.69 | 32.54 | −124.35 | 0.15 |
| Max | 15.00 | 52.00 | 141.91 | 34.07 | 35,682.00 | 1,243.33 | 41.95 | −114.31 | 5.00 |

### Key Characteristics

- **Target capping:** `MedHouseVal` is capped at 5.0 ($500,000), creating a visible spike at the upper bound in the histogram.
- **Feature capping:** `HouseAge` is capped at 52, suggesting data truncation.
- **Outliers:** Significant outliers exist in `AveRooms`, `AveBedrms`, `Population`, and `AveOccup` — often due to resort/vacation areas with few households but many rooms.
- **Strong predictor:** `MedInc` has the strongest linear correlation (0.69) with the target variable.
- **Spatial dependency:** `Latitude` and `Longitude` exhibit strong non-linear relationships with house prices, reflecting California's geography.
- **Multicollinearity:** `Latitude` and `Longitude` are strongly negatively correlated (−0.92), and `AveRooms` and `AveBedrms` are strongly positively correlated (0.85).

### Preprocessing

The following preprocessing steps are applied in this project:

1. **Missing values** — None present; no imputation needed.
2. **Standardization** — Features are standardized to zero mean and unit variance (z-score normalization) to ensure uniform scale across features and improve gradient descent convergence.
3. **Train/Val/Test split** — Data is split into 60% train, 20% validation, and 20% test with a fixed random seed for reproducibility.

### Loading the Data

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names
```

Or use the project's built-in loader:

```python
from Part1_Regression.dataset import CaliforniaHousingDataset

dataset = CaliforniaHousingDataset()
dataset.split(train_size=0.6, val_size=0.2, test_size=0.2, random_state=42)

# Access splits via attributes
X_train, y_train = dataset.X_train, dataset.y_train
X_val, y_val = dataset.X_val, dataset.y_val
X_test, y_test = dataset.X_test, dataset.y_test
```

### References

- Pace, R. K., & Barry, R. (1997). Sparse spatial autoregressions. *Statistics and Probability Letters*, 33(3), 291–297.
- [scikit-learn documentation — California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

---

## Forest CoverType

### Overview

| | |
|---|---|
| **Task** | Multi-class Classification (7 classes) |
| **Samples** | 581,012 |
| **Features** | 54 (10 quantitative + 44 binary) |
| **Target** | Forest cover type (integer 1–7) |
| **Missing Values** | None |
| **Source** | US Forest Service (USFS) & US Geological Survey (USGS) (Blackard, 1998) |
| **scikit-learn API** | [`sklearn.datasets.fetch_covtype`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html) |

The Forest CoverType dataset was built to predict forest cover type from cartographic variables only (no remotely sensed data). Each observation corresponds to a **30×30 meter forest cell**. The study area covers four wilderness areas in the Roosevelt National Forest in northern Colorado: Rawah, Neota, Comanche Peak, and Cache la Poudre.

### Features

#### Quantitative Features (10)

| Feature | Type | Description |
|---|---|---|
| `Elevation` | Integer | Elevation in meters |
| `Aspect` | Integer | Aspect in degrees azimuth (0–360) |
| `Slope` | Integer | Slope in degrees |
| `Horizontal_Distance_To_Hydrology` | Integer | Horizontal distance to nearest water source (m) |
| `Vertical_Distance_To_Hydrology` | Integer | Vertical distance to nearest water source (m) |
| `Horizontal_Distance_To_Roadways` | Integer | Horizontal distance to nearest roadway (m) |
| `Hillshade_9am` | Integer | Hillshade index at 9 AM (0–255) |
| `Hillshade_Noon` | Integer | Hillshade index at noon (0–255) |
| `Hillshade_3pm` | Integer | Hillshade index at 3 PM (0–255) |
| `Horizontal_Distance_To_Fire_Points` | Integer | Horizontal distance to nearest fire point (m) |

#### Binary Features (44)

| Feature Group | Count | Description |
|---|---|---|
| `Wilderness_Area_1` – `Wilderness_Area_4` | 4 | One-hot encoded wilderness area designation |
| `Soil_Type_1` – `Soil_Type_40` | 40 | One-hot encoded soil type designation |

#### Target Variable

| Value | Cover Type |
|---|---|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas Fir |
| 7 | Krummholz |

### Class Distribution

| Cover Type | Count | Percentage |
|---|---|---|
| 1 — Spruce/Fir | 211,840 | 36.46% |
| 2 — Lodgepole Pine | 283,301 | 48.76% |
| 3 — Ponderosa Pine | 35,754 | 6.15% |
| 4 — Cottonwood/Willow | 2,747 | 0.47% |
| 5 — Aspen | 9,493 | 1.63% |
| 6 — Douglas Fir | 17,367 | 2.99% |
| 7 — Krummholz | 20,510 | 3.53% |

> **⚠️ Class Imbalance:** Classes 1 and 2 together account for **85.22%** of all samples, while class 4 represents only **0.47%**. This significant imbalance must be addressed during model training (e.g., via class-weighted loss, stratified sampling).

### Key Characteristics

- **Large-scale dataset:** Over 580,000 samples, making it suitable for evaluating scalability of models.
- **Severe class imbalance:** Dominated by classes 1 (Spruce/Fir) and 2 (Lodgepole Pine); class 4 (Cottonwood/Willow) is extremely rare.
- **Mixed feature types:** Combines continuous topographic features with binary categorical encodings.
- **Spatial structure:** The four wilderness areas have distinct elevation profiles and species compositions:
  - *Neota* — highest elevation, primarily Spruce/Fir.
  - *Rawah & Comanche Peak* — Lodgepole Pine dominant, with Spruce/Fir and Aspen.
  - *Cache la Poudre* — lowest elevation, characterized by Ponderosa Pine, Douglas Fir, and Cottonwood/Willow.
- **Right-skewed distributions:** Most quantitative features (Slope, distances) exhibit right-skewed distributions.
- **No missing values:** The dataset is complete with no missing entries.

### Preprocessing

The following preprocessing steps are applied in this project:

1. **Missing values** — None present; no imputation needed.
2. **Standardization** — Quantitative features are standardized to zero mean and unit variance. Binary features are left unchanged.
3. **Train/Val/Test split** — Data is split into 60% train, 20% validation, and 20% test with stratified sampling to preserve class proportions.

### Loading the Data

```python
from sklearn.datasets import fetch_covtype

data = fetch_covtype()
X, y = data.data, data.target
feature_names = data.feature_names
```

Or use the project's built-in loader:

```python
from Part2_Classification.dataset import CovtypeDataset

dataset = CovtypeDataset()
dataset.split(train_size=0.6, val_size=0.2, test_size=0.2, random_state=42)

# Access splits via attributes
X_train, y_train = dataset.X_train, dataset.y_train
X_val, y_val = dataset.X_val, dataset.y_val
X_test, y_test = dataset.X_test, dataset.y_test
```

### References

- Blackard, J. (1998). Covertype. *UCI Machine Learning Repository*. DOI: [10.24432/C50K5N](https://doi.org/10.24432/C50K5N)
- [scikit-learn documentation — Forest CoverType](https://scikit-learn.org/stable/datasets/real_world.html#forest-covertypes)
