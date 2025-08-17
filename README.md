# Dallas Rental Price Prediction
Author: Ziyue Gong

Predict rental prices for apartment listings in **Dallas, Texas** using a cleaned subset of the “Apartment for Rent Classified” dataset from the UCI Machine-Learning Repository.

---

## 1. Project Overview
| Item | Details |
|------|---------|
| **Problem** | Estimate monthly rent from listing attributes. |
| **Dataset** | 99 826 records, 20 raw columns → filtered to **Dallas, TX** (≈ 2 500 rows). |
| **Target** | `price` (USD, monthly). |
| **Models Tested** | ElasticNet (linear) vs. Histogram-Based Gradient Boosting Regressor (HGB). |
| **Best Result** | HGBR with **MSE ≈ 344**, **R² ≈ 0.98** on test set. |

---

## 2. Data Pipeline

### 2.1 Feature Selection  
After exploratory plots and uniqueness checks we kept:

| Kept | Reason |
|------|--------|
| `bathrooms`, `bedrooms`, `square_feet` | Numeric drivers of price. |
| `amenities`, `has_photo`, `pets_allowed` | Categorical signals of quality / restrictions. |

Redundant or single-value columns for Dallas (e.g., `category`, `currency`, `state`) were dropped.

### 2.2 Cleaning & Encoding  
| Feature | Action |
|---------|--------|
| `bathrooms`, `bedrooms`, `square_feet` | Cast to numeric, drop ≤3 NaNs. |
| `amenities` | Parsed list → top-k items → `MultiLabelBinarizer`. |
| `has_photo` | One-hot: *Yes*, *Thumbnail*, *No*. |
| `pets_allowed` | Binary: 1 if any pets allowed, else 0. |

> **Log-transform** applied to `price` before model training.

### 2.3 Train / Test Split  
```text
80 % train  |  20 % test   (random_state = 42)
