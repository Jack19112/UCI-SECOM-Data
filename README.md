# Semiconductor Manufacturing — Predictive Quality Control

> Detecting process failures in semiconductor fabrication using 590 sensor signals, statistical feature selection, and ensemble ML, applied to the real-world UCI SECOM dataset.

---

## Problem Statement

In semiconductor manufacturing, each wafer passes through hundreds of process steps monitored by sensors. A failed wafer costs thousands of euros in materials and machine time. Detecting failures **before** they propagate — or understanding which sensors signal an upcoming failure — is one of the most valuable applications of machine learning in industrial production.

This project builds a full predictive quality pipeline on the **UCI SECOM dataset**: a real semiconductor manufacturing dataset with 590 anonymised sensor readings, extreme class imbalance, heavy missingness, and high multicollinearity — exactly the messy, high-dimensional data you encounter in production environments.

---

## Dataset

| Property | Value |
|---|---|
| Source |(https://www.kaggle.com/datasets/paresh2047/uci-semcom/data)|
| Samples | 1,567 production runs |
| Original features | 590 sensor signals |
| Target | Pass (−1 → 0) / Fail (1 → 1) |
| Time range | July – November 2008 |
| Class distribution | **93.36% Pass / 6.64% Fail** |
| Total missing cells | 41,950 (4.54% overall) |

The severe class imbalance (≈ 14:1) and high dimensionality make this a challenging real-world problem — not a clean Kaggle toy dataset.

---

## Analysis Pipeline

### 1. Missing Value Analysis
- **41,950** missing cells across 590 features
- Stratified missing rate by class: Pass rows average **4.55%** missing, Fail rows **4.32%** — confirming missingness is **not informative** of failure (MAR, not MNAR)
- Classified features into 4 tiers: Complete (53), Low-missing (485), Medium-missing (20), High-missing (32)
- **Dropped 32 features** with >50% missing — some with up to 91.2% null rate

### 2. Variance Analysis
- **116 zero-variance features** identified — constant sensor readings carrying no information
- Near-zero variance threshold: `1.23 × 10⁻⁷`
- **Total features dropped: 148** (116 zero-variance + 32 high-missing)
- **Remaining for analysis: 442 features**

### 3. Distribution Analysis
- **293 features highly skewed** (|skew| > 1)
- **243 features extremely skewed** (|skew| > 2) — typical for sensor data with rare fault events
- Kurtosis analysis revealed extreme leptokurtic distributions (kurtosis > 1,500 in top features), indicating rare but extreme sensor excursions
- IQR outlier analysis: top features show 14–23% outlier rates — consistent with process upsets

### 4. Statistical Feature Selection — Mann-Whitney U + BH Correction
Rather than simple correlation, a **non-parametric Mann-Whitney U test** was applied to each of the 442 features to identify which sensors show statistically different distributions between Pass and Fail runs. P-values were corrected for multiple testing using the **Benjamini-Hochberg FDR procedure** (α = 0.05).

| Result | Value |
|---|---|
| Features tested | 442 |
| Significant features (raw p < 0.05) | ~40 |
| **Significant after BH correction** | **17** |
| Top feature (Feature 59) p-value | 5.68 × 10⁻¹¹ |

**Volcano plot** visualises the joint effect size (mean difference) and statistical significance — features in the upper right quadrant are both large in effect and highly significant.

### 5. Correlation Analysis
- **349 highly correlated feature pairs** (|r| > 0.9) — expected in sensor data where multiple sensors monitor related process parameters
- Correlation heatmap computed on top 50 highest-variance features
- Target correlation: Feature 59 shows the strongest linear relationship with the Pass/Fail outcome (|r| = 0.156)

### 6. Temporal Analysis
- Failures resampled at weekly frequency reveal **clustering patterns** — failures are not uniformly distributed over time
- 7-day rolling means of top features show **sensor drift** over the production period
- **Inter-failure gap statistics:**

| Statistic | Value |
|---|---|
| Mean gap between failures | 71.5 hours |
| Median gap | 8.67 hours |
| Min gap | 0.05 hours |
| Max gap | 741.4 hours |

The large gap between mean and median indicates failures often cluster in bursts, followed by long stable periods — a classic pattern in semiconductor equipment degradation.

### 7. Dimensionality Reduction

Three dimensionality reduction methods applied on the full cleaned, imputed, and scaled feature set:

| Method | Key Finding |
|---|---|
| **PCA** | 86 components explain 80% variance · 128 for 90% · 160 for 95% · confirms high redundancy |
| **t-SNE** | Applied on PCA-50 pre-reduction for speed · partial class separation visible in 2D embedding |
| **UMAP** | Cleaner cluster structure than t-SNE · fail samples form identifiable sub-clusters |

The UMAP embedding reveals that failure samples are not randomly distributed in feature space — they occupy specific regions — supporting the hypothesis that failures are predictable from sensor readings.

---

## Modelling

### Feature Selection for Modelling
Top 50 statistically significant features (by corrected p-value from Mann-Whitney analysis) selected as model inputs — reducing dimensionality while preserving the most discriminative signals.

### Preprocessing Pipeline
```
Raw 590 features
    → Drop zero-variance + high-missing (−148 features) → 442
    → Select top-50 statistically significant features
    → Median imputation (SimpleImputer)
    → StandardScaler
    → SMOTE oversampling (93:7 → balanced)
    → 80/20 train/test split
```

**SMOTE** (Synthetic Minority Oversampling Technique) was used to address the severe 14:1 class imbalance before training.

### Results

| Model | AUC-ROC | Accuracy | F1 (Fail class) | Precision (Fail) | Recall (Fail) |
|---|---|---|---|---|---|
| Logistic Regression | 0.8227 | 74% | 0.74 | 0.72 | 0.75 |
| **Random Forest** | **0.9983** | **98%** | **0.98** | 0.97 | 1.00 |
| XGBoost | 0.9979 | 94% | 0.94 | 0.89 | 1.00 |

**Random Forest selected as best model** — AUC-ROC of 0.9983 with 98% accuracy and near-perfect recall on the Fail class (1.00), meaning it catches essentially every failure.

> **Why AUC-ROC matters more than accuracy here:** A naive classifier that always predicts "Pass" achieves 93.4% accuracy — useless in practice. AUC-ROC of 0.9983 means the model correctly ranks a failure above a pass in 99.8% of cases.

### Feature Importance (XGBoost)
The top predictive features align with the statistically significant sensors identified during EDA — confirming that the Mann-Whitney selection was effective. Feature 59 and Feature 103 appear as the two most important sensors across both the statistical analysis and the tree-based importance scores.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Statistical testing | `scipy.stats` (Mann-Whitney U), `statsmodels` (Benjamini-Hochberg) |
| Dimensionality reduction | `sklearn` PCA, t-SNE · `umap-learn` UMAP |
| Outlier analysis | IQR method + Z-score cross-validation |
| Imbalance handling | `imbalanced-learn` SMOTE |
| Machine learning | `scikit-learn` (Logistic Regression, Random Forest) · `xgboost` |
| Preprocessing | `SimpleImputer`, `StandardScaler` |

---

## Project Structure

```
secom-predictive-quality/
├── uci_secom_data.ipynb    ← Full analysis: EDA + stats + modelling
├── uci-secom.csv           ← Dataset (download from UCI link below)
└── README.md
```

---


## Key Findings

1. **Only 17 of 590 sensors** are statistically significant predictors of failure after correcting for multiple comparisons — 97% of sensors carry no additional discriminative signal beyond noise.

2. **Failures cluster in time** — the median inter-failure gap (8.67 hours) is far below the mean (71.5 hours), suggesting equipment degradation events that trigger burst failures before the process stabilises.

3. **Missingness is not informative** — fail and pass samples have nearly identical missingness rates (4.32% vs 4.55%), ruling out "sensor blackout during failure" as a confounding signal.

4. **Random Forest achieves 0.9983 AUC-ROC** with 100% recall on failures — it correctly identifies every failed wafer in the test set, making it production-viable for a zero-miss-failure requirement.

5. **UMAP reveals separable failure clusters** — failures are not randomly scattered in feature space but form identifiable regions, explaining why sensor-based prediction is feasible despite the low failure rate.

---

## Industrial Relevance

This analysis mirrors real workflows in semiconductor fabs, automotive parts manufacturing, and precision optics production. The core pipeline — sensor data ingestion → high-dimensional feature reduction → statistical significance testing → ensemble classification — is directly applicable to:

- **Yield optimisation** in chip fabrication
- **Predictive maintenance** for CNC machines and laser systems
- **Quality gate automation** in automotive component lines

---
