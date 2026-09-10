# SmartCart Customer Segmentation System

Unsupervised machine learning project that segments e-commerce customers into distinct groups based on demographics, spending behavior, and purchase history. The insights can be used to drive targeted marketing, personalized offers, and customer retention strategies.

## Overview

The notebook (`e-commerce_customer_segmentation_system.ipynb`) walks through a full customer segmentation pipeline on the `smartcart_customers.csv` dataset:

1. **Exploratory Data Analysis (EDA)** — inspect shape, structure, and missing values.
2. **Data Preprocessing** — clean and impute missing values.
3. **Feature Engineering** — derive meaningful features (Age, Tenure, Total Spending, etc.).
4. **Outlier Removal** — filter unrealistic Age and Income values.
5. **Encoding & Scaling** — prepare categorical and numerical features for modeling.
6. **Dimensionality Reduction** — PCA for visualization and clustering.
7. **Cluster Analysis** — determine the optimal number of clusters (Elbow Method + Silhouette Score).
8. **Clustering** — K-Means and Agglomerative Clustering.
9. **Cluster Characterization** — profile each customer segment.

## Dataset

The project expects a CSV file named `smartcart_customers.csv` in the project root, containing customer records with fields such as:

- `Year_Birth`, `Education`, `Marital_Status`, `Income`
- `Kidhome`, `Teenhome`, `Dt_Customer`, `Recency`
- Spending columns: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- `Response` and other campaign/behavioral fields

> Note: This dataset is not included in the repository. Add your own copy of `smartcart_customers.csv` to the project root before running the notebook.

## Feature Engineering

| New Feature | Description |
|---|---|
| `Age` | Derived from `Year_Birth` |
| `Customer_Tenure_Days` | Days since customer joined, relative to the most recent join date |
| `Total_Spending` | Sum of all product spending categories |
| `Total_Children` | Sum of `Kidhome` and `Teenhome` |
| `Education` (regrouped) | Simplified into `Undergraduate`, `Graduate`, `Postgraduate` |
| `Living_With` | Simplified marital status into `Partner` or `Alone` |

## Methodology

- **Preprocessing**: Missing `Income` values are filled with the median. Irrelevant/raw columns (`ID`, `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`, and individual spending columns) are dropped after feature engineering.
- **Outlier Removal**: Records with `Age >= 90` or `Income >= 600,000` are excluded.
- **Encoding**: Categorical features (`Education`, `Living_With`) are one-hot encoded using `OneHotEncoder`.
- **Scaling**: All features are standardized with `StandardScaler`.
- **Dimensionality Reduction**: `PCA` reduces the feature space to 3 components for clustering and visualization.
- **Optimal Cluster Count**: Determined using the **Elbow Method** (WCSS + `KneeLocator`) and **Silhouette Score**.
- **Clustering Algorithms**:
  - `KMeans` (k=4)
  - `AgglomerativeClustering` (ward linkage, k=4)
- **Cluster Profiling**: Cluster sizes, income vs. spending scatter plots, and per-cluster feature averages (`groupby("cluster").mean()`).

## Tech Stack

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (`OneHotEncoder`, `StandardScaler`, `PCA`, `KMeans`, `AgglomerativeClustering`, `silhouette_score`)
- kneed (`KneeLocator`)

## Installation

```bash
git clone https://github.com/<your-username>/smartcart-customer-segmentation.git
cd smartcart-customer-segmentation
pip install -r requirements.txt
```

### requirements.txt

```
pandas
numpy
matplotlib
seaborn
scikit-learn
kneed
jupyter
```

## Usage

1. Place `smartcart_customers.csv` in the project root.
2. Launch Jupyter Notebook:

   ```bash
   jupyter notebook e-commerce_customer_segmentation_system.ipynb
   ```

3. Run all cells sequentially to reproduce the preprocessing, clustering, and visualizations.

## Results

The pipeline segments customers into **4 distinct clusters** based on income, spending, age, and household composition. Each cluster represents a different customer profile (e.g., high-income high-spenders, budget-conscious families, etc.), visualized via:

- 3D PCA scatter plots colored by cluster
- Cluster size distribution (count plot)
- Income vs. Total Spending scatter plot by cluster
- Per-cluster mean feature summary table

## Project Structure

```
.
├── e-commerce_customer_segmentation_system.ipynb   # Main analysis notebook
├── smartcart_customers.csv                          # Dataset (not included)
├── requirements.txt                                  # Python dependencies
└── README.md                                         # Project documentation
```

## Future Improvements

- Add cluster labeling/naming based on business interpretation (e.g., "Premium Loyalists", "Budget Families").
- Experiment with DBSCAN or Gaussian Mixture Models for comparison.
- Build an interactive dashboard (Streamlit/Plotly Dash) for exploring segments.
- Automate hyperparameter selection for `k` and PCA components.

## License

This project is open-source and available under the [MIT License](LICENSE).
