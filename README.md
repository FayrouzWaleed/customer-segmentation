# Task 2: Customer Segmentation

Clusters mall customers into segments by annual income and spending score.

## Dataset
[Mall Customer Segmentation Data (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
Download `Mall_Customers.csv` and place it in `data/`.

## What it does
1. Loads and cleans the data
2. Scales features (StandardScaler) and plots the raw groupings
3. Finds the optimal number of clusters (elbow method + silhouette score)
4. Applies K-Means and visualizes clusters in 2D

**Bonus**
- DBSCAN clustering for comparison
- Average income and spending score per cluster

## Run
```bash
pip install -r requirements.txt
python main.py
```
Plots and a clustered CSV are saved to `outputs/`.

## Tools
Python, Pandas, Matplotlib, Scikit-learn
