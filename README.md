#  Near-Earth Objects Hazard Classification

A machine learning project that predicts whether a near-Earth object (asteroid) is **potentially hazardous** using NASA data covering **338,000+ close-approach records from 1910 to 2024**.

> Built as part of the Data Science Fundamentals program – Microsoft Student Club EELU (Feb 2025).

##  Results

| Metric | Score |
|---|---|
| Accuracy | **93%** |
| ROC-AUC | **0.98** |

##  Workflow

1. **Data cleaning:** checked and removed missing values
2. **Exploratory data analysis:**
   - Distribution of absolute magnitude
   - Absolute magnitude vs. estimated diameter
   - Relative velocity outliers (boxplot)
   - Correlation heatmap of numerical features
3. **Preprocessing:**
   - Features: `absolute_magnitude`, `estimated_diameter_min`, `estimated_diameter_max`, `relative_velocity`, `miss_distance`
   - Balanced the classes by downsampling the non-hazardous majority class
   - Stratified 80/20 train/test split and `StandardScaler`
4. **Modeling:** `RandomForestClassifier` (100 trees)
5. **Evaluation:** accuracy, classification report, ROC curve and AUC

##  Tech Stack

Python · Pandas · Matplotlib · Seaborn · Scikit-learn

## Dataset

[NASA Nearest Earth Objects (1910–2024)](https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024) on Kaggle.
Download the CSV and place it in a `data/` folder:

```
data/nearest-earth-objects(1910-2024).csv
```

##  How to Run

```bash
pip install -r requirements.txt
python neo_hazard_classification.py
```

Or open the code in Google Colab / Jupyter and run it cell by cell.
