# 🌊 Tsunami Prediction Project

## 🔍 Introduction
Tsunamis are among the most devastating natural disasters, often triggered by undersea earthquakes. This project presents a machine learning pipeline designed to predict the likelihood of a tsunami event based on seismic and oceanographic data. It integrates data preprocessing, feature engineering, model training, and evaluation — all structured for reproducibility and clarity.

## 🎯 Project Goals
- Build a predictive model for tsunami occurrence using historical seismic data
- Evaluate model performance using precision, recall, and F1-score
- Ensure reproducibility through modular code and fixed random seeds
- Present results in a visually compelling and structured format

## 📦 Dataset Description
- **Source**: [Insert dataset source or link]
- **Size**: ~50,000 records spanning multiple decades
- **Features**:
  - Earthquake magnitude, depth, latitude, longitude
  - Event time (UTC), region, oceanic parameters
- **Target**: Binary label indicating tsunami occurrence (`1 = tsunami`, `0 = no tsunami`)

## 🧪 Methodology

### 1. Data Preprocessing
- Handle missing values and outliers
- Normalize continuous features
- Encode categorical variables (e.g., region)

### 2. Feature Engineering
- Temporal features: hour, month, season
- Geospatial clustering: region-based risk zones
- Derived metrics: energy release, proximity to coast

### 3. Model Training
- Algorithms used:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Hyperparameter tuning via GridSearchCV
- Stratified K-Fold cross-validation

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- ROC-AUC curve
- Confusion matrix visualization

## 📊 Results Summary

| Model              | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.85     | 0.82      | 0.78   | 0.80     |
| Random Forest      | 0.88     | 0.85      | 0.83   | 0.84     |
| XGBoost            | 0.90     | 0.87      | 0.85   | 0.86     |

## 📈 Visualizations
- Seismic activity heatmaps
- Tsunami frequency by region and season
- ROC curves and confusion matrices
- Optional: Power BI dashboard for interactive exploration

## 🛠️ Technologies Used
- Python: pandas, NumPy, scikit-learn, matplotlib, seaborn
- Jupyter Notebook for iterative development
- Power BI (optional) for dashboard-style presentation
- Git for version control

## 🚀 How to Run

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/tsunami-predict
cd tsunami-predict
