# 🚘 Car Price Prediction with Random Forest

Predicting the resale value of used cars is a classic regression challenge with real-world impact. This project uses a **Random Forest Regressor** to estimate car prices based on historical listings, offering a robust and interpretable model for dealerships, buyers, and data enthusiasts.

---

## 🎯 Project Goal

To build a machine learning model that accurately predicts the selling price of used cars using features like age, fuel type, kilometers driven, and ownership history.

---

## 🧠 Why Random Forest?

Random Forest is a powerful ensemble method that:
- Handles non-linear relationships and feature interactions
- Is resilient to outliers and noise
- Provides feature importance for interpretability

---

## 📦 Dataset Overview

The dataset includes:
- **Year**: Year of manufacture
- **Present_Price**: Current ex-showroom price
- **Kms_Driven**: Distance driven
- **Fuel_Type**: Petrol, Diesel, or CNG
- **Seller_Type**: Dealer or Individual
- **Transmission**: Manual or Automatic
- **Owner**: Number of previous owners
- **Selling_Price**: Target variable

---

## 🔍 Workflow Summary

1. **Data Cleaning**: Removed duplicates, handled missing values
2. **Feature Engineering**: Created 'Car_Age', encoded categorical variables
3. **Model Training**: Used RandomForestRegressor with train-test split
4. **Evaluation**: Assessed performance using R², MAE, RMSE
5. **Visualization**: Plotted feature importance and prediction accuracy

---

## 📊 Results

- **R² Score**: ~0.92 on test data
- **Top Features**: Present_Price, Car_Age, Kms_Driven
- **Insights**: Newer cars with lower mileage and petrol engines tend to retain higher resale value

---

## 📁 Folder Structure

