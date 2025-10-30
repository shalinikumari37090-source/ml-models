# 🌍 Earthquake Tsunami Prediction using Random Forest Classifier

This project uses a Random Forest Classifier to predict whether an earthquake event is likely to trigger a tsunami. The model is trained on a real-world dataset of global earthquakes from 2015 to 2022, with features capturing seismic intensity, location, and geophysical parameters.

---

## 📁 Dataset Overview

The dataset includes over 300 earthquake records with the following features:

| Feature       | Description |
|---------------|-------------|
| `magnitude`   | Richter scale magnitude of the earthquake |
| `cdi`         | Community Internet Intensity Map (perceived shaking) |
| `mmi`         | Modified Mercalli Intensity (instrumental) |
| `sig`         | Significance of the event (higher = more significant) |
| `nst`         | Number of seismic stations that reported the event |
| `dmin`        | Minimum distance to the nearest station (degrees) |
| `gap`         | Azimuthal gap between stations (degrees) |
| `depth`       | Depth of the earthquake (km) |
| `latitude`    | Latitude of the epicenter |
| `longitude`   | Longitude of the epicenter |
| `Year`, `Month` | Temporal features |
| `tsunami`     | Target variable (1 = tsunami occurred, 0 = no tsunami) |

---

## 🧠 Objective

To build a robust classification model that can predict the likelihood of a tsunami based on seismic and geospatial features of an earthquake.

---

## 🛠️ Tools & Libraries

- Python (pandas, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook / VS Code
- Random Forest Classifier (`sklearn.ensemble`)
- GridSearchCV for hyperparameter tuning
- Power BI / matplotlib for visualizations (optional)

---

## 📊 Workflow

1. **Data Preprocessing**
   - Handled missing values and outliers
   - Feature scaling and encoding (if needed)
   - Feature selection based on correlation and importance

2. **Model Training**
   - Split data into training and test sets (e.g., 80/20)
   - Trained Random Forest Classifier
   - Tuned hyperparameters using GridSearchCV

3. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix and ROC-AUC Curve
   - Feature importance visualization

4. **Deployment (Optional)**
   - Saved model using `joblib` or `pickle`
   - Created an interactive dashboard or API endpoint

---

## 📈 Sample Results

- Accuracy: ~XX% (fill in your result)
- Top Features: `magnitude`, `mmi`, `sig`, `depth`, `cdi`
- ROC-AUC: ~XX%

---

## 📌 Key Insights

- Earthquakes with higher magnitude, shallow depth, and high significance are more likely to trigger tsunamis.
- Geographical clustering shows tsunami-prone zones near subduction boundaries (e.g., Pacific Ring of Fire).

---

## 📂 Folder Structure
