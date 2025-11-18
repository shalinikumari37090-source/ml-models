import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# Load input data
input_data = pd.read_csv("input.csv")

# Drop unwanted columns
if "Unnamed: 0" in input_data.columns:
    input_data = input_data.drop("Unnamed: 0", axis=1)

# Separate features and labels
if "tsunami" in input_data.columns:
    true_labels = input_data["tsunami"]
    input_features = input_data.drop("tsunami", axis=1)
else:
    print("⚠️ 'tsunami' column not found in input.csv. Accuracy check skipped.")
    exit()

# Transform features
transformed_features = pipeline.transform(input_features)

# Predict
predicted_labels = model.predict(transformed_features)

# Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(true_labels, predicted_labels))

# Confusion Matrix
print("\n🧮 Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))