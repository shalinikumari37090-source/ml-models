import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODELFILE = "model.pkl"
PIPELINE = "pipeline.pkl"

if not os.path.exists(MODELFILE):

    # Load and preprocess data
    data = pd.read_csv("earthquake.csv")
    data['mmi_cat'] = pd.cut(data['mmi'], bins=[0.0, 3.0, 6.0, 9.0, 12.0, np.inf], labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(data, data['mmi_cat']):
        train_set = data.loc[train_index].drop('mmi_cat', axis=1)
        test_set = data.loc[test_index].drop('mmi_cat', axis=1)
        test_set.to_csv("input.csv", index=False)

    # Separate features and labels
    df_features = train_set.drop('tsunami', axis=1)
    df_labels = train_set['tsunami']

    # Build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    df_prepared = pipeline.fit_transform(df_features)

    # Train model
    model = RandomForestClassifier()
    model.fit(df_prepared, df_labels)

    # Save pipeline and model
    joblib.dump(pipeline, PIPELINE)
    joblib.dump(model, MODELFILE)

    print("model is trained")

    # ✅ Accuracy Evaluation
    test_features = test_set.drop('tsunami', axis=1)
    test_labels = test_set['tsunami']
    test_prepared = pipeline.transform(test_features)
    test_predictions = model.predict(test_prepared)

    acc = accuracy_score(test_labels, test_predictions)
    print(f"Accuracy on test set: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(test_labels, test_predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(test_labels, test_predictions))

else:
    # Load model and pipeline
    model = joblib.load(MODELFILE)
    pipeline = joblib.load(PIPELINE)

    input_data = pd.read_csv("input.csv")

    # Prepare input features
    if "tsunami" in input_data.columns:
        input_features = input_data.drop("tsunami", axis=1)
    else:
        input_features = input_data.copy()

    if "Unnamed: 0" in input_features.columns:
        input_features = input_features.drop("Unnamed: 0", axis=1)

    transformed_data = pipeline.transform(input_features)
    predictions = model.predict(transformed_data)

    input_data["tsunami"] = predictions
    input_data.to_csv("output.csv", index=False)
    print("inference complete")

# ✅ Accuracy Check (if true labels are available in input.csv)
if "tsunami" in input_data.columns:
    true_labels = input_data["tsunami"]
    predicted_labels = predictions

    acc = accuracy_score(true_labels, predicted_labels)
    print(f"\n✅ Model Accuracy on input.csv: {acc:.4f}")
    print("\n📊 Classification Report:\n", classification_report(true_labels, predicted_labels))
    print("\n🧮 Confusion Matrix:\n", confusion_matrix(true_labels, predicted_labels))
else:
    print("\n⚠️ Accuracy check skipped: 'tsunami' column not found in input.csv.")    