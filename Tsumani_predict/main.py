import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODELFILE = "model.pkl"
PIPELINE = "pipeline.pkl"

if not os.path.exists(MODELFILE):

    data = pd.read_csv("earthquake.csv")
    data['mmi_cat'] = pd.cut(data['mmi'] , bins = [0.0,3.0,6.0,9.0,12.0,np.inf] , labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=2 , test_size=0.2 , random_state=42 )

    for train_index , test_index in split.split(data , data['mmi_cat']):
        train_set = data.loc[train_index].drop('mmi_cat', axis = 1)
        test_set = data.loc[test_index].drop('mmi_cat', axis = 1).to_csv("input.csv")

    df = train_set.copy() 

    df_features = df.drop('tsunami', axis = 1)
    df_labels = df['tsunami']

    pipeline = Pipeline([
        ("scaler" , StandardScaler())
    ])

    df_prepared = pipeline.fit_transform(df_features)

    model = RandomForestClassifier()
    model.fit(df_prepared, df_labels)

    joblib.dump(pipeline,PIPELINE)
    joblib.dump(model,MODELFILE)

    print("model is trained")
else:
    model = joblib.load(MODELFILE)
    pipeline = joblib.load(PIPELINE)

    input_data = pd.read_csv("input.csv")

    # Drop label column if present
    if "tsunami" in input_data.columns:
        input_features = input_data.drop("tsunami", axis=1)
    else:
        input_features = input_data.copy()

    # Drop index column if present
    if "Unnamed: 0" in input_features.columns:
        input_features = input_features.drop("Unnamed: 0", axis=1)

    # ✅ Use only cleaned features for transformation
    transformed_data = pipeline.transform(input_features)
    predictions = model.predict(transformed_data)

    input_data["tsunami"] = predictions
    input_data.to_csv("output.csv", index=False)
    print("inference complete")
    
   