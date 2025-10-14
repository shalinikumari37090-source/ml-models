import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs,cat_attribs):
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("impute",SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline , num_attribs),
        ("cat", cat_pipeline , cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    
    df = pd.read_csv("car details v4.csv")

    owner_map = { 'First':1, 'Fourth':2, 'Second':3, 'Third':4, 'UnRegistered Car':5,'4 or More':5}
    df['owner_cat'] = df['Owner'].map(owner_map)

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index,test_index in split.split(df,df['owner_cat']):
        df.loc[test_index].drop('owner_cat',axis=1).to_csv("input.csv")
        data = df.loc[train_index].drop('owner_cat',axis=1)

    data_features = data.drop("Price", axis=1)
    data_labels = data['Price']

    num_attribs = data_features[['Year','Kilometer','Length','Width','Height','Seating Capacity','Fuel Tank Capacity']].columns.tolist()
    cat_attribs = data_features[['Make','Model','Fuel Type','Transmission','Location','Color','Owner','Seller Type','Engine','Max Power','Max Torque','Drivetrain']].columns.tolist()  

    pipeline = build_pipeline(num_attribs,cat_attribs)
    data_prepared = pipeline.fit_transform(data_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(data_prepared,data_labels)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    print("model is trained")
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data['Price'] = predictions

    input_data.to_csv("output.csv")
    print("inference complete")
