import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pileline.pkl"

def build_pipeline(num_attribs,cat_attribs):
    num_pipeline = Pipeline([
    ("scaler" , StandardScaler()),
    ("impute", SimpleImputer(strategy="median"))
    ])

    cat_pipeline = Pipeline([
        ("encoder" , OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num" , num_pipeline , num_attribs),
        ("cat" , cat_pipeline , cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):

    
    df = pd.read_csv("car_details.csv")

    owner_map = {
        'First Owner':1,
    'Fourth & Above Owner':2,
    'Second Owner':3,
    'Test Drive Car':4,
    'Third Owner':5
    }

    df["owner_cat"] = df['owner'].map(owner_map)

    split = StratifiedShuffleSplit(n_splits=1 , test_size= 0.2 , random_state= 42)

    for train_index,test_index in split.split(df,df['owner_cat']):
        df.loc[test_index].drop('owner_cat', axis=1).to_csv("input.csv")
        data = df.loc[train_index].drop('owner_cat', axis=1)

    
    data_features = data.drop('selling_price',axis=1)
    data_labels = data['selling_price']

    num_attribs = data_features.drop(['name','fuel','seller_type','transmission','owner','mileage','engine','max_power','torque'], axis=1).columns.tolist()
    cat_attribs = data_features.drop(['year','km_driven','seats'],axis=1).columns.tolist()

    pipeline = build_pipeline(num_attribs,cat_attribs)
    data_prepared = pipeline.fit_transform(data_features)

    model = LinearRegression()
    model.fit(data_prepared,data_labels)

    joblib.dump(model , MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("traning is done . Congrats !")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data['selling_price'] = predictions
    input_data.to_csv("output.csv")

    print("inference complete")

