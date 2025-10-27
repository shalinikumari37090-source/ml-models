import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier ,SGDClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
    
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline  ,num_attribs),
        ("cat", cat_pipeline , cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_excel("Untitled.xlsx")

    df['int_cat'] = pd.cut(df['Interest_Rate'], bins=[0,2.0,4.0,6.0,8.0,np.inf], labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index,test_index in split.split(df,df['int_cat']):
        df.loc[test_index].drop(['int_cat'], axis=1).to_excel("input.xlsx" , index=False)
        data = df.loc[train_index].drop(['int_cat'], axis=1)

    data_fea = data.drop(["Default_Risk"],axis=1)
    data_labels = data["Default_Risk"]

    num_attribs = data_fea.drop(["Education_Level","Marital_Status","Home_Ownership","Loan_Purpose"], axis=1)
    
    col_list = ['Age',
    'Credit_Score',
    'Debt_to_Income',
    'Employment_Years',
    'ID',
    'Income',
    'Interest_Rate',
    'Loan_Amount',
    'Loan_Term',
    'Property_Value']
    num_attribs = num_attribs[col_list].apply(pd.to_numeric,errors="coerce").columns.tolist()
    cat_attribs = data_fea[["Education_Level","Marital_Status","Home_Ownership","Loan_Purpose"]].columns.tolist()

    pipeline = build_pipeline(num_attribs,cat_attribs)
    data_prepared = pipeline.fit_transform(data_fea)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(data_prepared,data_labels)

    joblib.dump(model , MODEL_FILE)
    joblib.dump(pipeline , PIPELINE_FILE)

    print("model is trained . Congrats")

else:

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_excel("input.xlsx")
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data["Default_Risk"] = predictions 

    input_data.to_excel("output.xlsx" , index=False)
    print("inference is complete. Result saved to output.xlsx")   

