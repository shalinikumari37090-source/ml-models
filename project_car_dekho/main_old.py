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
    strat_train_set = df.loc[train_index].drop('owner_cat', axis=1)
    strat_test_set = df.loc[test_index].drop('owner_cat', axis=1)

data = strat_train_set.copy()

data_features = data.drop('selling_price',axis=1)
data_labels = data['selling_price']

num_attribs = data_features.drop(['name','fuel','seller_type','transmission','owner','mileage','engine','max_power','torque'], axis=1).columns.tolist()
cat_attribs = data_features.drop(['year','km_driven','seats'],axis=1).columns.tolist()

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

data_prepared = full_pipeline.fit_transform(data_features)
# print(data_prepared)

dec_mod = DecisionTreeRegressor()
dec_mod.fit(data_prepared,data_labels)
dec_preds = dec_mod.predict(data_prepared)
dec_rmse = root_mean_squared_error(data_labels,dec_preds)
dec_rmses = -cross_val_score(dec_mod,data_prepared,data_labels,scoring="neg_root_mean_squared_error",cv=10)
# print(pd.Series(dec_rmses).describe())

random_mod = RandomForestRegressor()
random_mod.fit(data_prepared,data_labels)
random_preds = random_mod.predict(data_prepared)
random_rmse = root_mean_squared_error(data_labels,random_preds)
random_rmses = -cross_val_score(random_mod,data_prepared,data_labels,scoring="neg_root_mean_squared_error",cv=10)
# print(pd.Series(random_rmses).describe())

lin_mod = LinearRegression()
lin_mod.fit(data_prepared,data_labels)
lin_preds = lin_mod.predict(data_prepared)
lin_rmse = root_mean_squared_error(data_labels,lin_preds)
lin_rmses = -cross_val_score(lin_mod,data_prepared,data_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(lin_rmses).describe())

# print(lin_rmse)
