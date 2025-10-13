import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


df = pd.read_csv("housing.csv")

df['income_cat'] = pd.cut(df["median_income"],
                    bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                    labels=[1,2,3,4,5])

split= StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(df,df["income_cat"]):
    strat_train_set = df.loc[train_index].drop('income_cat',axis=1)
    strat_test_set = df.loc[test_index].drop('income_cat',axis=1)

housing = strat_train_set.copy()

housing_labels = housing["median_house_value"]
housing = housing.drop('median_house_value',axis=1)

num_attribs = housing.drop(['ocean_proximity'],axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standard", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)

# print(housing_prepared)
# print(housing_prepared.shape)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_pred = lin_reg.predict(housing_prepared)
lin_error = root_mean_squared_error(housing_labels,lin_pred)
cross_val = -cross_val_score(lin_reg,
                             housing_prepared,
                             housing_labels,
                             scoring="neg_root_mean_squared_error", 
                             cv=10
                             )
# print(pd.Series(cross_val).describe())

dec_tree_reg = DecisionTreeRegressor()
dec_tree_reg.fit(housing_prepared,housing_labels)
dec_tree_pred = dec_tree_reg.predict(housing_prepared)
dec_tree_error = root_mean_squared_error(housing_labels,dec_tree_pred)
cross_val = -cross_val_score(dec_tree_reg,
                             housing_prepared,
                             housing_labels,
                             scoring="neg_root_mean_squared_error", 
                             cv=10
                             )
# print(pd.Series(cross_val).describe())

# print(dec_tree_error)

random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared,housing_labels)
random_forest_pred = random_forest_reg.predict(housing_prepared)
random_forest_error = root_mean_squared_error(housing_labels,random_forest_pred)
cross_val = -cross_val_score(random_forest_reg,
                             housing_prepared,
                             housing_labels,
                             scoring="neg_root_mean_squared_error", 
                             cv=10
                             )
print(pd.Series(cross_val).describe())
# print(random_forest_error)