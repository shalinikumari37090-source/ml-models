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


df = pd.read_excel("Untitled.xlsx")

df['int_cat'] = pd.cut(df['Interest_Rate'], bins=[0,2.0,4.0,6.0,8.0,np.inf], labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(df,df['int_cat']):
    strat_train_set = df.iloc[train_index].drop(['int_cat'], axis=1)
    strat_test_set = df.iloc[test_index].drop(['int_cat'], axis=1)

data = strat_train_set.copy()

data_fea = data.drop(["Default_Risk"],axis=1)
data_labels = data["Default_Risk"]

num_attribs = num_attribs_1 = data_fea.drop(["Education_Level","Marital_Status","Home_Ownership","Loan_Purpose"], axis=1)

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
# print(num_attribs)

cat_attribs = data_fea[["Education_Level","Marital_Status","Home_Ownership","Loan_Purpose"]].columns.tolist()

# print(cat_attribs)

num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline  ,num_attribs),
    ("cat", cat_pipeline , cat_attribs)
])

data_prepared = full_pipeline.fit_transform(data_fea)

# print(banking_prepared.shape)

rid_class = RidgeClassifier()
rid_class.fit(data_prepared,data_labels)
rid_preds = rid_class.predict(data_prepared)
# rid_rmse  = root_mean_squared_error(data_labels,rid_preds)
rid_rmses = -cross_val_score(rid_class, data_prepared,data_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(rid_rmses).describe())


tree_class = DecisionTreeClassifier()
tree_class.fit(data_prepared,data_labels)
tree_preds = tree_class.predict(data_prepared)
tree_rmse  = root_mean_squared_error(data_labels,tree_preds)
tree_rmses = -cross_val_score(tree_class, data_prepared,data_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(tree_rmses).describe())
# print(tree_rmse)


random_class = RandomForestClassifier()
random_class.fit(data_prepared,data_labels)
random_preds = random_class.predict(data_prepared)
random_rmse  = root_mean_squared_error(data_labels,random_preds)
random_rmses = -cross_val_score(random_class, data_prepared,data_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(random_rmses).describe())

# print(random_rmse)


sdg_class = SGDClassifier()
sdg_class.fit(data_prepared,data_labels)
sdg_preds = sdg_class.predict(data_prepared)
sdg_rmse  = root_mean_squared_error(data_labels,sdg_preds)
sdg_rmses = -cross_val_score(sdg_class, data_prepared,data_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(sdg_rmses).describe())

# print(sdg_rmse)