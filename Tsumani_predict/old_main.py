from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

data = pd.read_csv("earthquake.csv")

data['mmi_cat'] = pd.cut(data['mmi'] , bins = [0.0,3.0,6.0,9.0,12.0,np.inf] , labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=2 , test_size=0.2 , random_state=42 )

for train_index , test_index in split.split(data , data['mmi_cat']):
    train_set = data.loc[train_index].drop('mmi_cat', axis = 1)
    test_set = data.loc[test_index].drop('mmi_cat', axis = 1)

df = train_set.copy() 

df_features = df.drop('tsunami', axis = 1)
df_labels = df['tsunami']

pipeline = Pipeline([
    ("scaler" , StandardScaler())
])

df_prepared = pipeline.fit_transform(df_features)

dec_mod = DecisionTreeClassifier()
dec_mod.fit(df_prepared,df_labels)
dec_pred = dec_mod.predict(df_prepared)
dec_error = root_mean_squared_error(dec_pred , df_labels)
cross_val = -cross_val_score(dec_mod , df_prepared , df_labels , scoring="neg_root_mean_squared_error" , cv=10)
print(pd.Series(cross_val).describe())

