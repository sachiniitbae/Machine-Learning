import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import train_test_split


bikesData = pd.read_csv("hour.csv")
#bikesData["dteday"].dtypes
#bikesData["weathersit"].value_counts()
#bikesData.info()
#len(bikeData.columns)

#bikesData["yr"].unique()
#len(bikesData["yr"].unique())

bikesData["hum"].mean()
bikesData.describe()
columnsToDrop = ["dteday","atemp","registered","casual","instant"]
bikesData = bikesData.drop(columnsToDrop,axis =1)
columnsToScale = ["temp","hum","windspeed"]

scaler = StandardScaler()
bikesData[columnsToScale] = scaler.fit_transform(bikesData[columnsToScale])
#print(bikesData[columnsToScale])
#print(bikesData.shape[0])
bikesData["dayCount"] = pd.Series(range(bikesData.shape[0]))/24
#bikesData

train_set,test_set = train_test_split(bikesData,test_size = 0.3,random_state = 42)
train_set.sort_values("dayCount",axis = 0,inplace = True)
test_set.sort_values("dayCount",axis = 0,inplace = True)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor

trainingCols = train_set.drop(["cnt"],axis=1)
trainingLabels = train_set["cnt"]
#trainingLabels


###DecsionTreeRegression
dec_reg = DecisionTreeRegressor(random_state = 42)
dt_mae_scores= -cross_val_score(dec_reg,trainingCols,trainingLabels,cv = 10,scoring = "neg_mean_absolute_error")
display_scores(dt_mae_scores)
dt_mse_scores = np.sqrt(-cross_val_score(dec_reg,trainingCols,trainingLabels,cv = 10, scoring="neg_mean_squared_error"))
display_scores(dt_mse_scores)


###LinearRegression
lin_reg = LinearRegression()
lr_mae_scores  = -cross_val_score(lin_reg,trainingCols,trainingLabels,cv=10,scoring = "neg_mean_absolute_error")
display_scores(lr_mae_scores)
lr_mse_scores = np.sqrt(-cross_val_score(lin_reg,trainingCols,trainingLabels,cv=10,scoring="neg_mean_squared_error"))
display_scores(lr_mse_scores)

###ForestRegression
forest_reg = RandomForestRegressor()

rf_mae_scores = -cross_val_score(forest_reg,trainingCols,trainingLabels,cv=10,scoring = "neg_mean_absolute_error")
display_scores(rf_mae_scores)
rf_mse_scores = np.sqrt(-cross_val_score(forest_reg,trainingCols,trainingLabels,cv=10,scoring="neg_mean_squared_error"))
display_scores(rf_mse_scores)


""" fine-tuning  RandomForest model."""
 
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_estimators': [120, 150], "max_features":[10,12],"max_depth":[15,28]},]
grid_search = GridSearchCV(forest_reg,param_grid,cv=10,scoring="neg_mean_squared_error")
grid_search.fit(trainingCols,trainingLabels)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

final_model = grid_search.best_estimator_
test_set.sort_values('dayCount', axis= 0, inplace=True)
test_x_cols = (test_set.drop(['cnt'], axis=1)).columns.values
test_y_cols = 'cnt'

X_test = test_set.loc[:,test_x_cols]
y_test = test_set.loc[:,test_y_cols]

test_set.loc[:,'predictedCounts_test'] = final_model.predict(X_test)

mse = mean_squared_error(y_test, test_set.loc[:,'predictedCounts_test'])
final_mse = np.sqrt(mse)
print("Final_rmse: ",final_mse)
test_set.describe()