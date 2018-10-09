# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 06:40:51 2018

@author: thanh.bui
"""

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

#%% 1) Trouvez les données
# Télecharger les données via:
# https://static.oc-static.com/prod/courses/files/initiez-vous-au-machine-learning/house_data.csv
#%% 2) Nettoyez les données
# Charger les donnees
house_df = pd.read_csv('house_data.csv', delimiter=',')
house_df.head()
# Verify statistics
house_df.describe()
# Verifier NaN 
house_df.isnull().sum()
# Drop the rows that contain NaN values
cleaned_house_df = house_df.dropna(axis=0, how='any')

#%% 3) Passez a l'exploration
# On affiche le nuage de pointes dont on dispose
plt.figure()
plt.plot(cleaned_house_df['surface'], cleaned_house_df['price'], 'ro', markersize=4)
plt.xlabel('Surface')
plt.ylabel('Price')
# Plot the histogram
cleaned_house_df.hist(bins=30, figsize=(12,8))
# Scatter plots
scatter_matrix(cleaned_house_df)

#%% 4) Modélisez les données à l'aide du machine learning
X = cleaned_house_df.loc[:,['surface', 'arrondissement']].values
#X = cleaned_house_df.loc[:,'surface'].values.reshape(-1,1)
y = cleaned_house_df.loc[:, 'price'].values.reshape(-1,1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply linear regression
lr_price = LinearRegression()
lr_price.fit(X_train, y_train)
y_lr_pred = lr_price.predict(X_test)
print('Coefficients: \n', lr_price.coef_)
print('Mean squared error: %.2f\n' %mean_squared_error(y_test, y_lr_pred) )


# GridSearch to find the best set of hyperparameters for SVR and RFR
def grid_search_report(results, n_top=3):
    '''Find the first three sets of parameters that provide the highest testing scores 
    '''   
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0} \n".format(results['params'][candidate]))

            
params_svr = {'C': [1, 10, 100, 1000], 'epsilon': [0.05, 0.1, 0.2], 'kernel': ['linear', 'rbf'] }
svr_gs = SVR()
# Scikit-Learn cross-validation features expect a utility function (greater is better) rather than a cost function (lower is better)
grid_search_svr = GridSearchCV(svr_gs, params_svr, cv=5, scoring='neg_mean_squared_error')
grid_search_svr.fit(X_train, y_train)
grid_search_svr.best_params_
grid_search_report(grid_search_svr.cv_results_)

params_rf = {'n_estimators': [10, 50, 100, 200, 500], 'max_features': [1, 2]}
rf_gs = RandomForestRegressor()
# Scikit-Learn cross-validation features expect a utility function (greater is better) rather than a cost function (lower is better)
grid_search_rf = GridSearchCV(rf_gs, params_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)
grid_search_rf.best_params_
grid_search_report(grid_search_rf.cv_results_)

#%% 5) Evaluate et interprétez les résultats
# Support vector machines regression
svr_price = SVR(C=10, epsilon=0.1, kernel='linear')
svr_price.fit(X_train, y_train)
y_svr_pred = svr_price.predict(X_test)
print('Mean squared error for SVR: %.2f\n' %mean_squared_error(y_test, y_svr_pred))

# Random forest regression
rf_price = RandomForestRegressor(n_estimators = 200, max_features=1)  # 200, 1
rf_price.fit(X_train, y_train)
y_rf_pred = rf_price.predict(X_test)
print('Mean squared error rf: %.2f\n' %mean_squared_error(y_test, y_rf_pred))

plt.figure(), plt.plot(y_test), plt.plot(y_lr_pred), plt.plot(y_svr_pred), 
plt.plot(y_rf_pred), plt.legend(['original', 'LR', 'SVR', 'RF'])

#%% 6) Deployez le modèle en production