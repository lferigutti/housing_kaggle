# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 10:19:56 2020

@author: leo_f

We are gonna practise with Housing Advance Techniques.

The idea of this script is to practise with linear models only, and practise features interactions,
feature interactions, and see how different and better or worse are the linear models using those techniques.
Also we are gonna add information during data cleaning using the aproach of the book introduction to Machine Learning
with Python
 

1st approach - I normalized and did some polynomial interactions, and feature selection using random forest.
Results: much worse than base results. Polynomial interaction make a huge matrix, so I decided not to use it.

When I didn't normalize SalePrice, the performance was slightly better


"""
# Libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split



# Import the data

data_train = pd.read_csv('C:\\Users\\leo_f\\Documents\\ML\\Kaggle Competitions\\Housing\\train.csv')
data_test = pd.read_csv('C:\\Users\\leo_f\\Documents\\ML\\Kaggle Competitions\\Housing\\test.csv')

data = pd.concat([data_train,data_test], ignore_index=True)

sale_price_total = data.loc[:,'SalePrice']



# Basic Data cleaning
#print(data.info())

info_null = pd.DataFrame(data.isnull().sum().sort_values(ascending=False), columns=['null_items'])


info_null['Percentage'] = info_null['null_items']/2919

# Decision: we are gonna eliminate the following features(more than 15% of missing values):
# PoolQC, MiscFeature,Alley,Fence,FireplaceQu, LotFrontage)
# We also gonna eliminate (due to pre-data exploration): GarageCars, 1stFlrSF

data_clean = data.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','GarageCars','1stFlrSF'], axis=1)


# We are going to complete the missing data with simple imputation, using the most frequency value
# We can also use knn for this, and also pipeline as well

data_clean.drop('SalePrice', axis = 1, inplace =True)

imput_freq = SimpleImputer(strategy='most_frequent')
imput_freq.fit(data_clean)
data_clean_imputed = pd.DataFrame(imput_freq.transform(data_clean), columns= data_clean.columns)

# Recover all the types of the original DF
for col in data_clean_imputed.columns:
    data_clean_imputed[col]=data_clean_imputed[col].astype(data_clean[col].dtypes)


# Basic pre-proccesing (just Getdummies)

data_clean_imputed_dummies = pd.get_dummies(data_clean_imputed)



# Basic Linear models (base results)

# Prepear the X and y

X = data_clean_imputed_dummies.loc[:1459,:]
y = sale_price_total.loc[:1459]
X_test_final = data_clean_imputed_dummies.loc[1460:,:]

# Basic results
 
linre = LinearRegression()
ridge = RidgeCV()
lasso = LassoCV()

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

# Linear Regression
linre.fit(X_train,y_train)
y_pred_linre = linre.predict(X_test)
base_score_linre = np.sqrt(metrics.mean_squared_error(y_test,y_pred_linre))
print('Linear Regression Score:',base_score_linre)

# Ridge Regression
ridge.fit(X_train,y_train)
y_pred_ridge = ridge.predict(X_test)
base_score_ridge = np.sqrt(metrics.mean_squared_error(y_test,y_pred_ridge))
print('Ridge Regression Score:',base_score_ridge)

# Lasso Regression
lasso.fit(X_train,y_train)
y_pred_lasso = lasso.predict(X_test)
base_score_lasso = np.sqrt(metrics.mean_squared_error(y_test,y_pred_lasso))
print('Lasso Regression Score:', base_score_lasso)

# Simple submision   Kaggle Score = 0.15
#ridge.fit(X,y)
#y_pred_final= ridge.predict(X_test_final)

#submision_1 = pd.DataFrame({'ID': data.loc[1460:,'Id'],'SalePrice':y_pred_final})
#submision_1.to_csv('housing_first_sub.csv', index=False)

## We are gonna do some preproccesing. Let's normalize some variables. 

# Normalize: SalePrice, GarageArea, TotalBsmSF

y_norm = np.log(y + 1)
data_clean_imputed_dummies_norm = data_clean_imputed_dummies.copy()
data_clean_imputed_dummies_norm['GarageAreaNorm'] = np.log(data_clean_imputed_dummies.loc[:,'GarageArea']+1)
data_clean_imputed_dummies_norm['TotalBsmtSFNorm'] = np.log(data_clean_imputed_dummies.loc[:,'TotalBsmtSF']+1)

X_norm = data_clean_imputed_dummies_norm.loc[:1459,:]
#X_test_final_norm = data_clean_imputed_dummies_norm.loc[1460:,:]


# Feature Selection (choose only 100 Features)
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

select = SelectFromModel(RandomForestRegressor(random_state=1),threshold='median')
select.fit(X_norm, y)
X_norm_select = select.transform(X_norm)
#X_test_final_norm = select.transform(X_test_final_norm)


# Model building

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_norm_select,y, random_state=0)

# The result is waaay worse
ridge.fit(X_train_1,y_train_1)
y_pred_ridge_norm = ridge.predict(X_test_1)
#y_pred_ridge_norm = np.exp(y_pred_ridge_log)-1
#y_test_1_norm = np.exp(y_test_1)-1
norm_select_score_ridge = np.sqrt(metrics.mean_squared_error(y_test_1,y_pred_ridge_norm))
print('Ridge Regression Score:',norm_select_score_ridge)


# Feature Interaction
#poly = PolynomialFeatures(degree=2,include_bias=False).fit(X_norm)
#X_norm_poly = poly.transform(X_norm)
#X_test_final_poly = poly.fit_transform(X_test_final_norm)







