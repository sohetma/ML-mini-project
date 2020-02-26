# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:57:23 2018

@author: Sohet
"""
from ML_functions import *

def ml_linear_reg(X1,y1,X1_train, X1_test, y1_train,y1_test):
  
  # Fitting Multiple Linear Regression to the Training set
  regressor = LinearRegression()
  model = regressor.fit(X1_train, y1_train)
  
  # Predicting the Test set results
  y_pred = regressor.predict(X1_test)
  
  return X1_train, X1_test, y1_train,y1_test,model,regressor,y_pred
  
  
def ml_linear_reg_opt(X1,y1,ts=0.2):
  SL = 0.05
  l,c = X1.shape
  X_opt = X1[:, [0, 1, 2, 3, 4, 5, 6, 7,]]
  X_modeled = backwardElimination(X_opt,y1,SL,l,c)
  X1_opt_train, X1_opt_test, y1_opt_train, y1_opt_test = train_test_split(X_modeled, y1, test_size = ts, random_state = 0)
  X1_opt_train, X1_opt_test, y1_opt_train, y1_opt_test,model_opt,regressor_opt,y_pred_opt = ml_linear_reg(X_modeled,y1,X1_opt_train, X1_opt_test, y1_opt_train, y1_opt_test)
  return model_opt, regressor_opt, y_pred_opt