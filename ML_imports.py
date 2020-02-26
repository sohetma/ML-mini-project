# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:49:37 2018

@author: user
"""

# Importing the libraries
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import seaborn
seaborn.set(context="paper", font="monospace")
from sklearn.cross_validation import cross_val_score
import itertools as it
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.learning_curve import validation_curve
#from sklearn.model_selection import validation_curve
#from yellowbrick.model_selection import ValidationCurve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC