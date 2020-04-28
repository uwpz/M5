# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:01:38 2017

@author: Uwe
"""

#######################################################################################################################-
# Libraries + Parallel Processing Start ----
#######################################################################################################################-

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats.mstats import winsorize
import dill
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_validate, RepeatedKFold, learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgbm

sns.set(style="whitegrid")

#######################################################################################################################-
# Parameters ----
#######################################################################################################################-

dataloc = "./data/"
plotloc = "./output/"




#######################################################################################################################-
# My Functions ----
#######################################################################################################################-

print("run init")
def setdiff(a, b):
    return [x for x in a if x not in set(b)]
    