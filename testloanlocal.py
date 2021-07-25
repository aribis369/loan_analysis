#!/usr/bin/env python
# coding: utf-8


import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn import tree
#from sklearn.metrics import classification_report
#from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from pickle import load, dump




'''

loan_amnt, emp_length, annual_inc, delinq_2yrs, inq_last_6mths, mths_since_last_delinq, mths_since_last_record, open_acc, pub_rec, revol_bal, revol_util, total_acc, purpose, year   

'''


'''
print(debt_consolidation)#2
print(car)#1
print(credit_card)#0
print(educational)#2
print(home_improvement)#1
print(house)#1
print(major_purchase)#0
print(medical)#2
print(moving)#2
print(other)#2
print(renewable_energy)#2
print(small_business)#2
print(vacation)#1
print(wedding)#0

'''




# scikit-learn==0.22.2.post1




X_one = [2.2000e+04, 6.0000e+00, 7.0000e+04, 1.0000e+00, 0.0000e+00, 1.0000e+01,
  1.3000e+02, 1.1000e+01, 0.0000e+00, 3.8928e+04, 8.5000e-01, 3.6000e+01,
  2.0000e+00, 1.9830e+03]

X_one_ = np.array(X_one).reshape(1, -1)
print(X_one_)

model_ = load(open('model.pkl', 'rb'))
scaler_ = load(open('scaler.pkl', 'rb'))
pca_ = load(open('pca.pkl', 'rb'))



X_one_ = scaler_.transform(X_one_)
X_one_ = pca_.transform(X_one_)
prediction = model_.predict(X_one_)

print(prediction[0])





