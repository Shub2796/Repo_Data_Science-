# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#loading libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report as cr, confusion_matrix as cm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pylab
import statistics as st
from scipy import stats
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report as cr, confusion_matrix as cm
from sklearn.feature_selection import f_classif as fs
from sklearn.feature_selection import RFE
from sklearn import preprocessing # for label encoding
from io import StringIO
# pip install pydotplus
#import pydotplus

#Import data set

cancer = pd.read_csv('E:/My Project Data science/Halthcare/Breast Cancer Wisconsin (Diagnostic) Data Set/data.csv')

df=cancer.copy()

df.columns

col=['id','Unnamed: 32']

df.drop(col,axis=1,inplace=True)

y = df.diagnosis
df.drop('diagnosis',axis=1,inplace=True)

df.isnull().sum()


corrmat=df.corr()
fig,ax=plt.subplots()
fig.set_size_inches(11,11)
sns.heatmap(corrmat)

def checkcorrelation(df,threshold):
                            col_corr=set()
                            cor_matrix=df.corr()
                            for i in range(len(cor_matrix.columns)):
                                for j in range(i):
                                    if abs(cor_matrix.iloc[i,j])>threshold:
                                        colname=cor_matrix.columns[i]
                                        col_corr.add(colname)
                            return col_corr

                        checkcorrelation(df,0.9)

col1=['area_mean','area_se','area_worst',
 'concave points_mean',
 'concave points_worst',
 'perimeter_mean',
 'perimeter_se',
 'perimeter_worst',
 'radius_worst',
 'texture_worst']

df.drop(col1,axis=1,inplace=True)

#Dummeis 

y1 = pd.get_dummies(y, drop_first=True)

#Spliting now.
            x_train,x_test,y_train,y_test = train_test_split(df, y1, random_state = 10,test_size = 0.3)


            # from sklearn.linear_model import LogisticRegression
            # model = LogisticRegression()
            # model.fit(x_train, y_train)

# create logistic regression object
            from sklearn import linear_model as lm
            reg = lm.LogisticRegression()

# train the model using the training sets
            reg.fit(x_train, y_train)

# making predictions on the testing set
            y_pred = reg.predict(x_test)

import sklearn.metrics as metrics
# comparing actual response values (y_test) with predicted response values (y_pred)
            print("Logistic Regression model accuracy(in %):",
            metrics.accuracy_score(y_test, y_pred) * 100)
            print(metrics.confusion_matrix(y_test, y_pred))

# save confusion matrix and slice into four pieces---- deep diving into confusion matrix
            confusion = metrics.confusion_matrix(y_test, y_pred)
            print(confusion)
            #[row, column]
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))                 #Accuracy by calculation
            # Confusion maytrix

classification_error = (FP + FN) / float(TP + TN + FP + FN) #Error
print(classification_error*100)
            
sensitivity = TP / float(FN + TP)
print(sensitivity)
            

specificity = TN / (TN + FP)
print(specificity)

false_positive_rate = FP / float(TN + FP)
print(false_positive_rate)
print(1 - specificity)

precision = TP / float(TP + FP)
print(precision)

      

            """Receiver Operating Characteristic (ROC)"""
            # IMPORTANT: first argument is true values, second argument is predicted values
            # roc_curve returns 3 objects fpr, tpr, thresholds
            # fpr: false positive rate
            # tpr: true positive rate
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            plt.plot(fpr, tpr)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.rcParams['font.size'] = 12
            plt.title('ROC curve for Bank Lending classifier')
            plt.xlabel('False Positive Rate (1 - Specificity)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.grid(True)


            """AUC - Area under Curve"""

            # AUC is the percentage of the ROC plot that is underneath the curve:
            # IMPORTANT: first argument is true values, second argument is predicted probabilities
            print(metrics.roc_auc_score(y_test, y_pred))


            # F1 Score FORMULA
            F1 = 2 * (precision * sensitivity) / (precision + sensitivity)