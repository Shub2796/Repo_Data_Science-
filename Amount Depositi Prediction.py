# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 20:33:10 2020

@author: ss186249
"""
#loading libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pylab
import statistics as st
from scipy import stats
import scipy.stats

#Import data set
data=pd.read_csv("path/train.csv")
data.columns

#to check uniqu identifier
for val in data:
    print(val, " ", data[val].unique().shape)
    
#Removing common values and identifier
data=data.isnull().sum()
cols=["id","location.Code","headquarter"]
data=data.drop(cols,axis=1) 

#Extra Analysis- We have removed feew columns whoch have multiple levels   
cols1=["date_of_establishment","location","loc.details","state","location.Code"]
data=data.drop(cols,axis=1)  

# We have removed the identifier and keep all relevent data in df dataframe 
final=['deposit_amount_2011','deposit_amount_2012','deposit_amount_2013','deposit_amount_2014','deposit_amount_2015','deposit_amount_2016','deposit_amount_2017']
df=data[final]

#Checked the null for overall data
data.isnull().any()
for val in df:
    print(val, " ", (df[val].isnull().sum()/df.shape[0])*100)

#Number of rows affected by NA
data.isnull().sum()

#No of rows getting affctected by removing na's 
 no_of_rows = df[df.isna().sum(axis=1)>=1].shape[0]
 no_of_rows
 

#In case want to impute Na's : Imputation according to data type
                    def imputenull(df):
                        for col in df.columns:
                            if df[col].dtypes == 'int64' or df[col].dtypes == 'float64':
                                df[col].fillna((df[col].mean()), inplace=True)
                            else:
                                df[col].fillna(df[col].value_counts().index[0], inplace=True)


imputenull(df)

#Now checking on the dataset after imputing the columns
df.isnull().sum()

#For skewness we check the Bar plot and histogram
df.plot(kind='box',subplots=True,sharex=False, sharey=False)
df.plot(kind='hist',subplots=True,sharex=False, sharey=False)

#Checking skewness   
#df2=df.copy()               
df.skew()

#Substituting target varible in new varible
tar_var = df['deposit_amount_2017']

#Z score
z=np.abs(stats.zscore(df.tar_var))
print(z)
threshold=3
print(np.where(z>3))
x=np.where(z>3)
x
df.iloc[np.where(z>1.75)[:1]]

#IQR
Q1 = tar_var.quantile(0.25)
Q3 = tar_var.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
x = print(tar_var < (Q1 - 1.5 * IQR)) or (tar_var > (Q3 + 1.5 * IQR))
tar_var[x]
           
# Transformation                       
df2["deposit_amount_2017"] = df2["deposit_amount_2017"]**(1/3)
                        df2.deposit_amount_2017.plot(kind = "hist")
                        df2.deposit_amount_2017.skew() 

# Histograms : by removing density plot from top of it.
            sns.distplot(tar_var, kde=False, rug=False);

# Training & Test split of data
df.drop(tar_var, axis=1, inplace=True)

x_train=df
y_train=tar_var

#Import the test data

Test_data=pd.read_csv("Path/test.csv")
Test_data.columns

#to check uniqu identifier
for val in Test_data:
    print(val, " ", Test_data[val].unique().shape)
    
#Removing common values and identifier
Test_data=data.isnull().sum()
cols=["id","location.Code","headquarter"]
Test_data=Test_data.drop(cols,axis=1) 

#Extra Analysis- We have removed feew columns whoch have multiple levels   
cols1=["date_of_establishment","location","loc.details","state","location.Code"]
Test_data=Test_data.drop(cols,axis=1)  

# We have removed the identifier and keep all relevent data in df dataframe 
final=['deposit_amount_2011','deposit_amount_2012','deposit_amount_2013','deposit_amount_2014','deposit_amount_2015','deposit_amount_2016','deposit_amount_2017']
Test_data_df=data[final]

#Checked the null for overall data
data.isnull().any()
for val in Test_data_df:
    print(val, " ", (Test_data_df[val].isnull().sum()/df.shape[0])*100)

#No of rows getting affctected by removing na's 
 no_of_rows = Test_data_df[df.isna().sum(axis=1)>=1].shape[0]
 no_of_rows
 

#In case want to impute Na's : Imputation according to data type
                    def imputenull(Test_data_df):
                        for col in Test_data_df.columns:
                            if Test_data_df[col].dtypes == 'int64' or Test_data_df[col].dtypes == 'float64':
                                Test_data_df[col].fillna((Test_data_df[col].mean()), inplace=True)
                            else:
                                Test_data_df[col].fillna(df[col].value_counts().index[0], inplace=True)


imputenull(Test_data_df)

x_test=Test_data_df
y_test= #use target varible for test data

# Implementing model
from sklearn import linear_model as lm
model = lm.LinearRegression()
result = model.fit(x_train, y_train)

#Printing the coefficient
print(model.intercept_)
print(model.coef_)
model.fit
    
# predicting the values.
predictions = model.predict(x_test)
predictions
#Transformation into original form
predictions = predictions**3         
                     
# Getting the model eveluation from predicted values
            #from sklearn.metrics import mean_squared_error, r2_score
# model evaluation
            #rmse = mean_squared_error(y_test, predictions)
            #r2 = r2_score(y_test, predictions)