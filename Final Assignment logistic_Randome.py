# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:31:18 2020

@author: ss186249
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:24:22 2020

@author: Zankar
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
import numpy as np
from sklearn.feature_selection import RFE
from sklearn import preprocessing # for label encoding
from io import StringIO
import pydotplus

# Reading Excel File
data=pd.read_excel("C:/Users/ss186249/Documents/Bank_data.xlsx")

data1=data.copy()
# Basic Checks
data.shape

#for loop to check unique value
for val in data:
    print(val, " ", data[val].unique().shape)
    
# Removing columns having more than 80% Null Values
a=data.isnull().sum()
cols=["id","member_id","desc","mths_since_last_record","mths_since_last_major_derog","annual_inc_joint","dti_joint","verification_status_joint","open_acc_6m","open_il_12m","open_il_24m","mths_since_rcnt_il","total_bal_il","il_util","open_rv_12m","open_rv_24m","max_bal_bc","all_util","inq_fi","total_cu_tl","inq_last_12m","zip_code","policy_code","open_il_6m"]
data=data.drop(cols,axis=1)
data.shape

# Removing columns having strings
cols1=["emp_title","purpose","title"]
data=data.drop(cols1,axis=1)
data.shape

# Removing Columns having more than 80% 0's
cols2=["delinq_2yrs","pub_rec","total_rec_late_fee","recoveries","collection_recovery_fee","collections_12_mths_ex_med","acc_now_delinq","tot_coll_amt"]
data=data.drop(cols2,axis=1)
data.shape


# Removing columns having Biasness and 90% Singularities
cols3=["pymnt_plan","application_type"]
data=data.drop(cols3,axis=1)
data.shape


# Extra Analysis
#
data=data.drop('addr_state',axis=1)  
data=data.drop('sub_grade',axis=1)
data.shape

# Date Columns
cols5=["issue_d","earliest_cr_line","last_pymnt_d","next_pymnt_d","last_credit_pull_d"]
data=data.drop(cols5,axis=1)
data.shape  

# No of rows getting affctected by removing na's
                    no_of_rows = data[data.isna().sum(axis=1)>=1].shape[0]
                # % of rows getting affcted by removing Na's from column.
                    print((no_of_rows/data.shape[0])*100)

#  In case want to impute Na's : Imputation according to data type
    def imputenull(data):
        for col in data.columns:
            if data[col].dtypes == 'int64' or data[col].dtypes == 'float64':
                                data[col].fillna((data[col].mean()), inplace=True)
            else:
                data[col].fillna(data[col].value_counts().index[0], inplace=True)


imputenull(data)

data.isnull().sum()
 
 

#segregate numeric and categorical data
numcols=data.select_dtypes(exclude=['object']).columns     
factcols=data.select_dtypes(include=['object']).columns 


 
# Check for multicollinearity using HeatMap
cor=data[numcols[0:len(numcols)-1]].corr()     
cor=np.tril(cor,k=1)
sns.heatmap(cor,
            xticklabels=numcols[0:len(numcols)-1],
            yticklabels=numcols[0:len(numcols)-1],
            annot=True, linewidths=1,
            vmin=0,vmax=1,
            square=True)
                        
#'''Below is function to check for correlation and remove columns on basis of that'''
                        def checkcorrelation(dataset,threshold):
                            col_corr=set()
                            cor_matrix=dataset.corr()
                            for i in range(len(cor_matrix.columns)):
                                for j in range(i):
                                    if abs(cor_matrix.iloc[i,j])>threshold:
                                        colname=cor_matrix.columns[i]
                                        col_corr.add(colname)
                            return col_corr

                        checkcorrelation(data,0.6)

# Varibles which are highly corealted.
col6=['funded_amnt' ,'funded_amnt_inv','installment','annual_inc','revol_bal','out_prncp_inv','total_pymnt_inv','total_rec_prncp','total_rec_int','last_pymnt_amnt','tot_cur_bal','total_rev_hi_lim' ]
data=data.drop(col6,axis=1)         
data.shape      
           
data.columns
df=data.copy()

# Removing Spaces from Categorical Columns
for f in factcols:
    print("Factor variable = ", f)
    print(data[f].unique())
    print("***")

data.emp_length.value_counts()

def removeSpaces(x):
    x = x.strip()
    return(x)
for f in factcols:
    print(f)
    data[f] = data[f].apply(removeSpaces)


# 2) level Merging
factv=[]; oldv=[]; newv=[]

factv.append("verification_status")
oldv.append(['Verified','Source Verified'])
newv.append('Verified')

factv.append("home_ownership")
oldv.append(['ANY', 'OTHER','NONE'])
newv.append("OTHER")

factv.append("grade")
oldv.append(['F','G'])
newv.append("G")

print(factv)
print(oldv)
print(newv)

numcols
factcols
# parameters:
# df = dataframe (pandas)
# factv: factor variable
# oldv: list of old values to be replaced
# newv: new value to be replaced with

def replaceFactorValues(data,factv,oldv,newv):
    if (len(factv) == len(oldv) == len(newv) ):
        
        for i in range(0,len(factv)):
            data[factv[i]] [data[factv[i]].isin(oldv[i])] = newv[i]
            
            # internally, the above code translates to ...
            # census.workclass [census.workclass.isin(['','',''])] = 'new'
        msg = "SUCCESS: 1 Updates done"
    else:
        msg = "ERRCODE:-1  Inconsistent length in the input lists"
    
    return(msg)
    
#----------------------------or-------------------------
    
def replaceFactorValues(data,factv,oldv,newv):
    if (len(factv) == len(oldv) == len(newv) ):
        
        for i in range(0,len(factv)):
            data[factv[i]] [data[factv[i]].isin(oldv[i])] = newv[i]
            
            # internally, the above code translates to ...
            # census.workclass [census.workclass.isin(['','',''])] = 'new'
        msg = "SUCCESS: 1 Updates done"
    else:
        msg = "ERRCODE:-1  Inconsistent length in the input lists"
    
    return(msg)
 

ret = replaceFactorValues(data,factv,oldv,newv)
print(ret)

# Now Label Encoding instead of creating Dummies

def convertFactorsToNum(data,factcols):
    le = preprocessing.LabelEncoder()
    
    for f in factcols:
        data[f] = le.fit_transform(data[f])
    
    return(1)

ret = convertFactorsToNum(data,factcols)
print(ret)

data.columns
data.dtypes  
#drop columns which are highly co-related



# Rescaling Numerical Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
for val in data:
    if(data[val].dtypes in ['int64', 'float64']):
        data[[val]] = scaler.fit_transform(data[[val]])




#checking co relation after level mearging
    
cor=data[data[0:len(data)-1]].corr()     
cor=np.tril(cor,k=1)
sns.heatmap(cor,
            xticklabels=numcols[0:len(numcols)-1],
            yticklabels=numcols[0:len(numcols)-1],
            annot=True, linewidths=1,
            vmin=0,vmax=1,
            square=True)
                        
#'''Below is function to check for correlation and remove columns on basis of that'''
                        def checkcorrelation(dataset,threshold):
                            col_corr=set()
                            cor_matrix=dataset.corr()
                            for i in range(len(cor_matrix.columns)):
                                for j in range(i):
                                    if abs(cor_matrix.iloc[i,j])>threshold:
                                        colname=cor_matrix.columns[i]
                                        col_corr.add(colname)
                            return col_corr

                        checkcorrelation(data,0.6)
#copy of processed data 
df3=data.copy()



col=['grade','out_prncp', 'total_acc']
data.drop(col,axis=1,inplace=True)
data.shape
data.columns

#sampling the data in balanced form

tarwar_0 = data[data["default_ind"]==0]
tarwar_1 = data[data["default_ind"]==1]
subset_of_default_ind_0 = tarwar_0.sample(n=40000)
subset_of_default_ind_1 = tarwar_1.sample(n=40000)
data = pd.concat([subset_of_default_ind_1, subset_of_default_ind_0])

# Now split the data into Train and test for Model Building
totalcols=len(data.columns)
train,test=train_test_split(data,test_size=0.3)   

#--------------------------------------------------------------------------
#this below code for validateion 
#tarwar_0 = train[train["default_ind"]==0]
#tarwar_1 = train[train["default_ind"]==1]
#subset_of_default_ind_0 = tarwar_0.sample(n=30000)
#subset_of_default_ind_1 = tarwar_1.sample(n=30000)
#train = pd.concat([subset_of_default_ind_1, subset_of_default_ind_0])
#--------------------------------------------------------------------------


trainx=train.iloc[:,0:totalcols-1]
trainy=train.iloc[:,13]
print('{},{}'.format(trainx.shape,trainy.shape))            
            
testx=test.iloc[:,0:totalcols-1]
testy=test.iloc[:,13]            
print('{} {}'.format(testx.shape,testy.shape))  

testy.value_counts()        

# Now Build the Logistic Regression Model
m1=sm.Logit(trainy,trainx).fit()
# Now summarize the model
m1.summary()

# prediction on the test data
p2=m1.predict(testx)

p2[0:6]

# count of y-variables in test set
testy.value_counts()

# start with the initial cutoff as 0.5
# converting probabilities into classes
predY = p2.copy()

predY[predY > 0.5] = 1
predY[predY < 0.5] = 0
predY.value_counts()

# confusion matrix
cm(list(testy),list(predY))
# cm(testy,predY)
# testy.value_counts()

# classification report
print(cr(testy,predY))

# AUC and ROC curve
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(testy, predY)
roc_auc = metrics.auc(fpr,tpr)

# comparing actual response values (testy) with predicted response values (predY)
            print("Logistic Regression model accuracy(in %):",
            metrics.accuracy_score(testy, predY) * 100)
            print(metrics.confusion_matrix(testy, predY))
pred
# save confusion matrix and slice into four pieces---- deep diving into confusion matrix
            confusion = metrics.confusion_matrix(testy, predY)
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
            fpr, tpr, thresholds = metrics.roc_curve(testy, predY)
            plt.plot(fpr, tpr)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.rcParams['font.size'] = 12
            plt.title('ROC curve for bank lending classifier')
            plt.xlabel('False Positive Rate (1 - Specificity)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.grid(True)


            """AUC - Area under Curve"""

            # AUC is the percentage of the ROC plot that is underneath the curve:
            # IMPORTANT: first argument is true values, second argument is predicted probabilities
            print(metrics.roc_auc_score(testy, predY))


            # F1 Score FORMULA
            F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
            F1

# calculate cross-validated AUC only for logistics
        from sklearn.model_selection import cross_val_score as cv
        cross_val_score(m1, trainx, trainy, cv=1, scoring='roc_auc').mean()





#------------------------------------------------------------------------------------------------------------------

# Same data using randome forest Algorithm   

# Applying randomne forest df3 
        
df3=data.copy()    

Y=df3['default_ind']
Y.shape
df3.drop("default_ind", axis=1, inplace=True)


df3.shape
df3.columns  


x_train,x_test,y_train,y_test = train_test_split(df3, Y, random_state = 20,test_size = 0.3)

print("TRAIN X {}, TRAIN Y {},TEST X {}, TEST Y{}".format(x_train.shape,y_train.shape,x_test.shape,y_test.shape))

#Concat train x and y for validation 
c1 = pd.concat([x_train,y_train],axis=1)

#sampling the data in balanced form

tarwar_0 = c1[c1["default_ind"]==0]
tarwar_1 = c1[c1["default_ind"]==1]
subset_of_default_ind_0 = tarwar_0.sample(n=30000)
subset_of_default_ind_1 = tarwar_1.sample(n=30000)
c1 = pd.concat([subset_of_default_ind_1, subset_of_default_ind_0])

y_train=c1['default_ind']
x_train=c1.drop("default_ind", axis=1)

#x_train=c1

# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestClassifier
# create regressor object
regressor = RandomForestClassifier(n_estimators = 100, random_state = 0)
# fit the regressor with x and y data
regressor.fit(x_train,y_train)
# Making prediction.

m1=RandomForestClassifier(n_estimators = 500, max_depth=8, max_features=3).fit(x_train,y_train)


y_pred = regressor.predict(x_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
print(metrics.confusion_matrix(y_test, y_pred))

# save confusion matrix and slice into four pieces---- deep diving into confusion matrix

confusion = metrics.confusion_matrix(y_test, y_pred)
print(confusion)
# [row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))  # Accuracy by calculation

classification_error = (FP + FN) / float(TP + TN + FP + FN)  # Error
print(classification_error * 100)


sensitivity = TP / float(FN + TP)
print(sensitivity)


specificity = TN / (TN + FP)
print(specificity)

false_positive_rate = FP / float(TN + FP)
print(false_positive_rate)
print(1 - specificity)

precision = TP / float(TP + FP)
print(precision)
print(metrics.precision_score(y_test, y_pred))


from sklearn.metrics import classification_report as cr

print(cr(y_test,y_pred))

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
plt.title('ROC curve for bank lending classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


"""AUC - Area under Curve"""

# AUC is the percentage of the ROC plot that is underneath the curve:
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, y_pred))

# F1 Score FORMULA
F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

F1

# Variable importance in RandomForest
importances = regressor.feature_importances_
indices = np.argsort(importances)
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), df3.columns)
plt.xlabel('Relative Importance')


# plotting confusion matrix in case of multiclass classifier
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

#-------------------------------------------------------------------------
# SVM Code
# Firstly we find the COST PARAMETER using cross validation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report as cr,confusion_matrix as cm,accuracy_score
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import neighbors
from sklearn import svm
from sklearn import preprocessing
import statsmodels.api as sts
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from io import StringIO
import pydotplus

 list_C=np.logspace(-3,2,6)
 list_C


 cvs=[]
 
 1.e-03
 1.e-02
 1.e-01
 1.e+00
 1.e+01
 1.e+02
 
 for c in list_C:
     model=svm.SVC(kernel="linear",C=c)
     score1=cross_val_score(model,trainx,trainy,cv=5,scoring="accuracy")
     cvs.append(score1.mean())
     
 print(cvs)    

opt_c=list_C[cvs.index(max(cvs))]
print("Optimal Cost Parameter is: ",opt_c)

# Building Models using Cost Parameter

# 1. Kernel = LINEAR

m_linear=svm.SVC(kernel="linear",C=opt_c).fit(trainx,trainy)
p_linear=m_linear.predict(testx)
 # Confusion Matrix and Classification report for Linear Kernel
  print(cm(list(testy),list(p_linear)))
  testy.value_counts()
  print(cr(testy,p_linear))
  print("Accuracy Score of Linear Kernel is: ",accuracy_score(testy,p_linear)*100)


# 2. Kernel= SVC LINEAR
  
m_LSVC=svm.LinearSVC(C=opt_c).fit(trainx,trainy)
p_LSVC=m_LSVC.predict(testx)
   # Confusion Matrix and Classification Report of SVC Linear Kernel
   print(cm(list(testy),list(p_LSVC)))
   print(cr(testy,p_LSVC))

# 3. Kernel = RBF(Radial Basis Kernel Function)
# For RBF we need to find Gamma Parameter also
   
list_C=np.logspace(-3,2,6)
list_gamma=np.logspace(-2.55,0,6)
list_gamma
cvs1=[]
cvg1=[]

for c in list_C:
    for g in list_gamma:
        model=svm.SVC(kernel="rbf",C=c,gamma=g)
        score=cross_val_score(model,trainx,trainy,cv=5,scoring="accuracy")
        cvs1.append(score.mean())
        cvg1.append(str(c)+":"+str(g))        
cvs1
cvg1
print(max(cvs1))
print(max(cvg1))

# Get the corresponding Index position for C and Gamma parameter
c_gamma=cvg1[cvs1.index(max(cvs1))]
c_gamma
opt_C=float(c_gamma.split(":")[0])
opt_G=float(c_gamma.split(":")[1])
opt_C
opt_G
