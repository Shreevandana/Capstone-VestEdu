#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import random
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# # Reading Data In

# In[ ]:


data_doe = pd.read_excel("NewDOEData_corrected.xlsx") #reading DOE data


# # Subsetting the data by level and major 

# In[ ]:


undergrad = data_doe[data_doe['CREDDESC'] == "Bachelors Degree"]
grad = data_doe[data_doe['CREDDESC'] == "Master's Degree"]

data_level = undergrad.append(grad) # Subsetting the data by level

arts = data_level[data_level['MAJOR'] == "Fine Arts and Humanities"]
business = data_level[data_level['MAJOR'] == "Business"]
stem = data_level[data_level['MAJOR'] == "Science, Tech and Engineering"]

data_subset = arts.append(business).append(stem) # Subsetting the data by major


# # Converting categorical features to one hot vectors

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

lb_make = LabelEncoder()
level = lb_make.fit_transform(data_subset[["CREDDESC"]])  #converting level to categorical
rank = lb_make.fit_transform(data_subset[["RANK"]]) #converting rank to categorical
field = lb_make.fit_transform(data_subset[["MAJOR"]]) #converting major to categorical

features = pd.DataFrame()  #creating a new dataframe to feed into the model
features['LEVEL'] = level
features['RANK'] = rank
features['FIELD'] = field

lb = LabelBinarizer()
label = lb.fit_transform(data_subset['LOAN_STATUS_MDTI']) #converting loan status to binary


# # Fit a grid search XG boost model and get values for AUC and Accuracy

# In[ ]:


auc = []          
accuracy = []

model = XGBClassifier()                   #load the XGBoost classifier
parameters = {                            #hyperparamter tuning for model
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05],
}

grid_search = GridSearchCV(              #intializing the model
    estimator=model,
    param_grid=parameters,
    n_jobs = 10,
    verbose=True,
    cv = 10
)

train_auc = []
train_accu = []
test_auc = []
test_accu = []


X_train, X_test, y_train, y_test = train_test_split(features, label,    #split the data into 80-20 for train-test
                                                    test_size=0.2)
    
grid_search.fit(X_train,  y_train)       #fittig the model

#Repeated test train to get value of metrics 
for iter in range(10):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)
    
    #getting predictions for test and train
    pred = grid_search.best_estimator_.predict(X_test)
    pred_train = grid_search.best_estimator_.predict(X_train)
    
    #calculating AUC and accuracy for test
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred,pos_label=1)
    test_auc.append(metrics.auc(fpr, tpr))
    test_accu.append(metrics.accuracy_score(y_test,pred))
    
    #calculating AUC and accuracy for train
    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_train, pred_train,pos_label=1)
    train_auc.append(metrics.auc(fpr1, tpr1))
    train_accu.append(metrics.accuracy_score(y_train,pred_train))
    
print("Mean Value of Train AUC over ten runs", np.mean(train_auc), "\n")
print("Mean Value of Test AUC over ten runs", np.mean(test_auc), "\n")
print("Mean Value of Train Accuracy over ten runs", np.mean(train_accu)*100, "% \n")
print("Mean Value of Test Accuracy over ten runs", np.mean(test_accu)*100, "% \n")


# # Printing Feature Importances

# In[ ]:


print("The feature importance for Level is", grid_search.best_estimator_.feature_importances_[0]*100, "% \n")
print("The feature importance for Rank is", grid_search.best_estimator_.feature_importances_[1]*100, "% \n")
print("The feature importance for Field is", grid_search.best_estimator_.feature_importances_[2]*100, "% \n")

