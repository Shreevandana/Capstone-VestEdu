#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV


# # Reading Data In

# In[ ]:


doe = pd.read_excel("NewDOEData_corrected.xlsx")  #reading DOE data
lc = pd.read_csv("LC_LoanDataQ2_19.csv", encoding='latin-1') #reading lending club data
lc = lc[lc['purpose'] == "educational"]  #subsetting lending club data that have educational loans


# # Removing privacy supressed entries from DOE

# In[ ]:


doe = doe[doe['MD_EARN_WNE'] != "PrivacySuppressed"] #removing privacy supressed income values
doe = doe[doe['DEBTMEAN'] != "PrivacySuppressed"] #removing privacy supressed debt values


# # Preprocessing interest rate column from LC

# In[ ]:


y = []
for r in lc['int_rate']:
    r = r.replace("%", "")   #Removing % sign
    r = float(r)            #Converting the string value into a float 
    y.append(r)

lc['int_rate'] = y          #Sending the processed interest rate back to the lc dataframe


# # Subsetting the income and debt values from DOE

# In[ ]:


X_doe = doe[['DEBTMEAN','MD_EARN_WNE']]  #Subsetting the income and debt values
X_doe.columns = ['debt', 'income']       # Renaming them to match lc


# # Subsetting the income ,debt and loan status values from LC

# In[ ]:


int_df = pd.DataFrame()                   #creating a new data frame to train
int_df['debt'] = lc['funded_amnt']        #Get debt from lc
int_df['income'] = lc['annual_inc']       #Get income from lc
int_df['y'] = lc['int_rate']              #Get interest rate from lc
print("Original", int_df.shape)
int_df = int_df.dropna(how = "all")        #Checking if there are any null values in LC
print("New", int_df.shape)


# # Fit a grid search xg boost model

# In[ ]:


x = int_df[['debt', 'income']]     #creating a dataframe with debt and income values from lc 

X_train, X_test, y_train, y_test = train_test_split(x, y,              #split the data into 80-20 for train-test
                                                    test_size = 0.2)


model = XGBRegressor()            #load the XGBoost regressor 

parameters = {                    #hyperparamter tuning for model
    'max_depth': range (2, 10, 1),          
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05],
}
grid_search = GridSearchCV(       #intializing the model
    estimator=model,
    param_grid=parameters,
    n_jobs = 10,
    verbose=True,
    cv = 10
)

grid_search.fit(X_train, y_train) #fittig the model


# # Predict the interest rate for DOE data

# In[ ]:


interest = []

#Repeated test train to get value of metrics 
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    int_test = grid_search.best_estimator_.predict(X_test)
    interest.append(mse(y_test,int_test))
    
#Predicting the interest rate on DOE 
int_doe = grid_search.best_estimator_.predict(X_doe)


# # DOE Interest Rate distribution

# In[ ]:


print("The mean of the predicted distribution is", np.mean(int_doe))                     #Getting mean of the predicted interest rate
print("The maximum value of the predicted distribution is", np.max(int_doe))             #Getting max of the predicted interest rate
print("The minimum value of the predicted distribution is", np.min(int_doe))             #Getting min of the predicted interest rate
print("The standard deviation of the predicted distribution is", np.std(int_doe))        #Getting standard deviation of the predicted interest rate
print("The median of the predicted distribution is", np.median(int_doe))                 #Getting median of the predicted interest rate



# # Writing out a new csv with the predicted interest rate

# In[ ]:


doe['INTEREST RATE'] = int_doe                #Creating a new column for interest rate in DOE
doe.to_csv("NewDOE_Interest_Rate.csv")        #Writing out the doe dataframe into a csv

