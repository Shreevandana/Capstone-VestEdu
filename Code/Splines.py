
# coding: utf-8

# ## Import Libraries

# In[121]:


import os
import time

import numpy as np
import pandas as pd

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns 

# Pre-processingo
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, LabelBinarizer, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# Grid-Search and Cross-Validation
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve, train_test_split, StratifiedKFold, KFold

#splines
import pyearth
plt.figure(figsize=(10, 5))
plt.rcParams['figure.dpi'] = 300
sns.set_palette("Set2")


# ## Read Data (Long format)

# In[2]:


cdrdf = pd.read_csv("alternative_cdr_long_format.csv")
cdrdf=cdrdf[cdrdf['forbes_university_rank']<=300]


# ## Pre-processing and correcting the columns

# In[3]:


from pandas.api.types import CategoricalDtype

# Correct all data-types
# 1. Correct the degree column to ordinal categorical column
deg = pd.Categorical(cdrdf.degree, 
                     categories=['UG', 'G'], 
                     ordered=True)

labels, unique = pd.factorize(deg, sort=True)

cdrdf.degree = labels

# 2. Scale the decimal cdr to percentages 0.02 = 2%
cdrdf['cdr2_100'] = cdrdf['cdr2'] * 100
cdrdf['cdr3_100'] = cdrdf['cdr3'] * 100


# In[4]:


reg_df = cdrdf.loc[:, ['forbes_university_rank', 'tier', 'degree', 'major', 'cdr2_100', 'cdr3_100']]


# In[5]:


oe = OrdinalEncoder(dtype = np.int64) # specific to degree
reg_df.loc[:, 'degree'] = oe.fit_transform(reg_df.loc[:, 'degree'].values.reshape(-1, 1))

# One-hot encoding for major and tier
reg_df = pd.get_dummies(reg_df, 
                        columns = ['major', 'tier'],
                        prefix = ['major', 'tier'],
                        drop_first = False)


# In[6]:


print(reg_df.head(), reg_df.shape)


# ## Train-test Split (80-20)

# In[100]:


# train-test split
X = reg_df.loc[:, ['forbes_university_rank', 'major_Arts', 'major_Business', 'major_STEM', 'degree', 
                   'tier_1', 'tier_2', 'tier_3', 'tier_4', 'tier_5', 'tier_6']]
y = reg_df.loc[:, ['cdr2_100', 'cdr3_100']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# # Train Model

# In[101]:


x_cols = ['forbes_university_rank', 'major_Business', 'major_STEM','major_Arts', 'degree'] #predictors we are interested in
earth=pyearth.Earth(feature_importance_type='gcv',allow_linear=False) #cdr2
earth2=pyearth.Earth(feature_importance_type='gcv',allow_linear=False) #cdr3


# In[102]:


earth.fit(X_train,np.array(y_train)[:,0])
earth2.fit(X_train,np.array(y_train)[:,1])


# In[ ]:


#feature importance
for i in range(0,5):
    print('The importance of {} in CDR 2 model is {}'.format(x_cols[i],earth.feature_importances_[i]*100))
    print('The importance of {} in CDR 3 model is {}'.format(x_cols[i],earth2.feature_importances_[i]*100))


# In[ ]:


#predicitons
y_hat=np.append(earth.predict(X_train),earth.predict(X_test))
y_hat2=np.append(earth2.predict(X_train),earth2.predict(X_test))


# In[133]:


new_df=X_train.append(X_test) # to hold predicitons 
new_df['pred_cdr2']=y_hat
new_df['pred_cdr3']=y_hat2


# In[124]:


#MSE
y_hat = earth.predict(X_test)
mse=np.average([(x-y)**2 for (x,y) in zip(y_hat,np.array(y_test)[:,0])])
mse
y_hat2 = earth2.predict(X_test)
mse2=np.average([(x-y)**2 for (x,y) in zip(y_hat2,np.array(y_test)[:,1])])
print('The MSE for the CDR2 model is {}'.format(mse))
print('The MSE for the CDR3 model is {}'.format(mse2))


# # Plotting

# In[126]:


new_df=new_df.sort_values('forbes_university_rank') #to make plot look better


# In[127]:


for i in range(2):
    temp=new_df[new_df['degree']==i]
    if i is 0:
        grad='Undergraduate'
    else:
        grad='Graduate'
    for j in range(1,4):
        temp2=temp[temp[temp.columns[j]]==1]
        
        plt.plot(list(temp2['forbes_university_rank'])[0:-5],list(temp2['pred_cdr2'])[0:-5],'-',label='{} {}'.format(grad,temp.columns[j]))
        plt.xlabel('University Rank')
        plt.ylabel('2 Year Default Rate Percentage')
        plt.ylim((-2,10))
        plt.legend(loc = 'upper left')
        plt.title('2 Year Default Rate Predictions')
    plt.show()


# In[140]:


for i in range(2):
    temp=new_df[new_df['degree']==i]
    if i is 0:
        grad='Undergraduate'
    else:
        grad='Graduate'
    for j in range(1,4):
        temp2=temp[temp[temp.columns[j]]==1]
        
        plt.plot(list(temp2['forbes_university_rank'])[0:-5],list(temp2['pred_cdr3'])[0:-5],'-',label='{} {}'.format(grad,temp.columns[j]))
        plt.xlabel('University Rank')
        plt.ylabel('3 Year Default Rate Percentage')
        plt.ylim((-2,10))
        plt.legend(loc = 'upper left')
        plt.title('3 Year Default Rate Predictions')
plt.show()

