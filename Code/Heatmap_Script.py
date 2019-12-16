
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_excel('NewDOEData_corrected.xlsx') #read in data


# In[89]:


df=df[['RANK', 'MAJOR', 'CREDDESC','LOAN_STATUS_MDTI']] #subset to columns we are interested in 


# In[90]:



df=df.rename(columns={'CREDDESC': 'EDUCATION_LEVEL', 'LOAN_STATUS_MDTI': 'LOAN_STATUS'}) #rename columns


# In[91]:


df2=df[df['EDUCATION_LEVEL']=="Master's Degree"] #remove degrees we aren't interested in 
df3=df[df['EDUCATION_LEVEL']=="Bachelor\x92s Degree"]
df2=df2.append(df3)


# In[92]:


import dython.nominal
df2=df2[df2['RANK']<=300] #nothing higher than rank 300


# In[93]:


dython.nominal.associations(df2, nominal_columns='all')

