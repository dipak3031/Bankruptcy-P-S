#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing


# In[59]:


df = pd.read_csv(r"C:\Users\user\Downloads\bankruptcy-prevention.csv", sep =';')
#loading data and use seperator for seperate the columns


# In[60]:


df


# In[61]:


df= df.rename({' management_risk': 'management risk',' financial_flexibility':'financial flexibility',' credibility':'credibility',' competitiveness':'competitiveness',' operating_risk':'operating risk',' class':'class'}, axis=1)
df.columns


# In[62]:


df["class"] = df["class"].replace("bankruptcy",0)
df["class"] = df["class"].replace("non-bankruptcy",1)


# In[63]:


df


# In[64]:


# Converting dataframe into array
data = df.values
data


# In[65]:


# Selecting Target 
X=data[:,0:6]
Y=data[:,6]
print(X.shape,Y.shape)


# In[66]:


from sklearn.model_selection import train_test_split,cross_val_score


# In[67]:


# Data Split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state=42)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[68]:


from sklearn.svm import SVC


# In[69]:


svc = SVC(kernel="rbf",C=12,gamma=0.5)
svc.fit(x_train, y_train)
pred_svc = svc.predict(x_test)


# In[70]:


import pickle


# In[71]:


from pickle import load


# In[72]:


file = 'SVC_Model.sav'
pickle.dump(svc, open(file,'wb'))


# In[73]:


model=pickle.load(open(r"C:\Users\user\Downloads\SVC_Model.sav",'rb'))

model = svc.predict(x_test)


# In[ ]:




