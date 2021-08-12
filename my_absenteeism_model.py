#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


os.getcwd()


# In[3]:


os.chdir('C:\\Users\\HP\\Downloads')
pre_data=pd.read_csv("Absenteeism_preprocessed.csv")


# In[4]:


pre_data.head()


# ## create target

# In[5]:


# 3>n ---> not acceptable absence= 0
#3<n ----> acceptable absence=1


# In[52]:


target=np.where(pre_data['Absenteeism Time in Hours']>
                pre_data['Absenteeism Time in Hours'].median(),1,0)


# In[53]:


pre_data['splited absenteesim time']=target


# In[54]:


data_with_target=pre_data.drop(['Absenteeism Time in Hours','Day of the Week',
                                            'Daily Work Load Average','Distance to Work'],axis=1)


# In[55]:


data_with_target.head()


# In[56]:


data_with_target.shape


# In[57]:


target.sum() / target.shape[0]


# ## choose inputs for regression

# In[12]:


data_with_target.shape


# In[13]:


data_with_target.iloc[:,:14].head()


# In[14]:


unscaled_input=data_with_target.iloc[:,:-1] 
# we choose all the column except 'splited absenteesim time' bcoz this is a target


# In[15]:


unscaled_input.shape


# ## standardize 

# don't std the dummie variables 
# which are reason_1,reason_2,reason_3,reason_4 and education

# In[16]:


from sklearn.preprocessing import StandardScaler
absent_scaler= StandardScaler()


# In[17]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[18]:


columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education']
columns_to_scale = [x for x in unscaled_input.columns.values if x not in columns_to_omit]
absent_scaler = CustomScaler(columns_to_scale)


# In[19]:


absent_scaler.fit(unscaled_input)


# In[20]:


scaled_input=absent_scaler.transform(unscaled_input)


# In[21]:


scaled_input.shape


# ## splitting for trainig and tsting

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(scaled_input, target, #train_size = 0.8, 
                                                                            test_size = 0.2, random_state = 20)


# In[24]:


#split=train_test_split(scaled_input,target,test_size=0.2,random_state=20)


# In[25]:


#x_train,x_test,y_train,y_test=split


# In[26]:


print(x_train.shape,y_train.shape)


# In[27]:


print(x_test.shape,y_test.shape)


# ## logistic regression model

# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ## training the model

# In[29]:


reg=LogisticRegression()


# In[30]:


reg.fit(x_train,y_train)
#solver='lbfgs'


# In[31]:


reg.score(x_train,y_train)


# In[32]:


reg.coef_


# ## check manually

# In[33]:


check=reg.predict(x_train)


# In[34]:


#y_train


# In[35]:


sum(check==y_train)


# In[36]:


check.shape


# In[37]:


np.sum(check==y_train)/check.shape
#it is same as reg.score(37)


# In[38]:


reg.coef_


# In[39]:


reg.intercept_


# ## summary table

# In[40]:


feature_name=unscaled_input.columns.values


# In[41]:


summary_table=pd.DataFrame(columns=['feature name'],data=feature_name)
summary_table['coefficient']=np.transpose(reg.coef_)


# In[42]:


summary_table.index=summary_table.index +1
summary_table.loc[0]=['intercept',reg.intercept_[0]]
summary_table=summary_table.sort_index()


# In[43]:


summary_table['odds ratio']=np.exp(summary_table.coefficient)
summary_table.sort_values('odds ratio',ascending=False)


# In[44]:


summary_table
# if coeff is near to 0 or adds ratio is 1 
# Distance to Work, Daily Work Load Average, Day of the Week this are the not so importent bcoz coeff is very low in 0.0s
# and odds ratio is in 1s


# ## testing

# In[59]:


reg.score(x_test,y_test)


# In[60]:


predicted_proba = reg.predict_proba(x_test)
predicted_proba


# In[62]:


predicted_proba[:,1]


# In[63]:


predicted_proba.shape


# ## save the model

# In[70]:


import pickle 
# to save model and send it consume very less memmery (1kb) to read we unpickle by using load function instand of dump


# In[68]:


with open('absenteeism_model', 'wb') as file:
    pickle.dump(reg, file)


# In[69]:


with open('absenteeism_scaler','wb') as file:
    pickle.dump(absent_scaler, file)


# last step is a deployment: it used to use model by non technical peoples
