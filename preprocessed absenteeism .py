#!/usr/bin/env python
# coding: utf-8

# ## import modules

# In[1]:


import pandas as pd
import os


# ## load data

# In[2]:


os.getcwd()


# In[3]:


os.chdir('C:\\Users\\HP\\Downloads')
raw_data=pd.read_csv("Absenteeism_data.csv")


# ## preprocess the data

# In[4]:


df=raw_data.copy()
df.head()


# In[5]:


df=df.drop(['ID'],axis=1)


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


Reason_dummie=pd.get_dummies(df['Reason for Absence'],drop_first=True)


# In[9]:


Reason_dummie['cheack']=Reason_dummie.sum(axis=1) # to check is there any null value


# In[10]:


Reason_dummie


# In[11]:


sum(Reason_dummie['cheack'])


# In[12]:


Reason_dummie=Reason_dummie.drop(['cheack'],axis=1)


# In[13]:


Reason_dummie


# In[14]:


df=df.drop(['Reason for Absence'],axis=1)


# In[15]:


reason_1=Reason_dummie.iloc[:,1:14].max(axis=1)
reason_2=Reason_dummie.iloc[:,15:17].max(axis=1)
reason_3=Reason_dummie.iloc[:,18:21].max(axis=1)
reason_4=Reason_dummie.iloc[:,22:].max(axis=1)


# ## concatinate (merge)

# In[16]:


df=pd.concat([df,reason_1,reason_2,reason_3,reason_4],axis=1)


# In[17]:


df


# In[18]:


df.columns.values


# In[19]:


df.columns=['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours','reason_1','reason_2','reason_3','reason_4']


# In[20]:


df


# In[21]:


rerecord=['reason_1','reason_2','reason_3','reason_4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']


# In[22]:


df=df[rerecord]


# In[23]:


df


# ## checkpoint 

# In[24]:


df1=df.copy() #for safty


# In[25]:


df1['Date']


# In[26]:


df1['Date']=pd.to_datetime(df1['Date'],format='%d/%m/%Y')


# In[27]:


df1['Date'][0].month


# In[28]:


month_list=[]
for i in range(700):
    month_list.append(df1['Date'][i].month)


# In[29]:


df1['month value']=month_list


# In[30]:


df1.head()


# In[31]:


df1['Date'][699].weekday()


# In[32]:


def day_to_weekday(date_value):
    return date_value.weekday()


# In[33]:


df1['day of week']=df1['Date'].apply(day_to_weekday)


# In[34]:


df1.head()


# In[35]:


df1['Education']=df1["Education"].map({1:0,2:1,3:1,4:1}) # divide into 2 parts 0 fot hhc and 1 for degree,graduation and phd


# In[36]:


df1.head()


# ## final check point

# In[37]:


processed_data=df1.copy()


# In[38]:


processed_data


# In[39]:


df1


# In[41]:


import numpy as np
np.savez('preprocessed_data',df1)


# this notebook is only used for preprocess the data, other algorithm like ml, test train all done in different notebook.
