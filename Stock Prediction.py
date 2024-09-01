#!/usr/bin/env python
# coding: utf-8

# # Stock Prediction:

# In[ ]:





# In[1]:


#importing
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r"C:\Users\HP\Downloads\stock_market.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.corr()


# In[9]:


df.nunique()


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


df.columns


# In[ ]:





# In[ ]:





# # Data Visualization:

# In[13]:


plt.figure(figsize=(30,10))
sns.lineplot(x='Date',y='Close',data=df)
plt.grid(axis="y")


# In[14]:


sns.pairplot(df)


# In[15]:


sns.scatterplot(df)


# In[16]:


sns.jointplot(df)


# In[17]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:





# In[ ]:





# # Data Modification:

# In[18]:


df


# In[19]:


df.columns


# In[20]:


x = df.drop(['Date','High', 'Low', 'Close', 'Adj Close', 'Volume'],axis=1)
x


# In[21]:


y = df['Close']
y


# In[ ]:





# In[ ]:





# # Machine Learning:

# In[22]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[24]:


x_train


# In[25]:


x_test


# In[26]:


y_train


# In[27]:


y_test


# In[28]:


print(df.shape)
print(x_train.shape)
print(x_test.shape)


# In[29]:


print(df.shape)
print(y_train.shape)
print(y_test.shape)


# In[30]:


model = linear_model.LinearRegression()


# In[34]:


model.fit(x_train,y_train)


# In[35]:


model.score(x_test,y_test)


# In[ ]:





# # Prediction:

# In[36]:


a = model.predict([[262.000]]) # input is Open and our predicion is close
print("Stock close:",a)


# In[ ]:





# In[ ]:




