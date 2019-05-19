 #In[7]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split



# In[8]:


os.getcwd()


# In[9]:


import warnings
warnings.filterwarnings("ignore")


# In[10]:

data=pd.read_csv('dtu.csv')
data=data.fillna(data.mean(),inplace=True)

train, test = train_test_split(data, test_size=0.25)

train.shape
test.shape





# In[11]:





# In[12]:


train.head()


# In[13]:


test.head()






# In[15]:





# In[16]:


train.dtypes


# In[17]:





# In[18]:





# In[19]:


X = train.iloc[:,1:6]


# In[20]:


X.head()


# In[21]:


y = train.iloc[:,0]
y.head()

# In[22]:


X_t=test.iloc[:,1:6]


# In[25]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[24]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test) 


# In[21]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# In[26]:


X_t.shape


# In[28]:


test.shape


# In[29]:


predtest=regressor.predict(X_t)


# In[33]:


sol=pd.DataFrame()


# In[34]:


sol['soldierId']=test.soldierId


# In[35]:


pred= pd.DataFrame(predtest)


# In[36]:


pred.head()


# In[37]:


sol['bestSoldierPerc']=pred.iloc[:,0]


# In[38]:


sol.head()


# In[39]:


sol.shape


# In[40]:


test.shape


# In[42]:


sol.to_csv('submission_file.csv',index=False)
