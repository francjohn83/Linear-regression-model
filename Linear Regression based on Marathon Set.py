
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv(r'C:\Users\francis\Downloads\MarathonData.csv')


# In[4]:


data.head(45)


# In[7]:


data = data.drop(['CrossTraining'], axis =1)


# In[9]:


data.head(45)


# In[5]:


data['Category'].value_counts()


# In[6]:


data.corr().plot()


# In[7]:


data.corr()


# In[30]:


data['Wall21'].dtypes


# In[31]:


data.info()


# In[8]:


#replace with NaN
data = data.replace(' -   ',np.nan)


# In[9]:


data = data.fillna(0)
data


# In[10]:


#convert from string to float
data.iloc[:,-3] = data.iloc[:,-3].astype(float)


# In[36]:


data.info()


# In[11]:


import seaborn as sns


# In[12]:


plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True,linewidths=2)


# In[13]:


dummies = pd.get_dummies(data['Name'])

data2 = pd.concat([data, dummies],axis=1)


# In[14]:


dummies2 = pd.get_dummies(data['Marathon'])


# In[15]:


data3 = pd.concat([data2, dummies2],axis=1)


# In[16]:


dummies3 = pd.get_dummies(data['Category'])


# In[17]:


data4 = pd.concat([data3,dummies3],axis=1)


# In[18]:


dummies4 = pd.get_dummies(data['CATEGORY'])


# In[19]:


data5 = pd.concat([data4,dummies4],axis=1)


# In[59]:


final_data = data5.drop(['Marathon','Name','Category','CATEGORY','CrossTraining'],axis=1)


# In[60]:



final_data = final_data.fillna(0)


# In[61]:


final_data.head(50)


# In[62]:


final_data['MarathonTime'].plot.density()


# In[63]:


final_data['MarathonTime'].plot.hist(bins=20,figsize=(8,4))


# In[64]:


X = final_data.drop(['MarathonTime'], axis = 1)


# In[65]:


y = final_data.iloc[:,5]


# In[52]:


y.head(5)


# In[66]:


from sklearn.cross_validation import train_test_split


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# In[68]:


#Check the size and shape of train/test splits (it should be in the ratio as per test_size parameter above)
print("Training feature set size:",X_train.shape)
print("Test feature set size:",X_test.shape)
print("Training variable set size:",y_train.shape)
print("Test variable set size:",y_test.shape)


# In[69]:


#Model fit and training

from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[70]:


lm = LinearRegression()


# In[71]:


#Fit the model on to the instantiated object itself

lm.fit(X_train,y_train)


# In[72]:


#Check the intercept and coefficients and put them in a DataFrame

print("The intercept term of the linear model:", lm.intercept_)


# In[74]:


cdf = pd.DataFrame(data=lm.coef_, index=X_train.columns, columns=["Coefficients"])


# In[77]:


train_pred = lm.predict(X_train)


# In[78]:


metrics.mean_squared_error(y_train,train_pred)


# In[79]:


test_pred = lm.predict(X_test)


# In[80]:


metrics.mean_squared_error(y_test,test_pred)


# In[81]:


metrics.mean_absolute_error(y_train,train_pred)


# In[82]:


np.sqrt(metrics.mean_squared_error(y_train,train_pred))


# In[83]:


print("R-squared value of this fit:",round(metrics.r2_score(y_train,train_pred),3))

