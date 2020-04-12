#!/usr/bin/env python
# coding: utf-8

# ## Name : Abhishek Chauhan(11704760) and Vishal Kumar(11704868)
# ### Section : KM030
# House Price Prediction
# Description : This is a notebook for visualization of various features which the sales price of houses. Then data is taken from the "Kaggle House Price Prediction" challenge.

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr


# # Loading the Data

# In[155]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# ## Preview the data

# In[156]:


train.head()


# In[157]:


test.head()


# In[158]:


train.describe()


# In[159]:


train.info()


# In[160]:


test.info()


# ## Data Manipulation and Visualization
# Lets check for NaN (null) values in the data

# In[161]:


train.isnull().sum()


# In[162]:


test.isnull().sum()


# Lets check for the mean, standard deviation for Sales price

# In[163]:


train['SalePrice'].describe()


# In[164]:


# Determining the Skewness of data 
print ("Skew is:", train.SalePrice.skew())


# In[165]:


plt.hist(train.SalePrice)
plt.show()
sns.distplot(train.SalePrice,fit=norm)
plt.ylabel =('Frequency')
plt.title = ('SalePrice Distribution')
#Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
#QQ plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
print("skewness: %f" % train['SalePrice'].skew())
print("kurtosis: %f" % train ['SalePrice'].kurt())


# Sales price is right skewed. So, we perform log transformation so that the skewness is nearly zero

# In[166]:


train["skewed_SP"] = np.log1p(train["SalePrice"])
print ("Skew is:", train['skewed_SP'].skew())
plt.hist(train['skewed_SP'], color='blue')
plt.show()
sns.distplot(train.SalePrice,fit=norm)
plt.ylabel=('Frequency')
plt.title=('SalePrice distribution');

(mu,sigma)= norm.fit(train['SalePrice']);

fig =plt.figure()
res =stats. probplot(train['SalePrice'], plot=plt)
plt.show() 


# Exploring the variables

# In[167]:


#correration matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,vmax=0.9, square=True)
plt.show()


# In[168]:


# Data Transformation 
print ("Original: \n") 
print (train.Street.value_counts(), "\n")


# In[169]:


train['SaleCondition'].value_counts()


# In[170]:


train['SaleType'].value_counts()


# In[171]:


# Turn into one hot encoding 
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)


# In[172]:


# Encoded 
print ('Encoded: \n') 
print (train.enc_street.value_counts())


# In[173]:


def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)
data = train.select_dtypes(include=[np.number]).interpolate().dropna()


# In[174]:


sum(data.isnull().sum() != 0)


# In[175]:


train.OverallQual.unique()


# In[176]:


# Linear Model for the  train and test
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)


# In[177]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)


# In[178]:


from sklearn import linear_model
from sklearn import ensemble
lr = ensemble.GradientBoostingRegressor()


# In[179]:


model = lr.fit(X_train, y_train)


# In[180]:


print ("R^2 is: \n", model.score(X_test, y_test))


# In[181]:


y_pred = model.predict(X_test)


# In[182]:


from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, y_pred))


# In[183]:


plt.scatter(y_pred, y_test, alpha=.75,color='b') 


# In[ ]:





# In[ ]:





# In[ ]:




