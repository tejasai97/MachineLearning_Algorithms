
# coding: utf-8



# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ### Check out the Data

# In[2]:

USAhousing = pd.read_csv('USA_Housing.csv')


# In[3]:

USAhousing.head()


# In[4]:

USAhousing.info()


# In[5]:

USAhousing.describe()


# In[6]:

USAhousing.columns



# In[7]:

sns.pairplot(USAhousing)


# In[8]:

sns.distplot(USAhousing['Price'])


# In[9]:

sns.heatmap(USAhousing.corr())


# ## Training a Linear Regression Model


# In[10]:

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# ## Train Test Split
# 

# In[11]:

from sklearn.model_selection import train_test_split


# In[12]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# ## Creating and Training the Model

# In[13]:

from sklearn.linear_model import LinearRegression


# In[14]:

lm = LinearRegression()


# In[15]:

lm.fit(X_train,y_train)


# ## Model Evaluation
# 
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# In[16]:

# print the intercept
print(lm.intercept_)


# In[24]:

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Income** is associated with an **increase of \$21.52 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area House Age** is associated with an **increase of \$164883.28 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Rooms** is associated with an **increase of \$122368.67 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Bedrooms** is associated with an **increase of \$2233.80 **.
# - Holding all other features fixed, a 1 unit increase in **Area Population** is associated with an **increase of \$15.15 **.
# 
# Does this make sense? Probably not because I made up this data. If you want real data to repeat this sort of analysis, check out the [boston dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html):
# 
# 

#     from sklearn.datasets import load_boston
#     boston = load_boston()
#     print(boston.DESCR)
#     boston_df = boston.data

# ## Predictions from our Model
# 
# Let's grab predictions off our test set and see how well it did!

# In[79]:

predictions = lm.predict(X_test)
#X
#test=X[:1]
#test
#Y=y[:1]
#Y
#test=('Avg. Area Income':10000, 'Avg. Area House Age':30, 'Avg. Area Number of Rooms':5,
#       'Avg. Area Number of Bedrooms':3, 'Area Population':200)
#predictions = lm.predict(test)


# In[80]:

plt.scatter(y_test,predictions)


# In[20]:

sns.distplot((y_test-predictions),bins=50);



# In[21]:

from sklearn import metrics


# In[22]:

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


