
# coding: utf-8

# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Get the Data

# In[2]:

ad_data = pd.read_csv('advertising.csv')


# **Check the head of ad_data**

# In[3]:

ad_data.head()


# ** Use info and describe() on ad_data**

# In[4]:

ad_data.info()


# In[5]:

ad_data.describe()


# ## Exploratory Data Analysis


# In[6]:

sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


# **Create a jointplot showing Area Income versus Age.**

# In[7]:

sns.jointplot(x='Age',y='Area Income',data=ad_data)


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[8]:

sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[9]:

sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')



sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')


# # Logistic Regression
# 

# In[11]:

from sklearn.model_selection import train_test_split


# In[12]:

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']


# In[13]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



# In[14]:

from sklearn.linear_model import LogisticRegression


# In[15]:

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)



# In[18]:

predictions = logmodel.predict(X_test)



# In[19]:

from sklearn.metrics import classification_report


# In[20]:

print(classification_report(y_test,predictions))

