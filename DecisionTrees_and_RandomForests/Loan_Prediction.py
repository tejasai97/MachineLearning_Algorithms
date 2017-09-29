
# coding: utf-8
 
# **Import the usual libraries for pandas and plotting. You can import sklearn later on.**

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Get the Data
# 

# In[2]:

loans = pd.read_csv('loan_data.csv')


# In[3]:

loans.info()


# In[4]:

loans.describe()


# In[5]:

loans.head()


# # Exploratory Data Analysis
# In[6]:

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')



# In[7]:

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# ** Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **

# In[8]:

plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# ** Let's see the trend between FICO score and interest rate. Recreate the following jointplot.**

# In[9]:

sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# ** Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy. Check the documentation for lmplot() if you can't figure out how to separate it into columns.**

# In[10]:

plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# # Setting up the Data
# 
# Let's get ready to set up our data for our Random Forest Classification Model!
# 
# **Check loans.info() again.**

# In[11]:

loans.info()


# ## Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
# 
# Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary.
# 
# **Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.**

# In[12]:

cat_feats = ['purpose']


# **Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data.**

# In[13]:

final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[14]:

final_data.info()


# ## Train Test Split
# 
# Now its time to split our data into a training set and a testing set!
# 
# ** Use sklearn to split your data into a training set and a testing set as we've done in the past.**

# In[15]:

from sklearn.model_selection import train_test_split


# In[16]:

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ## Training a Decision Tree Model
# 
# Let's start by training a single decision tree first!
# 
# ** Import DecisionTreeClassifier**

# In[17]:

from sklearn.tree import DecisionTreeClassifier


# **Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.**

# In[18]:

dtree = DecisionTreeClassifier()


# In[19]:

dtree.fit(X_train,y_train)


# ## Predictions and Evaluation of Decision Tree
# **Create predictions from the test set and create a classification report and a confusion matrix.**

# In[20]:

predictions = dtree.predict(X_test)


# In[21]:

from sklearn.metrics import classification_report,confusion_matrix


# In[22]:

print(classification_report(y_test,predictions))


# In[23]:

print(confusion_matrix(y_test,predictions))


# ## Training the Random Forest model
# 
# 

# In[24]:

from sklearn.ensemble import RandomForestClassifier


# In[25]:

rfc = RandomForestClassifier(n_estimators=600)


# In[26]:

rfc.fit(X_train,y_train)


# ## Predictions and Evaluation
# 

# In[28]:

predictions = rfc.predict(X_test)



# In[29]:

from sklearn.metrics import classification_report,confusion_matrix


# In[30]:

print(classification_report(y_test,predictions))


# **Show the Confusion Matrix for the predictions.**

# In[31]:

print(confusion_matrix(y_test,predictions))

