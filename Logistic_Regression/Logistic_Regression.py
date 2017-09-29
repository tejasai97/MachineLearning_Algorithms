
# coding: utf-8



# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## The Data
# 
# Let's start by reading in the titanic_train.csv file into a pandas dataframe.

# In[2]:

train = pd.read_csv('titanic_train.csv')


# In[3]:

train.head()


# In[4]:

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')



# In[5]:

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[6]:

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[7]:

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[8]:

sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[9]:

train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[10]:

sns.countplot(x='SibSp',data=train)


# In[11]:

train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# ____
# ### Cufflinks for plots


# In[12]:

import cufflinks as cf
cf.go_offline()


# In[13]:

train['Fare'].iplot(kind='hist',bins=30,color='green')


# ___
# ## Data Cleaning

# 

# In[14]:

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[15]:

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age



# In[16]:

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)



# In[17]:

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')



# In[18]:

train.drop('Cabin',axis=1,inplace=True)


# In[19]:

train.head()


# In[20]:

train.dropna(inplace=True)



# In[21]:

train.info()


# In[22]:

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[23]:

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[24]:

train = pd.concat([train,sex,embark],axis=1)


# In[25]:

train.head()


# 
# ## Train Test Split

# In[26]:

from sklearn.model_selection import train_test_split


# In[27]:

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# ## Training and Predicting

# In[28]:

from sklearn.linear_model import LogisticRegression


# In[29]:

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[30]:

predictions = logmodel.predict(X_test)



# ## Evaluation


# In[31]:

from sklearn.metrics import classification_report


# In[32]:

print(classification_report(y_test,predictions))



