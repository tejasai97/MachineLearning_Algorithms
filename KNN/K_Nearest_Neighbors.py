
# coding: utf-8


# 
# 

# In[1]:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')


# ## Get the Data
# 

# In[2]:

df = pd.read_csv("Classified Data",index_col=0)


# In[3]:

df.head()


# ## Standardize the Variables
# 

# In[4]:

from sklearn.preprocessing import StandardScaler


# In[5]:

scaler = StandardScaler()


# In[6]:

scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[7]:

scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[8]:

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# ## Train Test Split

# In[9]:

from sklearn.model_selection import train_test_split


# In[10]:

X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)


# ## Using KNN
# 
#
# In[11]:

from sklearn.neighbors import KNeighborsClassifier


# In[12]:

knn = KNeighborsClassifier(n_neighbors=1)


# In[13]:

knn.fit(X_train,y_train)


# In[14]:

pred = knn.predict(X_test)


# ## Predictions and Evaluations


# In[15]:

from sklearn.metrics import classification_report,confusion_matrix


# In[16]:

print(confusion_matrix(y_test,pred))


# In[17]:

print(classification_report(y_test,pred))


# ## Choosing a K Value
# 

# In[18]:

error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[19]:

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[21]:


knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

