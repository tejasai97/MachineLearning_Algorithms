
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

# In[2]:

df = pd.read_csv('KNN_Project_Data')


# **Check the head of the dataframe.**

# In[3]:

df.head() 



# In[4]:

sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')




# In[5]:

from sklearn.preprocessing import StandardScaler


# In[6]:

scaler = StandardScaler()


# ** Fit scaler to the features.**

# In[7]:

scaler.fit(df.drop('TARGET CLASS',axis=1))


# **Use the .transform() method to transform the features to a scaled version.**

# In[8]:

scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[9]:

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# # Train Test Split
# 
#
# In[10]:

from sklearn.model_selection import train_test_split


# In[11]:

X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)


# # Using KNN
# 

# In[12]:

from sklearn.neighbors import KNeighborsClassifier


# **Create a KNN model instance with n_neighbors=1**

# In[13]:

knn = KNeighborsClassifier(n_neighbors=1)


# **Fit this KNN model to the training data.**

# In[14]:

knn.fit(X_train,y_train)


# # Predictions and Evaluations

# In[15]:

pred = knn.predict(X_test)


# ** Create a confusion matrix and classification report.**

# In[16]:

from sklearn.metrics import classification_report,confusion_matrix


# In[17]:

print(confusion_matrix(y_test,pred))


# In[18]:

print(classification_report(y_test,pred))


# # Choosing a K Value


# In[19]:

error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))



# In[20]:

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# ## Retrain with new K Value
# 

# In[21]:

# NOW WITH K=30
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


