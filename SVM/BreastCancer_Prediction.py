
# coding: utf-8


# In[51]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Get the Data
# 

# In[52]:

from sklearn.datasets import load_breast_cancer


# In[54]:

cancer = load_breast_cancer()



# In[55]:

cancer.keys()



# In[4]:

print(cancer['DESCR'])


# In[56]:

cancer['feature_names']


# ## Set up DataFrame

# In[12]:

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()


# In[14]:

cancer['target']


# In[16]:

df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])


# Now let's actually check out the dataframe!

# In[8]:

df.head()


# # Exploratory Data Analysis
# 
# 


# ## Train Test Split

# In[57]:

from sklearn.model_selection import train_test_split


# In[58]:

X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)



# In[59]:

from sklearn.svm import SVC


# In[60]:

model = SVC()


# In[61]:

model.fit(X_train,y_train)


# ## Predictions and Evaluations
# 

# In[27]:

predictions = model.predict(X_test)


# In[45]:

from sklearn.metrics import classification_report,confusion_matrix


# In[46]:

print(confusion_matrix(y_test,predictions))


# In[62]:

print(classification_report(y_test,predictions))


# Woah! Notice that we are classifying everything into a single class! This means our model needs to have it parameters adjusted (it may also help to normalize the data).
# 

# # Gridsearch


# In[63]:

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[64]:

from sklearn.model_selection import GridSearchCV



# In[65]:

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)



# In[40]:

grid.fit(X_train,y_train)



# In[41]:

grid.best_params_


# In[ ]:

grid.best_estimator_



# In[48]:

grid_predictions = grid.predict(X_test)


# In[49]:

print(confusion_matrix(y_test,grid_predictions))


# In[50]:

print(classification_report(y_test,grid_predictions))


