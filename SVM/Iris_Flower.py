
# coding: utf-8


# In[17]:

# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[18]:

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[19]:

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

# ## Get the data
# 
# **Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **

# In[24]:

import seaborn as sns
iris = sns.load_dataset('iris')


# ## Exploratory Data Analysis
# 

# In[25]:

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')



# In[37]:

sns.pairplot(iris,hue='species',palette='Dark2')



# In[44]:

setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)


# # Train Test Split
# 

# In[45]:

from sklearn.model_selection import train_test_split


# In[47]:

X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# # Train a Model
# 

# In[48]:

from sklearn.svm import SVC


# In[49]:

svc_model = SVC()


# In[50]:

svc_model.fit(X_train,y_train)


# ## Model Evaluation

# In[51]:

predictions = svc_model.predict(X_test)


# In[52]:

from sklearn.metrics import classification_report,confusion_matrix


# In[53]:

print(confusion_matrix(y_test,predictions))


# In[54]:

print(classification_report(y_test,predictions))



# In[55]:

from sklearn.model_selection import GridSearchCV



# In[57]:

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 



# In[58]:

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)



# In[59]:

grid_predictions = grid.predict(X_test)


# In[60]:

print(confusion_matrix(y_test,grid_predictions))


# In[61]:

print(classification_report(y_test,grid_predictions))


