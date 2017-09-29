
# coding: utf-8


# In[3]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')



# In[4]:

df = pd.read_csv('kyphosis.csv')


# In[5]:

df.head()


# ## EDA
# 

# In[6]:

sns.pairplot(df,hue='Kyphosis',palette='Set1')


# ## Train Test Split
# 

# In[7]:

from sklearn.model_selection import train_test_split


# In[8]:

X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']


# In[9]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# ## Decision Trees
# 

# In[10]:

from sklearn.tree import DecisionTreeClassifier


# In[11]:

dtree = DecisionTreeClassifier()


# In[12]:

dtree.fit(X_train,y_train)


# ## Prediction and Evaluation 

# In[13]:

predictions = dtree.predict(X_test)


# In[14]:

from sklearn.metrics import classification_report,confusion_matrix


# In[15]:

print(classification_report(y_test,predictions))


# In[16]:

print(confusion_matrix(y_test,predictions))


# ## Tree Visualization
# 

# In[17]:

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[1:])
features


# In[18]:

dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  


# ## Random Forests

# In[41]:

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[45]:

rfc_pred = rfc.predict(X_test)


# In[46]:

print(confusion_matrix(y_test,rfc_pred))


# In[47]:

print(classification_report(y_test,rfc_pred))

