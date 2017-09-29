
# coding: utf-8


# It is **very important to note, we actually have the labels for this data set, but we will NOT use them for the KMeans clustering algorithm, since that is an unsupervised learning algorithm.** 



# In[103]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Get the Data


# In[104]:

df = pd.read_csv('College_Data',index_col=0)



# In[105]:

df.head()



# In[106]:

df.info()


# In[107]:

df.describe()


# ## EDA
# 

# In[111]:

sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)



# In[112]:

sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)



# In[109]:

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)



# In[110]:

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)



# In[113]:

df[df['Grad.Rate'] > 100]



# In[93]:

df['Grad.Rate']['Cazenovia College'] = 100


# In[94]:

df[df['Grad.Rate'] > 100]


# In[95]:

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# ## K Means Cluster Creation

# In[114]:

from sklearn.cluster import KMeans




kmeans = KMeans(n_clusters=2)



# In[116]:

kmeans.fit(df.drop('Private',axis=1))



# In[117]:

kmeans.cluster_centers_


# ## Evaluation
# 

# In[118]:

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[119]:

df['Cluster'] = df['Private'].apply(converter)


# In[122]:

df.head()



# In[123]:

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


