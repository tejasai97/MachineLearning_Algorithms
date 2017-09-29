
# coding: utf-8


# ## Import Libraries

# In[22]:

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## Create some Data

# In[23]:

from sklearn.datasets import make_blobs


# In[42]:

# Create Data
data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)


# ## Visualize Data

# In[43]:

plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')


# ## Creating the Clusters

# In[48]:

from sklearn.cluster import KMeans


# In[49]:

kmeans = KMeans(n_clusters=4)


# In[50]:

kmeans.fit(data[0])


# In[51]:

kmeans.cluster_centers_


# In[55]:

kmeans.labels_


# In[69]:

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')


