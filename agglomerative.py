#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the required libraries
from sklearn.cluster import AgglomerativeClustering 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import pickle


# In[2]:


#Data understanding
#Reading the dataset
df = pd.read_csv("Iris.csv")
# Show first five rows from data set
df.head()


# In[3]:


#Checking the shape
df.shape


# In[4]:


#count each species
df['Species'].value_counts()


# In[5]:


#Checking the Metadata Information
df.info()


# In[6]:


#Checking how data is spread
df.describe()


# In[7]:


#Checking correlation
plt.figure(figsize=(6,6))
corrmat = df.drop('Id',axis=1).corr()
sns.heatmap(corrmat, annot=True, square= True,cmap='rainbow')
plt.show()


# In[8]:


#Droping the Id and Species columns
X=df.drop(['Id', 'Species'],axis=1)


# In[9]:


print(X.shape)


# In[10]:


X.head()


# In[11]:


#Visualizing the data
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.scatter(df['SepalLengthCm'],df['SepalWidthCm'],color='blue')
plt.xlabel('SepalLength(Cm)')
plt.ylabel('SepalWidth(Cm)')

plt.subplot(1,2,2)
plt.scatter(df['PetalLengthCm'],df['PetalWidthCm'],color='blue')
plt.xlabel('PetalLength(Cm)')
plt.ylabel('PetalWidth(Cm)')


# Dendrogram for Hierarchical Clustering
#Importing libraries for Agglomerative Clustering
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot


# In[23]:


pyplot.figure(figsize=(10, 7))  
pyplot.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X, method='complete'))


# In[24]:


pyplot.figure(figsize=(10, 7))  
pyplot.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X, method='complete'))
plt.axhline(y=3,color='r',linestyle='--')
plt.show()


# # AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')

cluster_pred=model.fit_predict(X)
print(cluster_pred)

# In[27]:

X_grouped=X.copy()
X_grouped['Cluster'] = cluster_pred
X_grouped.head()


# In[28]:


X_grouped['Cluster'].value_counts()


# In[29]:


sns.scatterplot(x='PetalLengthCm',y='PetalWidthCm',data=X_grouped,palette=['red','blue','green'])


# In[30]:


sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',data=X_grouped,palette=['red','blue','green'])


# In[31]:


#Checking the Silhouette score
score_AGclustering_s  = silhouette_score(X,cluster_pred)
print('Silhouette Score: %.4f' % score_AGclustering_s )

pickle.dump(model,open("agglomerative.pkl","wb"))