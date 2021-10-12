#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the required libraries
from sklearn.cluster import KMeans
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


# # KMeans Clustering

# In[12]:


#Elbow method to find number of clusters
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)


# In[13]:


plt.title('The Elbow method')
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[14]:


#Model training
km = KMeans(n_clusters=2)
km.fit_predict(X)
print(km.labels_)
km.labels_


# In[15]:


km.cluster_centers_


# In[16]:


X_grouped=X.copy()
X_grouped['cluster']= km.labels_
X_grouped.head()


# In[17]:


#count each cluster
X_grouped['cluster'].value_counts()


# In[18]:


#Checking unique clusters
X_grouped.cluster.unique()


# In[19]:


#Scatter plot for 'SepalLengthCm' versus 'SepalWidthCm'
sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='cluster',data=X_grouped,palette=['green','red'])
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='*',s=200,label='centroids')
plt.legend()


# In[20]:


sns.scatterplot(x='PetalLengthCm',y='PetalWidthCm',hue='cluster',data=X_grouped,palette=['grey','red'])
plt.scatter(km.cluster_centers_[:,2],km.cluster_centers_[:,3],color='black',marker='*',s=200,label='centroids')
plt.legend()


# In[21]:


#Checking the Silhouette score
score_kemans_s = silhouette_score(X,km.labels_)
print('Silhouette Score of kmeans clustering: %.4f' % score_kemans_s)

pickle.dump(km,open("Kmeans.pkl","wb"))

