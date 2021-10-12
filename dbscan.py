#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the required libraries
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


# # DBSCAN Clustering

# In[32]:


#Importing libraries for DBSCAN
from sklearn.cluster import DBSCAN
from sklearn import metrics


# In[33]:


#Model training
db= DBSCAN(eps=0.9,min_samples=5)
db_predict=db.fit_predict(X)
print(db_predict)


# In[34]:

X_grouped=X.copy()
X_grouped['dbscan_predicted'] = db_predict
X_grouped.head()


# In[35]:


X_grouped['dbscan_predicted'].value_counts()


# In[36]:


sns.scatterplot(x='PetalLengthCm',y='PetalWidthCm',hue='dbscan_predicted',data=X_grouped,palette=['blue','green'])


# In[37]:


sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='dbscan_predicted',data=X_grouped,palette=['red','green'])


# In[38]:


score_dbsacn_s = silhouette_score(X, db_predict)
print('Silhouette Score of Dbscan clustering: %.4f' % score_dbsacn_s)

pickle.dump(db,open("dbscan.pkl","wb"))

