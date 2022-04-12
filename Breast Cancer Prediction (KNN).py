#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

sns.set()

#we can set the style by calling seaborn's set() method


# In[2]:


b_cancer = load_breast_cancer()
b_cancer


# In[3]:


X = pd.DataFrame(b_cancer.data, columns=b_cancer.feature_names)
X = X[['mean area', 'mean compactness']]
y = pd.Categorical.from_codes(b_cancer.target, b_cancer.target_names)
y = pd.get_dummies(y, drop_first=True)

#Make a Categorical type from codes and categories or dtype. This constructor #is useful if you already 
#have codes and categories/dtype and so do not need #the factorization step


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[5]:


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)


# In[6]:


y_pred = knn.predict(X_test)


# In[7]:


sns.scatterplot(x='mean area', y='mean compactness',hue='benign',data=X_test.join(y_test, how='outer'))


# In[9]:


plt.scatter(X_test['mean area'], X_test['mean compactness'],c=y_pred,cmap='coolwarm', alpha=0.7)
#c : color, sequence, or sequence of color
#Matplotlib allows you to adjust the transparency of a graph plot using the #alpha attribute.
#By default, alpha=1. If you want to make the graph plot more #transparent, then you can make alpha 
#less than 1, such as 0.5 or 0.25


# In[10]:


confusion_matrix(y_test, y_pred)


# In[ ]:


#Given our confusion matrix, our model has an accuracy of 121/143 = 84.6%.

