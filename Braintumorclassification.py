#!/usr/bin/env python
# coding: utf-8

# <h1>Brain tumor detection using SVM Binary classification 

# <h3>Support Vector Machine Assumptions</h3>
#  <h5>1.Support Vector Machines is a family of algorithms that can operate in linear and non-linear data sets.</h5>
#  <h5>2.Along with neural networks, SVMs are probably the best choice among many tasks where it is not<br> easy to find a good separation hyperplane.

# <h3>Loaded required Library</h3>

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# <h3>Load the dataset and prepare the data</h3>

# In[3]:


import os
#The os module provides dozens of functions for interacting with the operating system:
path = os.listdir("C:/Users/Harikrishnan/Desktop/Btumor/Training/")
# method in python is used to get the list of all files and directories in the specified directory
classes = {'no_tumor':0, 'pituitary_tumor':1}
# Machine learning works on numerical data
#so it is cross labelled, no tumour for class 0 and pitutary tumor into class 1 


# <h5>importing open cv module for image recognisation

# In[4]:


import cv2
#OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library.
Allimage = []
# created an empty list Allimage, append all the images into this list
Target = []
# created an empty list Target,  After storing the data into to classes
for cls in classes:
    pth = "C:/Users/Harikrishnan/Desktop/Btumor/Training/"+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        #imread is a method returns an image that is loaded from the specified file.
        #0 - It specifies to load an image in grayscale mode.
        img = cv2.resize(img, (200,200))
        # In machine learning algorithm all the images expected to have same dimension
        Allimage.append(img)
        # Appending all images into Allimage
        
        Target.append(classes[cls])
        #Appending the sorted data into Target


# In[5]:


Allimage = np.array(Allimage)
Target = np.array(Target)
#converting this into numpy array
print(Allimage.ndim)

Allimage_updated = Allimage.reshape(len(Allimage), -1)
#converting the dataset into 2d, sklearn works on 2 dimensional data
print(Allimage_updated)
#printing the data on 2d


# <h3>Data Analysis

# In[6]:


np.unique(Target)
# Find the unique elements of an array.


# In[7]:


pd.Series(Target).value_counts()
#checking the numberof non tumorous data and tumorous data from each classes


# In[8]:


Allimage.shape, Allimage_updated.shape
#comparing the shape before and after reshaping.
#Dataset was 3d then it conveted into 2d because sklearn works on 2d data


# <h3>Data Visualization</h3>

# In[9]:


plt.imshow(Allimage[0], cmap='gray')
# Display an Image in Grayscale in Matplotlib


# <h3>Split data

# <h5>In this step, we are going to split data in two parts (training and testing), so that we can train our model on training dataset and test its accuracy on unseen (test) data.

# In[10]:


allimagetrain, allimagetest, target_train, target_test = train_test_split(Allimage_updated, Target, random_state=10,
                                               test_size=.20)
# Using train_test_split() from the data science library scikit-learn,
# it can split arrays or matrices into random train and test subsets.
# random state : train_test_split splits the data randomly, so we should give an intiger as random state.
# so that the way of splitting must be same


# In[11]:


allimagetrain.shape, allimagetest.shape
#Shape of the data after splitting it into allimage train and all imagetest


# <h3>Feature Scaling

# <h5>Feature Scaling is a technique of bringing down the values of all the independent features<br>
# of our dataset on the same scale.</h5>
# <h5>Feature selection helps to do calculations in algorithms very quickly.<br>It is the important stage of data preprocessing.

# In[12]:


print(allimagetrain.max(), allimagetrain.min())
print(allimagetest.max(), allimagetest.min())
#printing the training and testing value before scaling
allimagetrain = allimagetrain/255
allimagetest = allimagetest/255
#Since the RGB Value changes from 0 to 255, Here it divided the value by 255,
# So that the allimagetrain and imagetest value would be from 0 to 1
print(allimagetrain.max(), allimagetrain.min())
print(allimagetest.max(), allimagetest.min())
#printing training and testing value after Scaling


# <h3>Model Training

# <h5>As we have done with preprocessing part, it is time to train our model. I am going to train model using SVM (Support Vector Machine).

# In[13]:


from sklearn.svm import SVC
# importing SVC algorithm from sklearn library


# In[14]:


sv = SVC()
#calling support vector classifier
sv.fit(allimagetrain, target_train)
#The algoriithm is learning the input data


# <h3>Evaluation

# In[15]:


print("Training Score:", sv.score(allimagetrain, target_train))
#printing the accuracy of Training score using sv.score


# <h3>Prediction

# In[16]:


pred = sv.predict(allimagetest)
#predicting the testdata using predict and checking how accurate the output coming


# In[17]:


misclassified=np.where(target_test!=pred)
#finding where the testing value is not accurate
misclassified


# In[18]:


print("Total Misclassified Samples: ",len(misclassified[0]))
#printing the total number of misclassified samples
print(pred[36],target_test[36])
#printing an example for misclassified data

