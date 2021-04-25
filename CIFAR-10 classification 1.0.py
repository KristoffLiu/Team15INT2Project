#!/usr/bin/env python
# coding: utf-8

# # Downloading and preparing the data

# In[3]:


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# In[5]:


(train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()


# In[6]:


train_x, test_x = train_x / 255.0, test_x / 255.0 #changes pixel values between 0 and 1


# # Creating a basic model with 3 layers

# In[9]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) #creates layers for a 2d array of pixels
model.add(layers.MaxPooling2D((2,2))) #pooling helps detect features in an image, max pooling finds largest value
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
#model.summary()


# In[10]:


model.add(layers.Flatten()) #reduces input by one dim
model.add(layers.Dense(64, activation = 'relu')) #regularisation of data
model.add(layers.Dense(10))
#model.summary()


# # Running, training, and evaluating data

# In[13]:


model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
#adam optimisatin is a slightly better gradient decent for computer vision
progress = model.fit(train_x, train_y, epochs=10,validation_data=(test_x, test_y))


# In[17]:


loss, acc = model.evaluate(test_x, test_y, verbose = 2)
print(acc)


# # Used websites

# https://www.tensorflow.org/tutorials/images/cnn
# https://www.tensorflow.org/addons/tutorials/layers_normalizations
# https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
