#!/usr/bin/env python
# coding: utf-8

# # Fashion Classification
# 
# In this session we are going to classify from a dataset what types of clothing accessories are recognised in an image.
# 
# - Our training set consists of 70,000 images and we'll divide them into 60,000 training images and 10,000 test images, these images are coverted into gray scale and are available in size of 28x28 pixels
# - We have converted one image into gray scale and flatten it's values of 784 pixels (28x28) in a row
# 
# There are 10 classes :
# 0 => T-shirt/Top
# 1 => Trousers
# 2 => Pullovers
# 3 => Dress 
# 4 => Coat
# 5 => Sandal
# 6 => Shirt
# 7 => Sneaker
# 8 => Bag
# 9 => Ankle Boot
# 
# <img src = 'Fashion_Classification_type_of_clothes_files/dataset_images.png' alt='dataset_images'>

# ## Importing our basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing our Fashion Classification Training and Testing Dataset
fashion_train_df = pd.read_csv('input/fashion-mnist_train.csv', sep=',')
fashion_test_df = pd.read_csv('input/fashion-mnist_test.csv', sep=',')

fashion_train_df.head()

fashion_train_df.shape


# ## Converting our Datasets into numpy arrays
training = np.array(fashion_train_df, dtype='float32')
testing = np.array(fashion_test_df, dtype='float32')


# ## Plotting our dataset tuples into a graph
# - First let's import random tuples from our dataset
# - We'll be skipping the label in our plot function.
# - We will use the imshow function of matplot library
# - Also resphaping our array back into it's original form i.e 28x28

import random
labels = {0 : 'T-shirt/Top',1 : 'Trousers',2 : 'Pullovers',3 : 'Dress' ,4 : 'Coat',5 : 'Sandal',6 : 'Shirt',7 : 'Sneaker',8 : 'Bag'
,9 : 'Ankle Boot'}
number = random.randint(1,60000)
plt.title(f'{labels[training[number,0]]}')
plt.imshow(training[number, 1:].reshape(28,28))


# - Let us view our data in a grid format to understand our data set
# - We'll create a grid of 15x15 to plot our categories of clothing
# - This grid will have images with figure size 0f 17 by 17 
# - It will randomly pick 255 tuples from the dataset

W_grid = 15
L_grid =15

fig,axes = plt.subplots(W_grid, L_grid, figsize = (17,17))

axes = axes.ravel() #Flatten our 15x15 matrix into array of 255

len_training = len(training)
    
for i in np.arange(0, W_grid * L_grid):
    index = random.randint(0, len_training)
    axes[i].imshow(training[index, 1:].reshape(28,28))
    axes[i].set_title(f'{labels[training[index,0]]}', fontsize = 8)
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.4)    


# ## Our first step is to divide our data into Training and Test sets

X_train = training[:,1:]/255 # We are doing so in order to Normalise our data for better computation
y_train = training[:, 0]


X_test = testing[:,1:]/255 # We are doing so in order to Normalise our data for better computation
y_test = testing[:, 0]


from sklearn.model_selection import train_test_split
X_train,X_validate,y_train,y_validate = train_test_split(X_train,y_train, test_size = 0.2, random_state= 12345)


# - Reshaping our training and test data into the size of 28x28

X_train = X_train.reshape(X_train.shape[0], *(28,28,1))
X_test = X_test.reshape(X_test.shape[0], *(28,28,1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28,28,1))


X_train[1,:,:].shape


import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# #### We'll add a Convolutional layer which will apply 32 feature maps on each and every one of our tuple

cnn_model = Sequential()
cnn_model.add(Conv2D(32,3,3, input_shape = (28,28,1), activation = 'relu'))


# #### Now adding a Max Pooling layer which will minimise our tuple into a 2x2 image preserving all the important features

cnn_model.add(MaxPooling2D(pool_size = (2,2)))


# #### Now our data is ready to be fed into a fully connected ANN but before that we need to flatten our matrix.

cnn_model.add(Flatten())


# #### Here we'll feed our points into the ANN with 32 hidden layers and gives us back an output of 11 according to our labels

cnn_model.add(Dense(output_dim = 32, activation = 'relu'))
cnn_model.add(Dense(output_dim = 11, activation = 'sigmoid'))


# #### In this step we'll apply the loss function considering that our outputs have multiple categories with an optimizer function and at the end fit our model

cnn_model.compile(loss= 'sparse_categorical_crossentropy', optimizer = Adam(lr = 0.01), metrics= ['accuracy'])
epochs = 50
cnn_model.fit(X_train,y_train, batch_size=512, nb_epoch = epochs, verbose=1,validation_data=(X_validate,y_validate))


# ## Now as Our model is done training we can move on to evalute and check how it performed

evalutaion = cnn_model.evaluate(X_test,y_test)
print(f'Test Accuracy : {evalutaion[1]}')



predicted_classes = cnn_model.predict_classes(X_test)


# ## Now as before We'll create a subplot to visually rectify how our CNN Model performed

W = 5
L = 5

fig, axes = plt.subplots(W,L, figsize=(17,17))

axes = axes.ravel()

for i in np.arange(0, W*L):
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title(f'True : {labels[y_test[i]]}, Predicted : {labels[predicted_classes[i]]}', fontsize = 9)
    axes[i].axis('off')
    
plt.subplots_adjust(wspace= 0.5)    


# ## Now creating a confusion matrix and implementing it in our seaborn heatmap to check out our true predictions

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm , annot = True)


# ## Now we'll create a classification report to check which class was identified more accrately then other classes


from sklearn.metrics import classification_report
target_names = [f"{labels[predicted_classes[i]]}" for i in range(0, len(labels.keys()))]
print(classification_report(y_test, predicted_classes, target_names= target_names))


# ## The average of our model turned out to be 90% so, I guess that's pretty good.
