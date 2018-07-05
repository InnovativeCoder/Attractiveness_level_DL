import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from PIL import Image
from sklearn.model_selection import train_test_split

labels = pd.read_csv("Final_data.csv")
del labels["Unnamed: 0"]

new_data = labels.copy()
del new_data["Image_no."]

new_data.insert(0,'Image_no',labels["Image_no."])

temp_data = pd.DataFrame()
temp_data.insert(0, "Image_no", np.arange(1,5000))
temp_data.insert(1, "Attractiveness label", np.arange(1,5000))
temp_data.insert(2, "Standard Deviation", np.arange(1,5000))

y_true = new_data['Attractivenss label']
y_true=list(y_true)

y_aug_true = []
for i in range(len(y_true)):
    k = 0
    while(k != 10):
        y_aug_true.append(y_true[i])
        k += 1

#Loading all the images sequence wise
data = 'resized_Aug/'
image_list = {}
images =np.arange(1,501)
image_list[1] = images

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

a = [1,2,3,4,5]
b = [300,600,700,200,100]

img = list(zip(a, b))
img
images = np.zeros(501, dtype='int')
images = list(images)
for i in range(1,501):
    images_of_one_folder = load_images('resized_Aug/SCUT-FBP-'+str(i)+'_resized/')
    #images_of_one_folder = np.array(images_of_one_folder)
    images[i] = images_of_one_folder
    print(i)

#img = load_images('Augumented/SCUT-FBP-100_resized/')

img = np.array(images[1][1])

img = []
for i in range(1, 501):
    for j in range(10):
        img.append(images[i][j])

X_train, X_test, Y_train, Y_test = train_test_split(img, y_aug_true)

arr = np.array(X_train)

print(arr.shape)

#Building Keras Model

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation

input_width = 250
input_height = 190
input_channels = 3
input_pixels = 142500

input_shape = (250,190,3)
n_conv1 = 320
n_conv2 = 640
stride_conv1 = 1
stride_conv2 = 1
conv1_k = 5
conv2_k = 5
max_pool1_k = 5
max_pool2_k = 5

n_hidden = 1000
n_out = 1

input_size_to_hidden = ((input_width//(max_pool1_k*max_pool2_k))*(input_height//(max_pool1_k*max_pool2_k))  * n_conv2)

#Creating a model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(input_size_to_hidden))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_out))


print(model.summary())

model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['accuracy'])

model.fit(arr, Y_train, epochs = 20)
