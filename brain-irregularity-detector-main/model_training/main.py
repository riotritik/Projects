import cv2 
##   read images from no/yes folder
import os 
##   to collect data from folder
from PIL import Image
## new module to process images can use openCV2 as well
import numpy as np
from sklearn.model_selection import train_test_split
## 80% data is used for training and 20% for testing
from keras.utils import normalize
## to normalize data
from keras.utils import to_categorical
## dont import if using binary cross entropy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
## building our model


image_directory='datasets/'

no_tumor_images=os.listdir(image_directory+ 'no/')
'''
# print(no_tumor_images) 
# print all images
# path='no0.jpg'
# print(path.split('.')[1]) 
# # to check if jpg or not
'''

dataset=[]
label=[]

for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        ## image read
        image=cv2.imread(image_directory+'no/'+image_name)
        ## default is BGR for open cv 
        ## we use PIL can also use openCV
        image=Image.fromarray(image,'RGB') 
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0) #no brain tumor

yes_tumor_images=os.listdir(image_directory+ 'yes/')
for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1) #yes brain tumor

'''        
# print(dataset)
# print(label)
## total 3000 images
# print(len(dataset))
# print(len(label))
'''

dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)
### 1st and 2nd arrays, 3rd test size(1=100%)
'''
# print(x_train.shape) 
# # output= (2400, 64, 64, 3) -> number of train images, sixe=64x64, 3= channels(RGB)
# # same as ^ Reshape = (n, image_width, image_height, n_channel)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)
'''

## normalize data to train machine
## the practice of organizing data entries to ensure they appear similar across all fields and records, making information easier to find, group and analyze
x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)


## emit this if using binary cross entropy
y_train=to_categorical(y_train , num_classes=2)
y_test=to_categorical(y_test , num_classes=2)


# Model Building
# 64,64,3

model=Sequential()

model.add(Conv2D(32, (3,3), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
# Binary CrossEntropy= 1, sigmoid
# Categorical Cross Entryopy= 2 , softmax
model.add(Activation('softmax'))


# Binary CrossEntropy= 1, sigmoid
# Categorical Cross Entryopy= 2 , softmax

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, 
    batch_size=16, 
    verbose=1, epochs=10, 
    validation_data=(x_test, y_test),
    shuffle=False)

model.save('Brain_tumor_model.h5')