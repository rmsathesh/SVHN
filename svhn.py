# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:53:54 2021

@author: hp
"""

import numpy as np
import tarfile
import os
import scipy.io
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, BatchNormalization, Input, Conv2DTranspose, concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split
import random
#%%
fpath = os.path.dirname(__file__)
os.chdir(fpath)


#%%
def create_model():
    inputs = Input((32, 32, 1))
    
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPool2D((2, 2)) (c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPool2D((2, 2)) (c2)
    
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPool2D((2, 2)) (c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = Flatten()(c4)
    
    d1 = Dense(units=1024, activation='relu')(p4)
    d1 = BatchNormalization()(d1)
    d2 = Dense(units=1024, activation='relu')(d1)
    
    o1 = Dense(units=11, activation='softmax')(d2)
    o2 = Dense(units=11, activation='softmax')(d2)
    o3 = Dense(units=11, activation='softmax')(d2)
    o4 = Dense(units=11, activation='softmax')(d2)
    o5 = Dense(units=11, activation='softmax')(d2)
    
    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5])
    
    return model
#%%

model = create_model()
print(model.summary())

#%%
with open('SVHN_data.pickle', 'rb') as f:
    tmp = pickle.load(f)
    train_dataset = tmp['train_dataset']
    valid_dataset = tmp['valid_dataset']
    train_labels = tmp['train_labels']
    valid_labels = tmp['valid_labels']
    del tmp
          
          
#%%

loss = SparseCategoricalCrossentropy()   
opti = Adam(learning_rate=0.0001)
metrics=['accuracy']
model.compile(loss=loss, optimizer=opti, metrics=metrics)
#%%   
def train_generator(train_dataset, train_labels, batch_size=16, seed=1):
    train_gen = ImageDataGenerator(rescale = 1/255., rotation_range=90, width_shift_range=0.05, height_shift_range=0.05,
                                       shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='wrap')
    
    gen = train_gen.flow(train_dataset, train_labels, batch_size=batch_size)
    while True:
        image_, label_ = gen.next()
        
        yield image_, np.hsplit(label_, 5)
        
def valid_generator(valid_dataset, valid_labels, batch_size=16, seed=1):
    valid_gen = ImageDataGenerator(rescale = 1/255.)
    
    gen = valid_gen.flow(valid_dataset, valid_labels, batch_size=batch_size)
    while True:
        image_, label_ = gen.next()
        
        yield image_, np.hsplit(label_, 5)     



#%%
    
train_data = train_generator(train_dataset, train_labels, batch_size=32)
valid_data = valid_generator(valid_dataset, valid_labels, batch_size=32)
model.fit_generator(train_data, validation_data=valid_data, validation_steps=10, steps_per_epoch=500, epochs=5)    
        
#%%
model.save('svhn.h5')


#%%
model_final = load_model('svhn.h5', compile=False)
print(model_final.summary())


#%%


    
test_gen =ImageDataGenerator(rescale=1/255.)

test_generator = test_gen.flow_from_directory(directory=r'test_mod/', target_size=(32,32), color_mode='grayscale', batch_size=1, class_mode=None, shuffle=False)

#%%
i = 0
for image in test_generator:
    pred = model_final.predict(image)
    pred = np.array(pred)
    pred = np.transpose(pred, (1, 0, 2))
    pred = np.argmax(pred, axis=2)[0]
    plt.imshow(image[0,:,:,0])
    plt.title(pred)
    plt.axis('off')
    plt.show()
    i += 1
    if i==15:
        break
#%%
i = 0
for image_, label_ in valid_generator(train_dataset, train_labels, batch_size=1):
    pred = model_final.predict(image_)
    pred = np.array(pred)
    pred = np.transpose(pred, (1, 0, 2))
    pred = np.argmax(pred, axis=2)[0]
    plt.imshow(image_[0,:,:,0])
    plt.title(pred)
    plt.axis('off')
    plt.show()
    i += 1
    if i==10:
        break
    
