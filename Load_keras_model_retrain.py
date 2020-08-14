#!/usr/bin/env python
# coding: utf-8

# In[52]:


from sklearn import metrics
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation
from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import load_model
import efficientnet.keras as efn 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# In[42]:


def focal_loss(alpha=0.25,gamma=3.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy


def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)

import efficientnet.tfkeras
from tensorflow.keras.models import load_model

model = load_model('keras_baseline_EffNB3_stu_freeze_V4.h5', custom_objects={'focal_crossentropy': focal_crossentropy})


for uu in range(100):
    model.layers[uu].trainable = True

# In[43]:


model.summary()


# In[64]:


BASE_PATH = "/root/"
val_df = pd.read_csv(BASE_PATH + "my_train.csv")
val_df.head()


# In[65]:


val_datagen=ImageDataGenerator(
    rescale=1./255,rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# In[90]:


val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='target',
    target_size=(256, 256),
    batch_size=32,
    shuffle=True,
    class_mode='raw')


# In[91]:


val_df['target'].value_counts()


# In[92]:


opt = Adam(lr=1e-5)
model.compile(loss=focal_loss(), metrics=[tf.keras.metrics.AUC()],optimizer=opt)


# In[93]:


# cb=[PlotLossesKeras()]
model.fit_generator(
    val_generator,
    #steps_per_epoch=nb_train_steps,
    epochs = 20#,
    #validation_data=validation_generator,
    #callbacks=cb,
    #validation_steps=nb_val_steps
)

model.save("keras_baseline_EffNB3_stu_freeze_V5.h5")


# In[94]:


# pre = model.predict(val_generator)


# In[96]:


# fpr, tpr, thresholds = metrics.roc_curve(val_df.target,pre, pos_label=1)
# metrics.auc(fpr, tpr)


# In[97]:


# pre


# In[ ]:




