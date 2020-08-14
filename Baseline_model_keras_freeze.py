#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras import backend as K
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# In[ ]:


nb_epochs = 10
batch_size= 64
#nb_train_steps = train_df.shape[0]//batch_size
# nb_val_steps=validation.shape[0]//batch_size
# print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))


# In[17]:


BASE_PATH = "/root/"
# train_df = pd.read_csv(BASE_PATH + "train.csv")
# test_df = pd.read_csv(BASE_PATH + "test.csv")

# train_df["image_path"] = train_df.apply(lambda x: str(
#     BASE_PATH + "jpeg/" + "train/" + f"{x.image_name}.jpg"), axis=1)
# test_df["image_path"] = test_df.apply(lambda x: str(
#     BASE_PATH + "jpeg/" + "test/" + f"{x.image_name}.jpg"), axis=1)

train_df = pd.read_csv(BASE_PATH + "my_train.csv")
train_df.head()


# In[18]:


train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,horizontal_flip=True)

val_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='target',
    target_size=(256, 256),
    batch_size=batch_size,
    shuffle=True,
    class_mode='raw')


# In[19]:


train_df['target'].value_counts()


# In[20]:


# !pip install efficientnet


# In[21]:


import efficientnet.keras as efn 
# model = efn.EfficientNetB0(weights='noisy-student')


# In[22]:


def EfficientNetB7_model( num_classes=None):

    model = efn.EfficientNetB3(weights='noisy-student', include_top=False, input_shape=(256, 256, 3))
    model.trainable = False
    x=Flatten()(model.output)
    output=Dense(1,activation='sigmoid')(x) # because we have to predict the AUC
    model=Model(model.input,output)
    
    return model

Eff_conv=EfficientNetB7_model(1)


# In[23]:


Eff_conv.summary()


# In[24]:


def focal_loss(alpha= 0.25,gamma= 3.0):
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


# In[25]:


opt = Adam(lr=1e-5)
Eff_conv.compile(loss=focal_loss(), metrics=[tf.keras.metrics.AUC()],optimizer=opt)


# In[ ]:





# In[26]:


# cb=[PlotLossesKeras()]
Eff_conv.fit_generator(
    train_generator,
    #steps_per_epoch=nb_train_steps,
    epochs=nb_epochs#,
    #validation_data=validation_generator,
    #callbacks=cb,
    #validation_steps=nb_val_steps
)


# In[ ]:


Eff_conv.save("keras_baseline_EffNB3_stu_freeze.h5")


# In[ ]:


# target=[]
# for path in train_df['image_path']:
#     img=cv2.imread(str(path))
#     img = cv2.resize(img, (224,224))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32)/255.
#     img=np.reshape(img,(1,224,224,3))
#     prediction=Eff_conv.predict(img)
#     target.append(prediction[0][0])

# target
    


# In[ ]:





# In[ ]:





# In[ ]:




