#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
os.environ['CUDA_VISIBLE_DEVICES']='1'


MODEL_NAME = 'keras_baseline_EffNB3_stu_freeze_V4.h5'
PRE_FILE = "my_test.csv"
save_file = True
SAVE_FILE = "my_test_TTA_B3_freeze_V4.csv"
get_AUC = False


# In[13]:


def focal_loss(alpha=0.25,gamma=2.0):
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

model = load_model(MODEL_NAME, custom_objects={'focal_crossentropy': focal_crossentropy})


# In[14]:


model.summary()


# In[15]:


BASE_PATH = "/root/"
val_df = pd.read_csv(BASE_PATH + PRE_FILE)
#val_df = pd.read_csv(BASE_PATH + "my_val.csv")
val_df.head()


# In[16]:


val_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range= 20 ,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# In[17]:


val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='image_name',
    target_size=(256, 256),
    batch_size=1,
    shuffle=False,
    class_mode='raw'
)
# batch_size most to be 1


# In[18]:


pre1 = model.predict(val_generator)
print('1')
pre2 = model.predict(val_generator)
print('2')
pre3 = model.predict(val_generator)
print('3')


# In[19]:


pre_all = np.hstack([pre1,pre2,pre3])
pre_all_tbl = pd.DataFrame(data = pre_all)
print(pre_all_tbl.head())


# In[20]:


target = np.mean(pre_all, 1)


# In[21]:


# fpr, tpr, thresholds = metrics.roc_curve(val_df.target,pre, pos_label=1)
# metrics.auc(fpr, tpr)
# pre


# In[22]:


if get_AUC == True:
    ans = val_df.target


# In[23]:


# target=[]
# for path in val_df['image_path']:
#     img = cv2.imread(str(path))
#     img = cv2.resize(img, (224,224))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32)/255.
#     img=np.reshape(img,(1,224,224,3))
#     prediction=model.predict(img)
#     target.append(prediction[0][0])

# # target


# In[24]:


res = val_df
res['target'] = target
res = res[['image_name','target']]
res.head()
if save_file == True:
    res.to_csv(SAVE_FILE,index=False)
# # res.to_csv(SAVE_FILE)


# In[25]:


if get_AUC == True:
    fpr, tpr, thresholds = metrics.roc_curve(np.array(ans),target, pos_label=1)
    my_auc = metrics.auc(fpr, tpr)
    print(MODEL_NAME)
    print(PRE_FILE)
    print(my_auc)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




