
# coding: utf-8

# ## Load necessary modules

# In[1]:


# # show images inline
# get_ipython().run_line_magic('matplotlib', 'inline')

# automatically reload modules when they have changed
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


# ## Load RetinaNet model

# In[2]:


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('res101.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
names_to_labels = { 'aeroplane'   : 0,
                     'bicycle'     : 1,
                     'bird'        : 2,
                     'boat'        : 3,
                     'bottle'      : 4,
                     'bus'         : 5,
                     'car'         : 6,
                     'cat'         : 7,
                     'chair'       : 8,
                     'cow'         : 9,
                     'diningtable' : 10,
                     'dog'         : 11,
                     'horse'       : 12,
                     'motorbike'   : 13,
                     'person'      : 14,
                     'pottedplant' : 15,
                     'sheep'       : 16,
                     'sofa'        : 17,
                     'train'       : 18,
                     'tvmonitor'   : 19,
                     'face'        : 20
 }

#names_to_labels={
#    'person':0
#}


labels_to_names = {v: k for k, v in names_to_labels.items()}

# ## Run detection on example

# In[3]:

num=1
img_paths=os.listdir('images')
for img_path in img_paths:
 print(img_path)
 start = time.time()
 # load image
 image = read_image_bgr('images/'+img_path) 


 # copy to draw on
 draw = image.copy()
 draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB) 

 # preprocess image for network
 image = preprocess_image(image)
 image, scale = resize_image(image)
 # process image
 boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
 print("processing time: ", time.time() - start)

 # correct for image scale
 boxes /= scale

 # visualize detections
 for box, score, label in zip(boxes[0], scores[0], labels[0]):
     # scores are sorted so we can break
     if score < 0.5:
         break
        
     color = label_color(label)
    
     b = box.astype(int)
     draw_box(draw, b, color=color)
    
     caption = "{} {:.3f}".format(labels_to_names[label], score)
     draw_caption(draw, b, caption)
    
 cv2.imwrite("results/"+str(num)+".jpg",draw)
 num+=1
#plt.figure(figsize=(15, 15))
#plt.axis('off')
#plt.imshow(draw)
#plt.show()

