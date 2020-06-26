#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy as np
import import_ipynb
from cifar10 import*
from relu import*
from softmax import*
from net_work import*
from cifar10_5classes import*
import numpy as np
from skimage.feature import hog
import pandas as pd
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

#reading the dataset

(x_train),(y_train),(x_test),(y_test)=dataset(6,9) 
#(x_train),(y_train),(x_test),(y_test)=cifar_5_dataset(5,6,7,8,9)
x_train = x_train.astype('float32')    
x_test = x_test.astype('float32')


# In[2]:


step_size=0.01
net=Network()
net.add(Relu_FC(144,100))
net.add(Relu_FC(100,64))
net.add(softmax_FC(64,5))
net.fit(x_train,y_train,step_size)
#total number of parameters in case of 5 classes used are 144x100+100x64+64x5+100+64+5=21289
#total number of parameters in case of 2 classes used are 144x100+100x64+64x2+100+64+2=20928


# In[3]:


out = net.predict(x_test)
soft_max_scores_test = np.exp(out-np.max(out, axis=1, keepdims=True))   
prob_soft_test = soft_max_scores_test / np.sum(soft_max_scores_test, axis=1, keepdims=True)
predicted_class = np.argmax(prob_soft_test, axis=1)
print ('accuracy is',np.mean(predicted_class == y_test)*100)
        


# In[ ]:




