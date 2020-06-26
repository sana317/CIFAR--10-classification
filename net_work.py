#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy as np
import import_ipynb
from cifar10 import*
from cifar10_5classes import*
import numpy as np
from skimage.feature import hog
import pandas as pd
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
class Network:
    
    def __init__(self):
        self.layers=[]
        
    
    def add(self,layer):
        self.layers.append(layer)
    
    def predict(self, input_data):
        samples = len(input_data)
        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)
        

        return output
        
        
    def fit(self,x_train,y_train,learning_rate):
        samples=len(x_train)
        #defining the batch size of 50 for batch gradient descent
        batch_size=50
        #running the training for 300 epochs 
        for i in range(300):
            batch=0
            while batch<10000:
                batch_initial=batch
                batch_final=batch+batch_size
                output=x_train[batch_initial:batch_final,:]
                batch=batch+batch_size
        
                for layer in self.layers:
                    output=layer.forward_propagation(output)
                #calculating the exponential scores after getting the linear scores after using softmax layer
                soft_max_scores = np.exp(output-np.max(output, axis=1, keepdims=True))
                prob_soft = soft_max_scores / np.sum(soft_max_scores, axis=1, keepdims=True)
                #calculation of cross entropy loss
                log_prob_soft = -np.log(prob_soft[range(batch_size),y_train[batch_initial:batch_final]])
                log_prob_soft=np.array([log_prob_soft]).T
                
                data_loss = np.sum(log_prob_soft)/batch_size
                loss=data_loss
                #doing backpropgation
                #finding gradient with respect to the loss
                dscores=prob_soft
                dscores[range(batch_size),y_train[batch_initial:batch_final]]-=1
                dscores/=batch_size
                error=dscores
                for layer in reversed(self.layers):
                    error=layer.backward_propagation(error,learning_rate)
            
            print('loss of', i, 'is', loss)
            

