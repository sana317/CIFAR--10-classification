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
class softmax_FC:
    
    def __init__(self,input_size,output_size):
        #initialization of weights and biases
        self.weights=0.01*np.random.randn(input_size,output_size)
        self.biases=np.zeros((1,output_size))
        
    def forward_propagation(self,input_data):
        #defining the forward propgation in case of softmax
        self.input=input_data
        self.output=np.dot(self.input,self.weights)+ self.biases
        return self.output
        
    def backward_propagation(self,output_error,learning_rate):
        #defining the back propagation in case of softmax
        weights_error=np.dot(self.input.T,output_error)
        self.weights -= learning_rate*weights_error
        self.biases -= learning_rate*np.sum(output_error,axis=0,keepdims=True)
        input_error=np.dot(output_error,self.weights.T)
        return input_error
#this layer is basically providing just the linear output i.e wx+b,we find the cross entropy loss using these scores and
#then directly differentiate the loss with respect to these linear scores
 

