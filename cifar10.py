#!/usr/bin/env python
# coding: utf-8

# In[1]:


#function for reading dataset for two classes
def dataset(class1,class2):
    import numpy as np
    import pandas as pd
    from six.moves import cPickle as pickle
    import matplotlib.pyplot as plt
    from skimage.feature import hog
    from skimage import color

#function to return a dictionary consisting of data,labels and data_names    
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

#separation of labels and data for batch 1 consisting of 10000 images
    b1=unpickle('data_batch_1')
    b1_data= b1[b'data']
    b1_data=np.array(b1_data)
    b1_labels= b1[b'labels']
    b1_labels=np.array([b1_labels]).T

#separation of labels and data for batch 2 consisting of 10000 images    
    b2=unpickle('data_batch_2')
    b2_data= b2[b'data']
    b2_data=np.array(b2_data)
    b2_labels= b2[b'labels']
    b2_labels=np.array([b2_labels]).T
    
#separation of labels and data for batch 3 consisting of 10000 images
    b3=unpickle('data_batch_3')
    b3_data= b3[b'data']
    b3_data=np.array(b3_data)
    b3_labels= b3[b'labels']
    b3_labels=np.array([b3_labels]).T
    
#separation of labels and data for batch 4 consisting of 10000 images
    b4=unpickle('data_batch_4')
    b4_data= b4[b'data']
    b4_data=np.array(b4_data)
    b4_labels= b4[b'labels']
    b4_labels=np.array([b4_labels]).T
    
#separation of labels and data for batch 5 consisting of 10000 images
    b5=unpickle('data_batch_5')
    b5_data= b5[b'data']
    b5_data=np.array(b5_data)
    b5_labels= b5[b'labels']
    b5_labels=np.array([b5_labels]).T
    
#separation of labels and data for test batch consisting of 10000 images
    test=unpickle('test_batch')
    test_data= test[b'data']
    test_data=np.array(test_data)
    test_labels= test[b'labels']
    test_labels=np.array([test_labels]).T
    x_test=test_data
    y_test=test_labels

#concatenation of data and labels to get a training set of 50000 images
    x_train=np.concatenate((b1_data, b2_data,b3_data,b4_data,b5_data), axis=0,out=None)
    y_train=np.concatenate((b1_labels, b2_labels,b3_labels,b4_labels,b5_labels), axis=0,out=None)
    

#separation of images of class1 and class2 so as to get total 10000  training images
    ro_train=[]
    for i in range(len(x_train)):
        if y_train[i]==class1 or y_train[i]==class2:
            ro_train.append(i)
    x_train1=np.array(x_train[ro_train])
    y_train1=np.array(y_train[ro_train])
    y_train2=[0 if i==class1 else 1 for i in y_train1 ]

#separation of images of class1 and class2 so as to get total 2000  test images
    ro_test=[]
    for i in range(len(x_test)):
        if y_test[i]==class1 or y_test[i]==class2:
            ro_test.append(i)
    x_test1=np.array(x_test[ro_test])
    y_test1=np.array(y_test[ro_test])
    y_test2=[0 if i==class1 else 1 for i in y_test1 ]


#converting the matrix of 10000x3072 to 10000x32x32x3    
    l=len(x_train1)
    x_train1=x_train1.reshape((l,3,32,32))
    x_train2=np.rollaxis(x_train1,1,4)
    l1=len(x_test1)
    x_test1=x_test1.reshape((l1,3,32,32))
    x_test2=np.rollaxis(x_test1,1,4)

    hog_images=[]
    hog_features=[]
    
#extraction of 144 hog features for training and test data
    for i in range(len(x_train2)):
        fd,hog_image =hog(x_train2[i], orientations=9, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True, multichannel=True)
        hog_images.append(hog_image)
        hog_features.append(fd)

    hog_features=np.array(hog_features)
    

    hog_images_test=[]
    hog_features_test=[]

    for i in range(len(x_test2)):
        fd_test,hog_image_test =hog(x_test2[i], orientations=9, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True, multichannel=True)
        hog_images_test.append(hog_image_test)
        hog_features_test.append(fd_test)

    hog_features_test=np.array(hog_features_test)
    
    return hog_features,y_train2,hog_features_test,y_test2  

