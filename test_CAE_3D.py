#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:15:48 2017
@author: Albert Juan Ramon
Main script for testing trained networks and saving results
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras import backend as K
import tensorflow as tf
from build_NN import create_nns, build_nn_SkipConn
from save_DICOM_data import save_DICOM_dataset_3D

""" Define network structure (using best one found in structure optimization step from [1])"""
model_name = 'FBP_CAE_Skip_3D'
network_type = 'CAE'
# model_name = 'FBP_CNN_Skip_3D'  
# network_type = 'CNN'

num_layers, num_filters, filter_size =  [4], [10], [3]
test_dose, max_epochs = '1_2', 300                          #Testing on 50% dose.

""" Create network structure for given parameters """
nns_list = create_nns(model_name, num_layers, num_filters, filter_size, network_type)
model_info = nns_list[0]

""" Define path with patient VOI masks """
masks_path =  "data/masks"

""" Define path with test data"""
L_H_T = "data/test/FBP_50%_0.22/"

""" Define paths for loading weights"""
model_name_ext =   model_info['Name'] + '_' + str(max_epochs) + '_epoch_' + test_dose
pweight = './weights/weights_' + model_name_ext  + '.hdf5'

""" Load network weights"""
network = build_nn_SkipConn(nns_list[0], network_type) 
network.summary()
network.load_weights(pweight)

""" Test on patients in test folder: """  
lstFilesDCM = os.listdir(L_H_T)
for ix, filenameDCM in enumerate(lstFilesDCM):
    
    mask = scipy.io.loadmat(os.path.join(masks_path, filenameDCM.strip('.dcm') + '_MASK.mat')); # Load VOI mask
    mask = mask['masqi']
        
    # Load DICOM data and predict image using loaded network.
    ds = dicom.read_file(os.path.join(L_H_T,filenameDCM)) 
    X = ds.pixel_array
    X = X/np.max(X[mask!=0])
    Y = network.predict(np.reshape(X,[1, X.shape[0], X.shape[1], X.shape[2],1]))
