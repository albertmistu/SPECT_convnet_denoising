#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:27:14 2018
@author: Albert Juan Ramon

These functions are used define several network structures
"""
from keras.models import *
from keras.layers import *
from keras import optimizers
from sklearn.model_selection import ParameterGrid

def build_nn_SkipConn(model_info, network_type):
    """
    This function builds and compiles a CAE/CNN network with skip connections given a hash table of the model's parameters.
    :param model_info:
    :return:
    """

    hidden, flt, acts, lay = model_info['Hidden layers'], model_info['Filter sizes'][0], model_info['Activations'], model_info['Num_layers']
    padd, ini, in_sh, name = model_info['Padding'], model_info['Initializer'], model_info['Input Shape'], model_info['Name']

    if network_type == "CAE":

        if lay == 2:
            print('Building 4 layer CAE Skip')
            input_img = Input(shape=(in_sh,in_sh,in_sh,1))  #Variable shape so that we can test it on whole images.
            x1 = Conv3D(lay, (flt,flt, flt), activation = acts[0], padding=padd, kernel_initializer=ini)(input_img)
            x1 = BatchNormalization()(x1)
            encoded=x1
            x3 = Deconvolution3D(1, (flt,flt, flt), activation = acts[1], padding=padd, kernel_initializer=ini)(encoded)
            output =  Add()([input_img,x3])
            model = Model(input_img, output)

        if lay == 4:
            print('Building 4 layer CAE Skip')
            input_img = Input(shape=(in_sh,in_sh,in_sh,1))  #Variable shape so that we can test it on whole images.
            x1 = Conv3D(lay, (flt,flt, flt), activation = acts[0], padding=padd, kernel_initializer=ini)(input_img)
            x1 = BatchNormalization()(x1)
            x2 = Conv3D(lay, (flt,flt, flt), activation = acts[1], padding=padd, kernel_initializer=ini)(x1)
            x2 = BatchNormalization()(x2)
            encoded=x2
            x3 = Deconvolution3D(lay, (flt,flt, flt), activation = acts[2], padding=padd, kernel_initializer=ini)(encoded)
            x3 = BatchNormalization()(x3)
            x3 = Add()([x1, x3])
            x4 = Deconvolution3D(1, (flt,flt, flt), activation = acts[3], padding=padd, kernel_initializer=ini)(x3)
            output =  Add()([input_img,x4])
            model = Model(input_img, output)

        if lay == 6:
            print('Building 6 layer CAE Skip')
            input_img = Input(shape=(in_sh,in_sh,in_sh,1))  #Variable shape so that we can test it on whole images.

            x1 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[0], kernel_initializer=ini)(input_img)
            x1 = BatchNormalization()(x1)
            x2 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[1], kernel_initializer=ini)(x1)
            x2 = BatchNormalization()(x2)
            x3 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[2], kernel_initializer=ini)(x2)
            x3 = BatchNormalization()(x3)
            encoded=x3
            x4 = Deconvolution3D(lay, (flt,flt, flt), padding=padd, activation = acts[3], kernel_initializer=ini)(encoded)
            x4 = BatchNormalization()(x4)
            x4 = Add()([x2, x4])
            x5 = Deconvolution3D(lay, (flt,flt, flt), padding=padd, activation = acts[4], kernel_initializer=ini)(x4)
            x5 = BatchNormalization()(x5)
            x5 = Add()([x1, x5])
            x6 = Deconvolution3D(1, (flt,flt, flt), padding=padd, activation = acts[5], kernel_initializer=ini)(x5)
            output =  Add()([input_img,x6])
            model = Model(input_img, output)


        if lay == 8:
            print('Building 8 layer CAE Skip')
            input_img = Input(shape=(in_sh,in_sh,in_sh,1))  #Variable shape so that we can test it on whole images.

            x1 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[0], kernel_initializer=ini)(input_img)
            x1 = BatchNormalization()(x1)
            x2 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[1], kernel_initializer=ini)(x1)
            x2 = BatchNormalization()(x2)
            x3 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[2], kernel_initializer=ini)(x2)
            x3 = BatchNormalization()(x3)
            x4 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[3], kernel_initializer=ini)(x3)
            x4 = BatchNormalization()(x4)
            encoded=x4
            x5 = Deconvolution3D(lay, (flt,flt, flt), padding=padd, activation = acts[4], kernel_initializer=ini)(encoded)
            x5 = BatchNormalization()(x5)
            x5 = Add()([x3, x5])
            x6 = Deconvolution3D(lay, (flt,flt, flt), padding=padd, activation = acts[5], kernel_initializer=ini)(x5)
            x6 = BatchNormalization()(x6)
            x6 = Add()([x2, x6])
            x7 = Deconvolution3D(lay, (flt,flt, flt), padding=padd, activation = acts[6], kernel_initializer=ini)(x6)
            x7 = BatchNormalization()(x7)
            x7 = Add()([x1, x7])
            x8 = Deconvolution3D(1, (flt,flt, flt), padding=padd, activation = acts[7], kernel_initializer=ini)(x7)
            output =  Add()([input_img,x8])
            model = Model(input_img, output)

        if lay == 10:
            print('Building 10 layer CAE Skip')
            input_img = Input(shape=(in_sh,in_sh,in_sh,1))  #Variable shape so that we can test it on whole images.

            x1 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[0], kernel_initializer=ini)(input_img)
            x1 = BatchNormalization()(x1)
            x2 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[1], kernel_initializer=ini)(x1)
            x2 = BatchNormalization()(x2)
            x3 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[2], kernel_initializer=ini)(x2)
            x3 = BatchNormalization()(x3)
            x4 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[3], kernel_initializer=ini)(x3)
            x4 = BatchNormalization()(x4)
            x5 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[4], kernel_initializer=ini)(x4)
            x5 = BatchNormalization()(x5)
            encoded=x5
            x6 = Deconvolution3D(lay, (flt,flt, flt), padding=padd, activation = acts[5], kernel_initializer=ini)(encoded)
            x6 = BatchNormalization()(x6)
            x6 = Add()([x4, x6])
            x7 = Deconvolution3D(lay, (flt,flt, flt), padding=padd, activation = acts[6], kernel_initializer=ini)(x6)
            x7 = BatchNormalization()(x7)
            x7 = Add()([x3, x7])
            x8 = Deconvolution3D(lay, (flt,flt, flt), padding=padd, activation = acts[7], kernel_initializer=ini)(x7)
            x8 = BatchNormalization()(x8)
            x8 = Add()([x2, x8])
            x9 = Deconvolution3D(lay, (flt,flt, flt), padding=padd, activation = acts[8], kernel_initializer=ini)(x8)
            x9 = BatchNormalization()(x9)
            x9 = Add()([x1, x9])
            x10 = Deconvolution3D(1, (flt,flt, flt), padding=padd, activation = acts[9], kernel_initializer=ini)(x9)
            output =  Add()([input_img,x10])
            model = Model(input_img, output)

    else:
        
        if lay == 2:
            print('Building 4 layer CNN Skip')
        input_img = Input(shape=(in_sh,in_sh,in_sh,1))  #Variable shape so that we can test it on whole images.
        x1 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[0], kernel_initializer=ini)(input_img)
        x1 = BatchNormalization()(x1)
        encoded=x1
        x3 = Conv3D(1, (flt,flt, flt), padding=padd, activation = acts[1], kernel_initializer=ini)(encoded)
        output =  Add()([input_img,x3])
        model = Model(input_img, output)

        if lay == 4:
            print('Building 4 layer CNN Skip')
            input_img = Input(shape=(in_sh,in_sh,in_sh,1))  #Variable shape so that we can test it on whole images.
            x1 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[0], kernel_initializer=ini)(input_img)
            x1 = BatchNormalization()(x1)
            x2 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[1], kernel_initializer=ini)(x1)
            x2 = BatchNormalization()(x2)
            encoded=x2
            x3 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[2], kernel_initializer=ini)(encoded)
            x3 = BatchNormalization()(x3)
            x3 = Add()([x1, x3])
            x4 = Conv3D(1, (flt,flt, flt), padding=padd, activation = acts[3], kernel_initializer=ini)(x3)
            output =  Add()([input_img,x4])
            model = Model(input_img, output)

        if lay == 6:
            print('Building 6 layer CNN Skip')
            input_img = Input(shape=(in_sh,in_sh,in_sh,1))  #Variable shape so that we can test it on whole images.
            x1 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[0], kernel_initializer=ini)(input_img)
            x1 = BatchNormalization()(x1)
            x2 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[1], kernel_initializer=ini)(x1)
            x2 = BatchNormalization()(x2)
            x3 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[2], kernel_initializer=ini)(x2)
            x3 = BatchNormalization()(x3)
            encoded=x3
            x4 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[3], kernel_initializer=ini)(encoded)
            x4 = BatchNormalization()(x4)
            x4 = Add()([x2, x4])
            x5 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[4], kernel_initializer=ini)(x4)
            x5 = BatchNormalization()(x5)
            x5 = Add()([x1, x5])
            x6 = Conv3D(1, (flt,flt, flt), padding=padd, activation = acts[5], kernel_initializer=ini)(x5)
            output =  Add()([input_img,x6])
            model = Model(input_img, output)


        if lay == 8:
            print('Building 8 layer CNN Skip')
            input_img = Input(shape=(in_sh,in_sh,in_sh,1))  #Variable shape so that we can test it on whole images.
            x1 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[0], kernel_initializer=ini)(input_img)
            x1 = BatchNormalization()(x1)
            x2 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[1], kernel_initializer=ini)(x1)
            x2 = BatchNormalization()(x2)
            x3 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[2], kernel_initializer=ini)(x2)
            x3 = BatchNormalization()(x3)
            x4 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[3], kernel_initializer=ini)(x3)
            x4 = BatchNormalization()(x4)
            encoded=x4
            x5 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[4], kernel_initializer=ini)(encoded)
            x5 = BatchNormalization()(x5)
            x5 = Add()([x3, x5])
            x6 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[5], kernel_initializer=ini)(x5)
            x6 = BatchNormalization()(x6)
            x6 = Add()([x2, x6])
            x7 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[6], kernel_initializer=ini)(x6)
            x7 = BatchNormalization()(x7)
            x7 = Add()([x1, x7])
            x8 = Conv3D(1, (flt,flt, flt), padding=padd, activation = acts[7], kernel_initializer=ini)(x7)
            output =  Add()([input_img,x8])
            model = Model(input_img, output)

        if lay == 10:
            print('Building 10 layer CNN Skip')
            input_img = Input(shape=(in_sh,in_sh,in_sh,1))  #Variable shape so that we can test it on whole images.
            x1 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[0], kernel_initializer=ini)(input_img)
            x1 = BatchNormalization()(x1)
            x2 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[1], kernel_initializer=ini)(x1)
            x2 = BatchNormalization()(x2)
            x3 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[2], kernel_initializer=ini)(x2)
            x3 = BatchNormalization()(x3)
            x4 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[3], kernel_initializer=ini)(x3)
            x4 = BatchNormalization()(x4)
            x5 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[4], kernel_initializer=ini)(x4)
            x5 = BatchNormalization()(x5)
            encoded=x5
            x6 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[5], kernel_initializer=ini)(encoded)
            x6 = BatchNormalization()(x6)
            x6 = Add()([x4, x6])
            x7 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[6], kernel_initializer=ini)(x6)
            x7 = BatchNormalization()(x7)
            x7 = Add()([x3, x7])
            x8 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[7], kernel_initializer=ini)(x7)
            x8 = BatchNormalization()(x8)
            x8 = Add()([x2, x8])
            x9 = Conv3D(lay, (flt,flt, flt), padding=padd, activation = acts[8], kernel_initializer=ini)(x8)
            x9 = BatchNormalization()(x9)
            x9 = Add()([x1, x9])
            x10 = Conv3D(1, (flt,flt, flt), padding=padd, activation = acts[9], kernel_initializer=ini)(x9)
            output =  Add()([input_img,x10])
            model = Model(input_img, output)

    if model_info['Optimization'] == 'adagrad':                                 # setting an optimization method
        opt = optimizers.Adagrad(lr = model_info["Learning rate"])
    elif model_info['Optimization'] == 'rmsprop':
        opt = optimizers.RMSprop(lr = model_info["Learning rate"])
    elif model_info['Optimization'] == 'adadelta':
        opt = optimizers.Adadelta()
    elif model_info['Optimization'] == 'adamax':
        opt = optimizers.Adamax(lr = model_info["Learning rate"])
    else:
        opt = optimizers.Adam(lr = model_info["Learning rate"])

    model.compile(optimizer=opt, loss='mean_squared_error')  # compile model

    return model

def create_nns(name, num_layers, num_filters, filter_size, network_type):
    """
    Creates neural network hyperparameter structures to be used as a baseline in determining the influence model depth & width has on performance.
    :param num_layers: list of number of layers to be tested
    :param num_filters: list of number of layers to be tested
    :param filter_size: list of filter sizes to be tested
    :return: list of model_info hash tables
    """
    nns=[]
    model_info = {}
    model_info['Name'] = name
    model_info['Input Shape'] = None
    model_info['Initializer'] = 'he_normal'
    model_info['Optimization'] = 'adam'
    model_info["Learning rate"] = 0.001

    if network_type == 'CAE':
        model_info['Padding'] = 'valid'
    else:
        model_info['Padding'] = 'same'

    param_grid = {'layers': num_layers, 'filters' : num_filters, 'f_sizes': filter_size}
    grid = ParameterGrid(param_grid)

    for params in grid:
        model_info_p = model_info.copy()
        model_info_p["Name"]=model_info['Name'] + '_' + str(params['layers']) + 'l_' + str(params['filters']) + 'f_' + str(params['f_sizes']) + 'fs'
        model_info_p['Num_layers'] = params['layers']
        model_info_p['Num_filters'] = params['filters']
        model_info_p['Filter size'] = params['f_sizes']
        model_info_p['Hidden layers'] =[model_info_p['Num_filters']] * (model_info_p['Num_layers']-1) + [1]
        model_info_p['Filter sizes'] = [model_info_p['Filter size']] * model_info_p['Num_layers']
        model_info_p['Activations'] = ['relu'] * model_info_p['Num_layers']

        nns.append(model_info_p)

    return nns