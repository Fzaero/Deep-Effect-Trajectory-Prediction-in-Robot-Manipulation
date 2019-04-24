#!/usr/bin/env python

from keras.layers import Input,Dense,LSTM,RepeatVector,concatenate,Conv2D,MaxPooling2D,Dropout,Flatten,UpSampling2D,TimeDistributed,CuDNNLSTM,Lambda
from keras.models import Model,Sequential
import tensorflow as tf
import keras
import numpy as np
from keras import regularizers

def model_type_1(n_neurons,seq_length, featureSize,lear=0.001,n_step=1):
	model = Sequential()
	model.add(CuDNNLSTM(n_neurons,return_sequences=True, input_shape=(seq_length, featureSize)))
	model.add(TimeDistributed(Dense(6*n_step,activation='linear')))
	adam = keras.optimizers.Adam(lr=lear)
	model.compile(loss='mean_squared_error',optimizer=adam)
	return model

def model_type_2(n_neurons,seq_length, featureSize,dropout=0.5,layer_size=1024,lear=0.001,n_step=1):
	supports = Input(shape=(featureSize-6,),name='supports')
	traj = Input(shape=(seq_length,6,),name='traj')
    
	supports_Dense = Dense(layer_size,activation='relu')(supports)
	supports_Dense =  Dropout(dropout)(supports_Dense)

	repeated_supp = RepeatVector(seq_length)(supports_Dense)
	concat_layer = concatenate([traj, repeated_supp])
	lstm_layer = CuDNNLSTM(n_neurons,return_sequences=True)(concat_layer)
	output = TimeDistributed(Dense(6*n_step,activation='linear'),name='output')(lstm_layer)
	model = Model(inputs=[supports, traj], outputs=[output])
	adam = keras.optimizers.Adam(lr=lear)
	model.compile(optimizer=adam,
                  loss={'output': 'mean_squared_error'})
	return model
                  
def model_type_3(n_neurons,seq_length, featureSize,dropout=0.5,layer_size=1024,lear=0.001,n_step=1):
	images = Input(shape=(128,128,1,),name='supports')
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(images)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)

	flattened= Flatten()(encoded)

	supports_Dense = Dense(layer_size,activation='relu')(flattened)
	supports_Dense = Dropout(0.5)(supports_Dense)

	traj = Input(shape=(seq_length,6,),name='traj')

	repeated_supp = RepeatVector(seq_length)(supports_Dense)
	concat_layer = concatenate([traj, repeated_supp])
	lstm_layer = CuDNNLSTM(n_neurons,return_sequences=True)(concat_layer)
	output = TimeDistributed(Dense(6*n_step,activation='linear'),name='output')(lstm_layer)
	adam = keras.optimizers.Adam(lr=lear)
	model = Model(inputs=[images, traj], outputs=[output,])
	model.compile(optimizer=adam,loss={'output': 'mean_squared_error'})
	return model           
