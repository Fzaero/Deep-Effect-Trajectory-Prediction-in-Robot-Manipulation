#!/usr/bin/env python

from keras.layers import Input,Dense,LSTM,RepeatVector,concatenate,Conv2D,MaxPooling2D,Dropout,Flatten,UpSampling2D
from keras.models import Model,Sequential

import keras

def model_type_1(n_neurons,seq_length, featureSize):
	model = Sequential()
	model.add(LSTM(n_neurons, input_shape=(seq_length, featureSize)))
	model.add(Dense(6,activation='linear'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def model_type_2(n_neurons,seq_length, featureSize):
	supports = Input(shape=(FeatureSize-6,),name='supports')
	traj = Input(shape=(seq_length,6,),name='traj')
    
	supports_Dense = Dense(1000,activation='relu')(supports)
	supports_Dense =  Dropout(0.25)(supports_Dense)

	repeated_supp = RepeatVector(seq_length)(supports_Dense)
	concat_layer = concatenate([traj, repeated_supp])
	lstm_layer = LSTM(n_neurons)(concat_layer)
	output = Dense(6,activation='linear',name='output')(lstm_layer)
	model = Model(inputs=[supports, traj], outputs=[output])
	model.compile(optimizer='adam',
                  loss={'output': 'mean_squared_error'})
	return model
                  
def model_type_3(n_neurons,seq_length, featureSize):
	images = Input(shape=(128,128,1,),name='supports')

	x = Conv2D(32, (3, 3), activation='relu', padding='same')(images)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)

	flattened= Flatten()(encoded)

	supports_Dense = Dense(1000,activation='relu')(flattened)
	supports_Dense = Dropout(0.25)(supports_Dense)

	traj = Input(shape=(seq_length,6,),name='traj')
	noise = Input(shape=(seq_length,10,),name='noise')

	repeated_supp = RepeatVector(seq_length)(supports_Dense)
	concat_layer = concatenate([traj, repeated_supp,noise])
	lstm_layer = LSTM(n_neurons)(concat_layer)
	output = Dense(6,activation='linear',name='output')(lstm_layer)
	model = Model(inputs=[images, traj,noise], outputs=[output])
	model.compile(optimizer='adam',
                  loss={'output': 'mean_absolute_error'})
	return model           
