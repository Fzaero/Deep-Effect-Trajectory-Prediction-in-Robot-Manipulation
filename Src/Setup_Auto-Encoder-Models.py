#!/usr/bin/env python
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

def Get_Auto_Encoder_Model(mid_filter_size):
	input_img = Input(shape=(128, 128, 1)) 

	x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(mid_filter_size, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)

	x = Conv2D(mid_filter_size, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

	encoder = Model(input_img, encoded)
	autoencoder = Model(input_img, decoded)

	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	return encoder,autoencoder
images = np.zeros((2080,128,128,1))

directory = 'Shapes/'
imageCount = 0
for gen in range(3,11):
	for shapeID in range(20):
		for edgeNum in range(gen):
			for experimentNum in range(2):
				images[imageCount] = image.img_to_array(image.load_img(directory+str(gen)+'gen/shape'+str(shapeID)+'/edge'+str(edgeNum)+'/experiment'+str(experimentNum)+'/image.jpg', grayscale=True).resize((128,128)))
				imageCount = imageCount + 1 	

np.random.shuffle(images)
x_train = (images[:2080*80/100].astype('float32') / 255. / 0.5).astype('int32')
x_test = (images[2080*80/100:].astype('float32') / 255. / 0.5).astype('int32')

for filter_size in [4,8,16]:
	encoder,autoencoder=Get_Auto_Encoder_Model(filter_size)
	autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=8,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
	autoencoder.save('88'+str(filter_size)+'autoencoder.h5')
	encoder.save('88'+str(filter_size)+'encoder.h5')
