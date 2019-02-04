#!/usr/bin/env python

from os import listdir
from os.path import isfile, join
from keras.models import load_model
from keras.preprocessing import image
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
import keras
import copy
import math
import numpy as np
import scipy.stats as stats
import pylab as pl
from sklearn.externals import joblib
dataset_path='trajectories/'
scalers_path='scalers/'
num_of_Data=1857

def_traj_lenght=74
started_moving_threshold=0.0000001
object_slip_threshold=0.03
def get_data_trajectories():
	onlyfiles = [str(s) + '.txt' for s in range(1,num_of_Data)]
	traj_files=[dataset_path+f for f in onlyfiles]
	dataframe = [np.loadtxt(traj).astype('float32') for traj in traj_files]
	series=list()
	for i in range(len(dataframe)):   
		trajectory = list()
		trajectory.append(dataframe[i][0,:])
		for t in range(def_traj_lenght):
			if sum((dataframe[i][t+1,:3]-dataframe[i][t,:3])**2)<started_moving_threshold:
				continue
			## If absolute difference is bigger than object_slip_threshold, object may slipped from end of tool.
			if abs(dataframe[i][t+1,2]-dataframe[i][t,2])>object_slip_threshold:
				break			
			## If difference is smaller than 0, object may slipped from end of tool.
			if dataframe[i][t+1,2]-dataframe[i][t,2]<0:
				break
			## If rotation-z difference is bigger than 0.40 radian, it means object slipped from end of tool. 
			if abs(dataframe[i][t+1,5])>0.40:
				break
			trajectory.append(dataframe[i][t+1,:])
		trajectorynp=np.array(trajectory)
		series.append(trajectorynp)
	return series
def get_data_supportPoint():
	supportPoints = np.loadtxt('supportPoints.txt')
	return supportPoints
def get_data_images():
	onlyfiles2 = [str(s) + '.jpg' for s in range(1,1857)]
	image_files=['images/'+f for f in onlyfiles2]
	images= np.zeros((num_of_Data-1,128,128,1))
	for i in range(num_of_Data-1):
		images[i] = image.img_to_array(image.load_img(image_files[i], grayscale=True).resize((128,128)))
	return images
def get_data_shapeContext(ring_wedges):
	shapeContexts = np.loadtxt('shapeContexts'+str(ring_wedges)+'.txt')
	return shapeContexts
def get_data_encoder(mid_filter_size,images):
	model_encoder = load_model('88'+str(mid_filter_size)+'encoder.h5')
	images_scaled = (images.astype('float32') / 255. / 0.5).astype('int32')
	autoEncoderFeatures= model_encoder.predict(images_scaled).reshape((num_of_Data-1,8*8*mid_filter_size))
	return autoEncoderFeatures

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, trajectoryIndex, imageAddresses,scaler_x,series,seq_length=15 ,batch_size=100, n_channels=1, shuffle=True):
		'Initialization'
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.series = series
		self.n_channels = n_channels
		self.shuffle = shuffle
		self.trajectoryIndexes=trajectoryIndex
		self.imageAddresses=imageAddresses
		self.scaler_x=scaler_x
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.trajectoryIndexes) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		trajectory_ids_temps = [self.trajectoryIndexes[k] for k in indexes]
		imageAddresses_temps = [self.imageAddresses[k] for k in indexes]
        
        # Generate data
		X, y = self.__data_generation(trajectory_ids_temps,imageAddresses_temps)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.trajectoryIndexes))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, trajectory_ids_temps,imageAddresses_temps):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		x_traj = np.empty((self.batch_size, self.seq_length,6))
		x_images = np.empty((self.batch_size,128,128,1))
		x_noise = np.random.normal(0, 0.0001,(self.batch_size,self.seq_length,10))
		y = np.empty((self.batch_size,6))
		#(1,6)
		# Generate data
		for i, ID in enumerate(trajectory_ids_temps):
			traj=self.series[ID[0]]
			t=ID[1]
			randomASD= np.random.normal(0, 0.0001, 6)
			for tt in range(self.seq_length):
				if(t-self.seq_length+tt<0):
					x_traj[i,tt,:]=self.scaler_x.transform(traj[0:1,:]+randomASD)
				else:
					x_traj[i,tt,:]=self.scaler_x.transform(np.random.normal(0, 0.0001, 6) + traj[t-self.seq_length+tt:t-self.seq_length+1+tt,:])
			y[i,:] = self.scaler_x.transform(traj[t:t+1,:] + np.random.normal(0, 0.0001, 6))[0,:]
		for i, address in enumerate(imageAddresses_temps):
			x_images[i,] = image.img_to_array(image.load_img(address, color_mode = "grayscale").resize((128,128)))
		x_images= (x_images.astype('float32') / 255. / 0.5).astype('int32')
		return {'supports': x_images, 'traj': x_traj, 'noise': x_noise},{'output': y}

def get_data_for_model(modelNumber,features,FeatureSize,seq_length=0):
	#features= 0:No Features,1:Support Points,2:ShapeContext,3:AutoEncoder,4:Image
	#FeatureSize= No Feature:0,Support Points:4, ShapeContext:110,420,1640,AutoEncoder:256,512,1024 Image:(128,128,1)
	indexes = np.zeros(1799).astype('int')
	indexFile = open('indexes.txt','r')
	lines = indexFile.readlines()
	for i in range(len(lines)):
		indexes[i] = int(lines[i].split()[0])
	training=1400 #1400-1500 validation.
	series=get_data_trajectories()
	seriesnp=np.concatenate( [series[i] for i in indexes[:training]], axis=0 )
	scaler_x=scaler = MinMaxScaler()
	scaler_x.fit(seriesnp)
	if features==1:
		supportPoints = get_data_supportPoint()
		supportPointsnp=np.array([supportPoints[i,:] for i in indexes[:training]])
		scaler_sp = MinMaxScaler()
		scaler_sp.fit(supportPointsnp)
		return series,supportPoints,scaler_x,scaler_sp,indexes
	elif features==2:
		if FeatureSize==110:
			rings_wedges=200
		if FeatureSize==420:
			rings_wedges=800
		if FeatureSize==1640:
			rings_wedges=1600
		shapeContexts=get_data_shapeContext(rings_wedges)
		shapeContextsnp=np.array([shapeContexts[i,:] for i in indexes[:training]])	
		scaler_sc = MinMaxScaler()
		scaler_sc.fit(shapeContextsnp)
		return series,shapeContexts,scaler_x,scaler_sc,indexes   
	elif features==3:
		images = get_data_images()
		autoEncoderFeatures= get_data_encoder(FeatureSize/64,images)
		autoEncoderFeaturesnp=np.array([autoEncoderFeatures[i,:] for i in indexes[:training]])
		scaler_ae = MinMaxScaler()
		scaler_ae.fit(autoEncoderFeaturesnp)
		return series,autoEncoderFeatures,scaler_x,scaler_ae,indexes    
	if modelNumber==3:
		onlyfiles2 = [str(s) + '.jpg' for s in range(1,1857)]
		image_files=['images/'+f for f in onlyfiles2]
		trajectoryIndexTrain=list()
		imageAddressesTrain=list()
		for i in range(1400):   
			ix = indexes[i]
			for t in range(len(series[ix])-1):
				trajectoryIndexTrain.append((ix,t+1))
				imageAddressesTrain.append(image_files[ix])

		trajectoryIndexVal=list()
		imageAddressesVal=list()
		for i in range(1400,1500):   
			ix = indexes[i]
			trajectoryIndexVal.append(ix)
			imageAddressesVal.append(image_files[ix])
		TrainDg=DataGenerator(trajectoryIndexTrain, imageAddressesTrain,scaler_x,series,seq_length,200)
		ValDg=DataGenerator(trajectoryIndexVal, imageAddressesVal,scaler_x,series,seq_length,200)
		return series,TrainDg,ValDg,scaler_x,indexes

def prepare_data_for_model(modelNumber,features,FeatureSize,seq_length):
	if modelNumber==3:
		series,TrainDg,ValDg,scaler_x,indexes=get_data_for_model(modelNumber,features,FeatureSize,seq_length)
		scaler_filename = scalers_path+ "scaler_x.save"
		joblib.dump(scaler_x, scaler_filename) 
		return TrainDg,ValDg
	else:
		series,selected_features,scaler_x,scaler_features,indexes= get_data_for_model(modelNumber,features,FeatureSize)
		scaler_filename = scalers_path+ "scaler_x.save"
		joblib.dump(scaler_x, scaler_filename) 
		scaler_filename = scalers_path+"scaler_f"+str(features)+"_fs"+ str(FeatureSize)+".save"
		joblib.dump(scaler_features, scaler_filename) 
		samplesAll_x=list()
		samplesAll_y=list()
		if modelNumber==1:
			for i in indexes[:1400]:
				sample=np.zeros([seq_length,FeatureSize+6])
				for t in range(seq_length):            
					sample[t,:6]=scaler_x.transform(series[i][0:1,:])
					sample[t,6:]=scaler_features.transform(selected_features[i:i+1,:])
				for t in range(len(series[i])-1):
					samplesAll_x.append(copy.copy(sample));
					samples_y=scaler_x.transform(series[i][t+1:t+2]).reshape(6)
					samplesAll_y.append(copy.copy(samples_y));
					sample[:seq_length-1,:]=sample[1:seq_length,:]
					sample[seq_length-1,:6]=samples_y[:]
			x = np.array(samplesAll_x)
			y = np.array(samplesAll_y)
			return x,y
		else :
			samplesAll_features=list()
			for i in indexes[:1400]:   
				sample=np.zeros([seq_length,6])
				scaled_features = scaler_features.transform(selected_features[i:i+1,:])[0,:]
				for t in range(seq_length):            
					sample[t,:]=scaler_x.transform(series[i][0:1,:])
				for t in range(len(series[i])-1):  
					samplesAll_x.append(copy.copy(sample));
					samples_y=scaler_x.transform(series[i][t+1:t+2]).reshape(6)
					samplesAll_y.append(copy.copy(samples_y));
					sample[:seq_length-1,:]=sample[1:seq_length,:]
					sample[seq_length-1,:6]=samples_y[:]
					samplesAll_features.append(copy.copy(scaled_features))
			x = np.array(samplesAll_x)
			x_features = np.array(samplesAll_features)
			y = np.array(samplesAll_y)
			return {'supports': x_features, 'traj': x},{'output': y}
