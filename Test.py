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
from Prepare_Data import *

dataset_path='trajectories/'
scalers_path='scalers/'
num_of_Data=1857
onlyfiles2 = [str(s) + '.jpg' for s in range(1,1857)]

def_traj_lenght=74
started_moving_threshold=0.0000001
object_slip_threshold=0.03

## TODO: SEPERATE DATA PREPARATION AND RUNNING MODEL
def validate_model(modelNumber,features,FeatureSize,seq_length,model):
	scaler_filename = scalers_path+ "scaler_x.save"
	scaler_x= joblib.load(scaler_filename) 
	trajectoryOrij=list()
	trajectory=list()
	val_len = 100
	if modelNumber==3:
		series,_,_,_,indexes=get_data_for_model(modelNumber,features,FeatureSize)
		sample=np.zeros([val_len,seq_length,6])
		sample2=np.zeros([val_len,128,128,1])
		sample3=np.zeros([val_len,seq_length,10])
		max_traj_len=0
		trajectoryIndexVal=indexes[1400:1500]
		for ii in range(val_len):
			i = trajectoryIndexVal[ii]
			imageAddres='images/'+str(i+1)+'.jpg'
			trajectoryOrij.append(series[i])
			trajectory.append(np.zeros_like(series[i]))
			sample2[ii,:,:,:]=image.img_to_array(image.load_img(imageAddres, color_mode = "grayscale").resize((128,128)))
			if len(series[i])>max_traj_len:
				max_traj_len=len(series[i])
			for t in range(seq_length):        
				sample[ii,t,:6]=scaler_x.transform(series[i][0:1,:])
		sample2=(sample2.astype('float32') / 255. / 0.5).astype('int32')
		for t in range(max_traj_len-1):
			result = model.predict([sample2,sample,sample3], batch_size=100, verbose=0)
			for i in range(val_len):
				if t<len(trajectoryOrij[i])-1:
					trajectory[i][t+1,:]=scaler_x.inverse_transform(result[i:i+1,:])
				sample[i,:seq_length-1,:]=sample[i,1:seq_length,:]
				sample[i,seq_length-1:seq_length,:6]=result[i,:]
	else :
		scaler_filename = scalers_path+"scaler_f"+str(features)+"_fs"+ str(FeatureSize)+".save"
		scaler_features = joblib.load(scaler_filename) 
		series,selected_features,_,_,indexes= get_data_for_model(modelNumber,features,FeatureSize)
		trajectoryIndexVal=indexes[1400:1500]
		max_traj_len=0
		if modelNumber==1:
			sample=np.zeros([val_len,seq_length,FeatureSize+6])
		if modelNumber==2:
			sample=np.zeros([val_len,seq_length,6])
			sample2=np.zeros([val_len,seq_length,FeatureSize])
		for ii in range(val_len):
			i = trajectoryIndexVal[ii]
			trajectoryOrij.append(series[i])
			trajectory.append(np.zeros_like(series[i]))
			if len(series[i])>max_traj_len:
				max_traj_len=len(series[i])
			for t in range(seq_length):        
				sample[ii,t,:6]=scaler_x.transform(series[i][0:1,:])
				if modelNumber==1:
					sample[ii,t,6:]=scaler_features.transform(selected_features[i:i+1,:])
				if modelNumber==2:
					sample2[ii,t,:]=scaler_features.transform(selected_features[i:i+1,:])
		for t in range(max_traj_len-1):
			if modelNumber==1:
				result = model.predict(sample, batch_size=100, verbose=0)
			if modelNumber==2:
				result = model.predict({sample,sample2}, batch_size=100, verbose=0)
			for i in range(val_len):
				if t<len(trajectoryOrij[i])-1:
					trajectory[i][t+1,:]=scaler_x.inverse_transform(result[i:i+1,:])
				sample[i,:seq_length-1,:]=sample[i,1:seq_length,:]
				sample[i,seq_length-1:seq_length,:6]=result[i,:]
	errorsxyz=list()
	for ii in range(val_len): #5 for training visualization
		errorsxyz.append(math.sqrt(sum((trajectoryOrij[ii][-1,:3]-trajectory[ii][-1,:3])**2)))
	print ('Total XYZ MSE Error'+' Mean='+str(np.mean(errorsxyz)*100)[:4]+' Std='+str(np.std(errorsxyz)*100)[:4])
	return np.mean(errorsxyz),np.std(errorsxyz)*100
def test_model(modelNumber,features,FeatureSize,seq_length,model):
	scaler_filename = scalers_path+ "scaler_x.save"
	scaler_x= joblib.load(scaler_filename) 
	trajectoryOrij=list()
	trajectory=list()
	val_len = 250
	if modelNumber==3:
		series,_,_,_,indexes=get_data_for_model(modelNumber,features,FeatureSize)
		sample=np.zeros([val_len,seq_length,6])
		sample2=np.zeros([val_len,128,128,1])
		sample3=np.zeros([val_len,seq_length,10])
		max_traj_len=0
		trajectoryIndexVal=indexes[1500:1750]
		for ii in range(val_len):
			i = trajectoryIndexVal[ii]
			imageAddres='images/'+str(i+1)+'.jpg'
			trajectoryOrij.append(series[i])
			trajectory.append(np.zeros_like(series[i]))
			sample2[ii,:,:,:]=image.img_to_array(image.load_img(imageAddres, color_mode = "grayscale").resize((128,128)))
			if len(series[i])>max_traj_len:
				max_traj_len=len(series[i])
			for t in range(seq_length):        
				sample[ii,t,:6]=scaler_x.transform(series[i][0:1,:])
		sample2=(sample2.astype('float32') / 255. / 0.5).astype('int32')
		for t in range(max_traj_len-1):
			result = model.predict([sample2,sample,sample3], batch_size=250, verbose=0)
			for i in range(val_len):
				if t<len(trajectoryOrij[i])-1:
					trajectory[i][t+1,:]=scaler_x.inverse_transform(result[i:i+1,:])
				sample[i,:seq_length-1,:]=sample[i,1:seq_length,:]
				sample[i,seq_length-1:seq_length,:6]=result[i,:]
	else :
		scaler_filename = scalers_path+"scaler_f"+str(features)+"_fs"+ str(FeatureSize)+".save"
		scaler_features = joblib.load(scaler_filename) 
		series,selected_features,_,_,indexes= get_data_for_model(modelNumber,features,FeatureSize)
		trajectoryIndexVal=indexes[1500:1750]
		max_traj_len=0
		if modelNumber==1:
			sample=np.zeros([val_len,seq_length,FeatureSize+6])
		if modelNumber==2:
			sample=np.zeros([val_len,seq_length,6])
			sample2=np.zeros([val_len,seq_length,FeatureSize])
		for ii in range(val_len):
			i = trajectoryIndexVal[ii]
			trajectoryOrij.append(series[i])
			trajectory.append(np.zeros_like(series[i]))
			if len(series[i])>max_traj_len:
				max_traj_len=len(series[i])
			for t in range(seq_length):        
				sample[ii,t,:6]=scaler_x.transform(series[i][0:1,:])
				if modelNumber==1:
					sample[ii,t,6:]=scaler_features.transform(selected_features[i:i+1,:])
				if modelNumber==2:
					sample2[ii,t,:]=scaler_features.transform(selected_features[i:i+1,:])
		for t in range(max_traj_len-1):
			if modelNumber==1:
				result = model.predict(sample, batch_size=250, verbose=0)
			if modelNumber==2:
				result = model.predict({sample,sample2}, batch_size=250, verbose=0)
			for i in range(val_len):
				if t<len(trajectoryOrij[i])-1:
					trajectory[i][t+1,:]=scaler_x.inverse_transform(result[i:i+1,:])
				sample[i,:seq_length-1,:]=sample[i,1:seq_length,:]
				sample[i,seq_length-1:seq_length,:6]=result[i,:]
	errorsxyz=list()
	for ii in range(val_len): #5 for training visualization
		errorsxyz.append(math.sqrt(sum((trajectoryOrij[ii][-1,:3]-trajectory[ii][-1,:3])**2)))
	print ('Total XYZ MSE Error'+' Mean='+str(np.mean(errorsxyz)*100)[:4]+' Std='+str(np.std(errorsxyz)*100)[:4])
	return np.mean(errorsxyz),np.std(errorsxyz)*100,trajectoryOrij,trajectory

