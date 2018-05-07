"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/lipiodol`
Author: David G Ellis (https://github.com/ellisdg/3DUnetCNN)
"""

import keras.backend as K
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Concatenate, Flatten, Dropout, Lambda
from keras.layers import SimpleRNN, Conv2D, MaxPooling2D, ZeroPadding3D, Activation, ELU, TimeDistributed, Permute, Reshape
from keras.layers.normalization import BatchNormalization
import keras.layers as layers
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils

import argparse
import copy
import config
import importlib
import niftiutils.cnn_components as cnnc
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import math
from math import log, ceil
import glob
import numpy as np
import operator
import os
from os.path import *
import pandas as pd
import random
from scipy.misc import imsave
from skimage.transform import rescale
from niftiutils.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
import time
import keras.regularizers as regularizers

def norm(x):
	x -= K.min(x)
	x = x/K.max(x)
	return x

def build_cnn(optimizer='adam', lr=0.00002):
	"""Main class for setting up a CNN. Returns the compiled model."""
	importlib.reload(config)

	C = config.Config()

	proj = layers.Input(C.proj_dims)
	#x = layers.Permute((2,1,3))(img)
	x = layers.Reshape((C.proj_dims[0],-1))(proj)
	x = layers.Dense(1024, activation='tanh')(x) #, kernel_regularizer=regularizers.l1(0.01)
	x = layers.BatchNormalization()(x)
	#x = layers.Reshape((C.proj_dims[0],32,-1))(x)
	#x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
	#x = layers.Reshape((C.proj_dims[0],-1))(x)
	x = layers.Dense(1024, activation='tanh')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Reshape((C.proj_dims[0],32,32,-1))(x)
	x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
	#x = layers.UpSampling3D((1,2,2))(x)
	x = layers.MaxPooling3D((2,1,1))(x)
	x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv3DTranspose(1, 3, activation='sigmoid', padding='same')(x)
	img = layers.Reshape(C.world_dims)(x)
	#x = layers.Lambda(norm)(x)
	#x = layers.Permute((2,1,3))(x)
	#x = layers.Conv2D(64, (2,2), activation='relu', padding='same')(x)
	#x = layers.Conv2D(64, (2,2), padding='same')(x)

	model = Model(proj, img)
	model.compile(optimizer=RMSprop(lr=lr, decay=0.1), loss='mse')

	if False:
		x = layers.Reshape((C.proj_dims[0],-1))(proj)
		x = layers.Dense(1024, activation='tanh')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dense(1024, activation='tanh')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Reshape((C.proj_dims[0],32,32,-1))(x)
		x = layers.Conv3D(64, (3,3,3), activation='relu', padding='same')(x)
		x = layers.UpSampling3D((1,2,2))(x)
		x = layers.Conv3D(64, (3,3,3), activation='relu', padding='same')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Conv3DTranspose(1, (1,3,3), activation='sigmoid', padding='same')(x)

	return model

def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

	for e in tqdm(range(nb_epoch)):
		# Make generative images
		image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]    
		noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
		generated_images = generator.predict(noise_gen)
		
		# Train discriminator on generated images
		X = np.concatenate((image_batch, generated_images))
		y = np.zeros([2*BATCH_SIZE,2])
		y[0:BATCH_SIZE,1] = 1
		y[BATCH_SIZE:,0] = 1
		
		#make_trainable(discriminator,True)
		d_loss  = discriminator.train_on_batch(X,y)
		losses["d"].append(d_loss)
	
		# train Generator-Discriminator stack on input noise to non-generated output class
		noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
		y2 = np.zeros([BATCH_SIZE,2])
		y2[:,1] = 1
		
		#make_trainable(discriminator,False)
		g_loss = GAN.train_on_batch(noise_tr, y2 )
		losses["g"].append(g_loss)
		
		# Updates plots
		if e%plt_frq==plt_frq-1:
			plot_loss(losses)
			plot_gen()
####################################
### Training Submodules
####################################

def train_generator(n=8):
	C = config.Config()

	fns = glob.glob(r"D:\CBCT\Train\NPYs\*_img.npy")
	while True:
		lesion_ids = random.sample(fns, n)
		X_train = np.empty((n,*C.proj_dims))
		Y_train = np.empty((n,*C.world_dims))

		for ix, lesion_id in enumerate(lesion_ids):
			X_train[ix] = np.load(lesion_id.replace("_img", "_proj"))
			Y_train[ix] = tr.rescale_img(np.load(lesion_id), C.world_dims)

		yield X_train, Y_train
