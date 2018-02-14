"""
Converts a nifti file to a numpy array.
Accepts either a single nifti file or a folder of niftis as the input argument.

Usage:
	python cnn_builder.py

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import keras.backend as K
import keras.layers as layers
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Concatenate, Flatten, Dropout, Lambda
from keras.layers import SimpleRNN, Conv2D, MaxPooling2D, ZeroPadding3D, Activation, ELU, TimeDistributed, Permute, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils

import copy
import config
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import math
import numpy as np
import operator
import os
import pandas as pd
import random
from scipy.misc import imsave
from skimage.transform import rescale
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import time

def build_cnn(optimizer='adam', dilation_rate=(1,1,1), padding=['same', 'valid'], pool_sizes = [(2,2,2), (2,2,2)],
	dropout=[0.1,0.1], f=[64,128,128], dense_units=100, kernel_size=(3,3,2), merge_layer=1,
	dual_inputs=False, run_2d=False, time_dist=True, stride=(1,1,1)):
	"""Main class for setting up a CNN. Returns the compiled model."""

	C = config.Config()

	projections = Input(shape=(C.dims[0], C.dims[1], C.num_projections))
	projection_frames = Input(shape=(2, C.num_projections))
	x = Reshape((C.dims[0], C.dims[1], C.num_projections, 1))(projections)
	x = Permute((3,1,2,4))(x)

	for layer_num in range(len(f)):
		x = layers.TimeDistributed(layers.Conv2D(filters=f[layer_num], kernel_size=kernel_size, padding='same'))(x)
		x = layers.TimeDistributed(layers.Dropout(dropout[0]))(x)
		x = layers.Activation('relu')(x)
		x = layers.TimeDistributed(layers.BatchNormalization(axis=4))(x)
		if layer_num == 0:
			x = TimeDistributed(layers.MaxPooling2D(pool_sizes[0]))(x)

	x = layers.TimeDistributed(layers.MaxPooling3D(pool_sizes[1]))(x)

	#x = SimpleRNN(128, return_sequences=True)(x)
	x = layers.SimpleRNN(dense_units)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(dropout[1])(x)
	else:
		x = layers.MaxPooling3D(pool_sizes[1])(x)
		x = Flatten()(x)

		x = Dense(dense_units)(x)#, kernel_initializer='normal', kernel_regularizer=l2(.01), kernel_constraint=max_norm(3.))(x)
		x = BatchNormalization()(x)
		x = Dropout(dropout[1])(x)
		x = Activation('relu')(x)

	if dual_inputs:
		non_img_inputs = Input(shape=(C.num_non_image_inputs,))
		#y = Dense(20)(non_img_inputs)
		#y = BatchNormalization()(y)
		#y = Dropout(dropout[1])(y)
		#y = Activation('relu')(y)
		x = Concatenate(axis=1)([x, non_img_inputs])

	pred_class = Dense(nb_classes, activation='softmax')(x)

	if not dual_inputs:
		model = Model(img, pred_class)
	else:
		model = Model([img, non_img_inputs], pred_class)
	
	#optim = Adam(lr=0.01)#5, decay=0.001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return model

def get_training_data():
	pass

def reconstruction_loss():
	pass