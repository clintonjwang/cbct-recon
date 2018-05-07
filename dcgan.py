from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import keras.layers as layers
import config
import matplotlib.pyplot as plt
import sys
import numpy as np
import cnn_builder as cbuild

class DCGAN():
    def __init__(self):
        # Input shape
        C = config.Config()
        optimizer = Adam(0.001)#Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='mse', optimizer=optimizer)

        # The generator takes projections as input and generates 3D imgs
        z = Input(C.proj_dims)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        C = config.Config()
        proj = layers.Input(C.proj_dims)
        x = layers.Reshape((C.proj_dims[0],-1))(proj)
        x = layers.Dense(1024, activation='tanh')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(1024, activation='tanh')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Reshape((C.proj_dims[0],16,16,-1))(x)
        x = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling3D((1,2,2))(x)
        x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
        #x = layers.Dropout(.25)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.Conv3DTranspose(1, 5, strides=(1,2,2), activation='sigmoid', padding='same')(x)
        img = layers.Reshape(C.world_dims)(x)
        model = Model(proj, img)
        model.summary()

        return model

    def build_discriminator(self):
        C = config.Config()
        d_input = Input(C.world_dims)
        y = layers.Reshape((*C.world_dims,1))(d_input)
        y = layers.Conv3D(64, 5, strides=2, activation='relu')(y)
        y = layers.Conv3D(64, 5, strides=2, activation='relu', padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.Conv3D(64, 3, strides=2, activation='relu', padding='same')(y)
        y = layers.Flatten()(y)
        y = layers.BatchNormalization()(y)
        discrim = layers.Dense(1, activation='sigmoid')(y)
        gan_model = Model(d_input, discrim)
        gan_model.summary()

        return gan_model

    def train(self, epochs, batch_size=16, save_interval=50):
        #half_batch = int(batch_size / 2)
        gen = cbuild.train_generator(batch_size)

        for epoch in range(epochs):
            X_train, Y_train = next(gen)

            if epoch % 5 == 0:
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Generate a half batch of backprojected 3D images
                gen_imgs = self.generator.predict(X_train)
                # Train the discriminator (real imgs classified as ones and generated as zeros)
                d_loss_real = self.discriminator.train_on_batch(Y_train, np.ones((batch_size, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
                d_loss = .5*np.add(d_loss_real, d_loss_fake)
            else:
                d_loss = [0, 0]

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (wants discriminator to mistake images as real)
            if epoch % 10 == 0:
                g_loss = self.generator.train_on_batch(X_train, Y_train)
            else:
                g_loss = 0
            c_loss = self.combined.train_on_batch(X_train, np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [C loss: %f]" % \
                    (epoch, d_loss[0], 100*d_loss[1], g_loss, c_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(gen_imgs, epoch)

    def save_imgs(self, gen_imgs, epoch):
        r,c = 2,len(gen_imgs)//2
        fig, axs = plt.subplots(r, c)
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,32], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(r"D:\CBCT\GAN results\mnist_%d.png" % epoch)
        plt.close()
