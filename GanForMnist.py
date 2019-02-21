# -*- coding: utf-8 -*-
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from numpy.random import rand

import matplotlib.pyplot as plt
import sys
import numpy as np

# %matplotlib inline

import os
if os.path.exists('./gdrive') == False:
  from google.colab import drive
  drive.mount('./gdrive')

epoch_no = 50000
batches = 32

model_path = './gdrive/My Drive/Colab Notebooks/model/'
model_generator = 'model_gen.h5'
model_discriminator = 'model_disc.h5'



class GanForMnist():
    def __init__(self,
                 img_rows=28,
                 img_cols=28,
                 channels=1):
        # Size of input data for MNIST
        self.img_rows = img_rows 
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Dimension of latent variable
        self.z_dim = 100

        self.optimizer = Adam(0.0002, 0.5)
        
        tf_exist_generator = os.path.exists(model_path+model_generator)
        tf_exist_discriminator = os.path.exists(model_path+model_discriminator)
        
        if tf_exist_generator and tf_exist_discriminator == False:
          self._define_models()
        else:
          print('Load existing model files')
          self._load_models()
          
    
    def _define_models(self):
      
      # Generator Model
      self.generator = self._build_generator()
      
      # Discriminator Model
      self.discriminator = self._build_discriminator()
      self.discriminator.compile(loss=binary_crossentropy,
                                 optimizer=self.optimizer,
                                 metrics=['accuracy'])
      
      # Combine two models
      self.combined = self._build_combined1()
      self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def _build_generator(self):
      noise_shape = (self.z_dim,)
      
      inputs = Input(shape=noise_shape)
      
      inout_mid = Dense(256, input_shape=noise_shape)(inputs)
      inout_mid = LeakyReLU(alpha=0.2)(inout_mid)
      inout_mid = BatchNormalization(momentum=0.8)(inout_mid)
      inout_mid = Dense(512)(inout_mid)
      inout_mid = LeakyReLU(alpha=0.2)(inout_mid)
      inout_mid = BatchNormalization(momentum=0.8)(inout_mid)
      inout_mid = Dense(1024)(inout_mid)
      inout_mid = LeakyReLU(alpha=0.2)(inout_mid)
      inout_mid = BatchNormalization(momentum=0.8)(inout_mid)
      inout_mid = Dense(np.prod(self.img_shape), activation='tanh')(inout_mid)

      outputs = Reshape(self.img_shape)(inout_mid)

      model = Model(inputs, outputs)
      model.summary()
      return model

      
    def _build_discriminator(self):
      
      inputs = Input(shape=self.img_shape)
      inout_mid = Flatten(input_shape=self.img_shape)(inputs)

      inout_mid = LeakyReLU(alpha=0.2)(inout_mid)
      inout_mid = Dense(256)(inout_mid)
      inout_mid = LeakyReLU(alpha=0.2)(inout_mid)

      output = Dense(1,activation='sigmoid')(inout_mid)

      model = Model(inputs, output)
      model.summary()
      return model

    def _build_combined1(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model
      
    def _load_models(self):
      self.generator = load_model(model_path+model_generator)
      self.discriminator = load_model(model_path+model_discriminator)
      self.combined = self._build_combined1()
      self.combined.compile(loss='binary_crossentropy', 
                            optimizer=self.optimizer)

    def train(self, epochs, batch_size=128, save_interval=50):

        # mnistデータの読み込み
        (X_train, _), (_, _) = mnist.load_data()

        # 値を-1 to 1に規格化
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        print('Start training : epoch = ' + str(epoch_no))
        for epoch in range(epochs):

            # ---------------------
            #  Discriminatorの学習
            # ---------------------

            # バッチサイズの半数をGeneratorから生成
            noise = np.random.normal(0, 1, (half_batch, self.z_dim))
            gen_imgs = self.generator.predict(noise)


            # バッチサイズの半数を教師データからピックアップ
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            
            # ---------------------
            #  Generatorの学習
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            # 生成データの正解ラベルは本物（1） 
            valid_y = np.array([1] * batch_size)
            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)
            
            
            # ----------------------
            # discriminatorを学習
            # 本物データと偽物データは別々に学習させる
            # -----------------------
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            # それぞれの損失関数を平均
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            

            # 進捗の表示
            if epoch % 1000 == 999:
              print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        
        self.generator.save(model_path + model_generator)
        self.discriminator.save(model_path + model_discriminator) 

            # 指定した間隔で生成画像を保存
            #if epoch % save_interval == 0:
            #    self.save_imgs(epoch)
        
                
    def save_imgs(self, epoch):
      # 生成画像を敷き詰めるときの行数、列数
      r, c = 5, 5

      noise = np.random.normal(0, 1, (r * c, self.z_dim))
      gen_imgs = self.generator.predict(noise)

      # 生成画像を0-1に再スケール
      gen_imgs = 0.5 * gen_imgs + 0.5
  
      fig, axs = plt.subplots(r, c)
      cnt = 0
      for i in range(r):
          for j in range(c):
              axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
              axs[i,j].axis('off')
              cnt += 1
          #fig.savefig("images/mnist_%d.png" % epoch)
          #plt.close()
          
    def show_generated_imgs(self, save='NO'):
      N = 9
      row, column = 3, 3

      noise = rand(N, self.z_dim)
      generated_imgs = self.generator.predict(noise)
      
      fig, axs = plt.subplots(row, column)
      cnt = 0
      for i in range(row):
        for j in range(column):
          axs[i, j].imshow(generated_imgs[cnt, :, :, 0], cmap='gray')
          axs[i, j].axis('off')
          cnt = cnt + 1

gan = GanForMnist()
gan.train(epochs=epoch_no, batch_size=batches, save_interval=5)
# gan.save_imgs(300)       s
gan.show_generated_imgs()