from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import time
import math
import datetime
import matplotlib.pyplot as plt
import random
import os
import csv
import multiprocessing
import shutil
import argparse
import tables
from tensorflow.keras import datasets, layers, models
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.disable_eager_execution()


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

'''
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = np.expand_dims(train_images,axis=3)
test_images = np.expand_dims(test_images,axis=3)
'''

def run(l,epochs=50):
    [n,i] = l
    loss_list = []
    acc_list = []
    model = models.Sequential()
    model.add(layers.Conv2D(n, (3, 3), activation='relu', input_shape=train_images[0].shape,padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(2*n, (3, 3), activation='relu',padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(2*n, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    def scheduler(epoch):
      if epoch < 3:
        return 0.01
      else:
        return 0.01 * np.cos(np.pi/2 * (epoch-2)/(epochs-2))
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
        
    #history = model.fit(train_images, train_labels, epochs=epochs,callbacks=[callback], validation_data=(test_images, test_labels),verbose=0)
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels),verbose=0)
    
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    train_loss, train_acc = model.evaluate(train_images,  train_labels, verbose=2)
    N = model.count_params()
    
    model.save('data/models/model'+str(n)+'_'+str(i))

    np.savetxt('data/loss_history/train_loss_hist'+str(n)+'_'+str(i),history.history['loss'])
    np.savetxt('data/acc_history/train_acc_hist'+str(n)+'_'+str(i),history.history['accuracy'])
    np.savetxt('data/loss_history/test_loss_hist'+str(n)+'_'+str(i),history.history['val_loss'])
    np.savetxt('data/acc_history/test_acc_hist'+str(n)+'_'+str(i),history.history['val_accuracy'])
    
    np.savetxt('data/losses/train_loss'+str(n)+'_'+str(i),[train_loss])
    np.savetxt('data/accuracies/train_acc'+str(n)+'_'+str(i),[train_acc])
    np.savetxt('data/losses/test_loss'+str(n)+'_'+str(i),[test_loss])
    np.savetxt('data/accuracies/test_acc'+str(n)+'_'+str(i),[test_acc])
    
    return N


fol = 'data'

if not os.path.exists('data'):
    os.mkdir('data')
    os.mkdir('data/losses')
    os.mkdir('data/accuracies')
    os.mkdir('data/models')
    os.mkdir('data/loss_history')
    os.mkdir('data/acc_history')
    f_name=os.path.basename(__file__)
    shutil.copyfile(f_name, 'data/SourceCode.txt')


n_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

parser = argparse.ArgumentParser(description='num_features and runs through command line')
parser.add_argument("--iter", default=0, type=int, help="the lower limit of num_features")
args = parser.parse_args()
ite=args.iter


pool_list = []
for n in n_list:
    step = 2
    for i in range(step*ite,step*(ite+1)):
        pool_list.append([n,i])

pool = multiprocessing.Pool(processes=len(pool_list))
print('\n\n\n RUNNING ITERATION',ite, '\n\n\n')
l = pool.map(run,pool_list)
pool.close()




#np.savetxt(fol+'/N',l)
