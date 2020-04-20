import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import argparse
import os
import shutil
import tensorflow as tf
from scipy.io import loadmat
from tensorflow.keras import datasets, layers, models



x = loadmat('/home/jupyter/NNScaling/Images/SVHN/test_32x32.mat')
(test_images, test_labels) = (x['X'],x['y'])
test_images = np.array([test_images[:,:,:,i] for i in range(test_images.shape[3])])

x = loadmat('/home/jupyter/NNScaling/Images/SVHN/train_32x32.mat')
(train_images, train_labels) = (x['X'],x['y'])
train_images = np.array([train_images[:,:,:,i] for i in range(train_images.shape[3])])

x = loadmat('/home/jupyter/NNScaling/Images/SVHN/extra_32x32.mat')
(extra_images, extra_labels) = (x['X'],x['y'])
extra_images = np.array([extra_images[:,:,:,i] for i in range(extra_images.shape[3])])

train_images = np.concatenate((train_images,extra_images),axis=0)
train_labels = np.concatenate((train_labels,extra_labels),axis=0)

train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels, test_labels = train_labels-1, test_labels-1


def run(l,epochs=5):
    [n,i] = l
    loss_list = []
    acc_list = []
  
    model = models.Sequential()
    model.add(layers.Conv2D(n, (3, 3), activation='relu', input_shape=train_images[0].shape,padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(n, (3, 3), activation='relu', input_shape=train_images[0].shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10))

    def scheduler(epoch):
      if epoch < 2:
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


n_list = np.array([1,2,3,4,5,6,7,8,10,12])

parser = argparse.ArgumentParser(description='num_features and runs through command line')
parser.add_argument("--iter", default=0, type=int, help="the lower limit of num_features")
args = parser.parse_args()
ite=args.iter


pool_list = []
for n in n_list:
    step = 1
    for i in range(step*ite,step*(ite+1)):
        pool_list.append([n,i])

pool = multiprocessing.Pool(processes=len(pool_list))
print('\n\n\n RUNNING ITERATION',ite, '\n\n\n')
l = pool.map(run,pool_list)
pool.close()




#np.savetxt(fol+'/N',l)
