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
import pickle
tf.compat.v1.disable_eager_execution()


class NeuralNet:

  def __init__(self, architecture,softmax=True):
    self.architecture=architecture
    self.W,self.b = self.create_global_variables()   
    self.softmax = softmax
    

  def weight_variable(self, shape):
    N=shape[0]
    temp=np.sqrt(N)
    sd=1/temp
    initial=tf.random.normal(shape=shape,stddev=sd,dtype=tf.dtypes.float64)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial=tf.random.normal(shape,stddev=0.5,dtype=tf.dtypes.float64)
    return tf.Variable(initial)

  def create_global_variables(self):
    Wcoef=[]
    bcoef=[]
    architecture=self.architecture
    for i in range(len(architecture)-1):
      W=self.weight_variable([architecture[i],architecture[i+1]])
      Wcoef.append(W)
      b=self.bias_variable([architecture[i+1]]) #we don't use bias_variable() because we want random values for weight rather than constant 0.1
      bcoef.append(b)
    return Wcoef,bcoef

  def forward_pass(self,x_input):
    architecture=self.architecture
    Wcoef=self.W
    bcoef=self.b
    y_int=x_input #the intermediate output (output of layers). Initialized to input value
    for i in range(len(architecture)-1):
      y=tf.matmul(y_int,Wcoef[i])+bcoef[i]
      if i==len(architecture)-2:
        if self.softmax:
            #y = tf.linalg.normalize(y,axis=1)[0]
            y_output=tf.nn.softmax(y)
        else: 
            y_output = y
      y_int=tf.nn.relu(y)
    return y_output

  def predict_one_layer(self,x_input,layer=-1): #layers can be labelled 1,2,3... starting from 1st hidden or -1,-2,.. starting from prefinal backwards
    architecture=self.architecture
    Wcoef=self.W
    bcoef=self.b
    if layer>0: pos = range(layer)
    elif layer==0: return self.predict(x_input)
    else: pos = range(len(architecture)-1+layer)
    y_int=x_input #the intermediate output (output of layers). Initialized to input value
    for i in pos:
        y=tf.matmul(y_int,Wcoef[i])+bcoef[i]
        y_int=tf.nn.relu(y)
        y_output = y #NOTE: the output is taken pre relu because y_int (and not y) the output of relu.
    return y_output  

  def predict_layers(self,x_input,layer=None):
    if layer is None:
        chkpt_dict={}
        for i in range(1,len(self.architecture)):
            chkpt_dict['layer'+str(i)] = self.predict_one_layer(x_input,layer=i)
        return chkpt_dict
    else:
        return self.predict_one_layer(x_input, layer=layer)

class Model:

  #Global Variables: 
  # sess, parent, daughter, x_input, loss, train_step

  def __init__(self,architecture,softmax=True,loss='KL',power=2,W=None,b=None):
    self.sess=tf.compat.v1.Session()
    self.daughter=NeuralNet(architecture=architecture,softmax=softmax)
    self.architecture = self.daughter.architecture
    self.softmax = self.daughter.softmax
    self.x_input=tf.compat.v1.placeholder(tf.float64,shape=[None,architecture[0]])
    self.y_input=tf.compat.v1.placeholder(tf.float64,shape=[None,architecture[-1]])
    self.learning_rate = tf.compat.v1.placeholder(tf.float64, shape=[])
    self.create_graph(loss,power) #Graph creation finishes with this last step
    self.loss_formula = loss
    self.power = power
    self.train_sess_count = 0
    self.history = {}
    self.loss_history = []
    if architecture[-1]==1 and loss=='KL': raise ValueError('using KL divergence with one output. use e.g. loss=\'power\' in class arguments')
    if architecture[-1]==1 and softmax: raise ValueError('using softmax with one output. set softmax=False in class arguments')   
    self.all_global_init()
    if W is not None and b is not None: self.weights_assign(W,b)

  def loss(self,loss,power=2):
    y_p=self.y_input
    y_d=self.daughter.forward_pass(self.x_input)
    if loss not in ['KL','power']: raise ValueError('choose argument loss from strings \'KL\',\'power\'')
    if loss=='KL':
        k = tf.keras.losses.KLDivergence()
        self.loss = tf.reduce_mean(k(y_p, y_d))
    elif loss=='power':
        self.loss=tf.reduce_mean(tf.reduce_mean(tf.pow(tf.abs(y_p-y_d),power),axis=[1]))    
        

  def create_graph(self,loss,power=2):
    self.loss(loss=loss,power=power)
    self.train_step=tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,var_list=[self.daughter.W,self.daughter.b])

#Until here we were still building the graph. The following steps only run the graph
  def all_global_init(self):
    self.sess.run(tf.compat.v1.global_variables_initializer())

  def weights_assign(self,W,b):
    for i in range(len(W)):
      ass_op=self.daughter.W[i].assign(W[i])
      self.sess.run(ass_op)
    for i in range(len(b)):
      ass_op=self.daughter.b[i].assign(b[i])
      self.sess.run(ass_op)
  
  def sess_close(self):
    self.sess.close()

  def get_weights(self):
    return self.sess.run(self.daughter.W),self.sess.run(self.daughter.b)
    
  def count_params(self):
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
    
  def predict(self,x):
    return self.sess.run(self.daughter.forward_pass(self.x_input),feed_dict={self.x_input: x})

  def predict_layer(self,x,layer=None):
    return self.sess.run(self.daughter.predict_layers(self.x_input,layer=layer),feed_dict={self.x_input: x})
    
  def evaluate(self,x,y):
    return self.sess.run(self.loss,feed_dict={self.x_input: x,self.y_input: y})
    
  def train_one_epoch_verbose(self,x,y,epoch=1,batch_size=200,learning_rate=0.001,lr_scheduler = None,asymp_steps=1000,skip=100): #call this function after session run. This is not part of graph
    loss_plot=[]
    asymp_loss=[]
    start_time=datetime.datetime.now().time()
    #x = np.array(x)
    #y = np.array(y)
    (lx,fx) = x.shape
    (ly,fy) = y.shape
    num_batches = lx//batch_size
    #x = x[0:num_batches*batch_size]
    #y = y[0:num_batches*batch_size]
    #x_batches = np.reshape(x,(num_batches,batch_size,fx))
    #y_batches = np.reshape(y,(num_batches,batch_size,fy))
    
    for i in range(num_batches):
        x_batch,y_batch = x[batch_size*i:batch_size*(i+1)],y[batch_size*i:batch_size*(i+1)]
        if lr_scheduler is not None: learning_rate = lr_scheduler(epoch-1+(i+1)/num_batches)
        self.sess.run(self.train_step,feed_dict={self.x_input: x_batch,self.y_input: y_batch,self.learning_rate:learning_rate})

        if i%skip==0:
                loss_local=self.sess.run(self.loss,feed_dict={self.x_input: x_batch,self.y_input: y_batch})
                loss_plot.append(loss_local)
                print('training:',i*100//num_batches,'%','  loss:',loss_local,' lr:',learning_rate,'                                                                                                              ',end='\r')
        if num_batches-i<=asymp_steps:
                loss_local=self.sess.run(self.loss,feed_dict={self.x_input: x_batch,self.y_input: y_batch})
                asymp_loss.append(loss_local)
    
    print()
    end_time=datetime.datetime.now().time()
    return {'asymptotic loss': asymp_loss,'loss history': loss_plot,'start time': start_time,'end time':end_time}

  def train_one_epoch_nonverbose(self,x,y,epoch=1,batch_size=200,learning_rate=0.001,lr_scheduler = None,asymp_steps=1000,skip=100): #call this function after session run. This is not part of graph
    loss_plot=[]
    asymp_loss=[]
    start_time=datetime.datetime.now().time()
    #x = np.array(x)
    #y = np.array(y)
    (lx,fx) = x.shape
    (ly,fy) = y.shape
    num_batches = lx//batch_size
    #x = x[0:num_batches*batch_size]
    #y = y[0:num_batches*batch_size]
    #x_batches = np.reshape(x,(num_batches,batch_size,fx))
    #y_batches = np.reshape(y,(num_batches,batch_size,fy))
    
    for i in range(num_batches):
        x_batch,y_batch = x[batch_size*i:batch_size*(i+1)],y[batch_size*i:batch_size*(i+1)]
        if lr_scheduler is not None: learning_rate = lr_scheduler(epoch-1+(i+1)/num_batches)
        self.sess.run(self.train_step,feed_dict={self.x_input: x_batch,self.y_input: y_batch,self.learning_rate:learning_rate})

        if i%skip==0:
                loss_local=self.sess.run(self.loss,feed_dict={self.x_input: x_batch,self.y_input: y_batch})
                loss_plot.append(loss_local)
                
        if num_batches-i<=asymp_steps:
                loss_local=self.sess.run(self.loss,feed_dict={self.x_input: x_batch,self.y_input: y_batch})
                asymp_loss.append(loss_local)
    
    end_time=datetime.datetime.now().time()
    return {'asymptotic loss': asymp_loss,'loss history': loss_plot,'start time': start_time,'end time':end_time}


  def train(self,x,y,batch_size=32,epochs=1,learning_rate=0.001,lr_scheduler = None,verbose=2,asymp_steps=1000,skip=100):
    # decide the most useful way of returning history etc
    self.train_sess_count += 1
    all_hist = {}
    loss_history = []
    for epoch in range(epochs):
        if verbose>0: print('epoch:',epoch+1)

        if verbose>1: 
            d = self.train_one_epoch_verbose(x,y,epoch=epoch+1,batch_size=batch_size,learning_rate=learning_rate,lr_scheduler=lr_scheduler,asymp_steps=asymp_steps,skip=skip)
        else: 
            d = self.train_one_epoch_nonverbose(x,y,epoch=epoch+1,batch_size=batch_size,learning_rate=learning_rate,lr_scheduler=lr_scheduler,asymp_steps=asymp_steps,skip=skip)
            
        all_hist['epoch'+str(epoch+1)] = d
        loss_history += d['loss history']
        if epoch==0: all_hist['start time'] = d['start time']
            
    all_hist['loss history'] = loss_history
    all_hist['loss history step size'] = skip
    all_hist['end time'] = d['end time']
    all_hist['asymptotic loss'] = d['asymptotic loss']
    
    self.history['training session {}'.format(self.train_sess_count)] = all_hist
    self.loss_history += loss_history
    
    return all_hist

  def save_weights(self,folder,W,b):
    W_new=[]
    b_new=[]
    for elem in W:
        temp=elem.tolist()
        W_new.append(temp)
    for elem in b:
        temp=elem.tolist()
        b_new.append(temp)
    name_W=folder+'/W'+'.csv'
    name_b=folder+'/b'+'.csv'
    with open(name_W, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(W_new)
    with open(name_b, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(b_new)
        
  def save_model(self,name,location='.'):
    fol = location+'/'+name
    os.mkdir(fol)
    W,b = self.get_weights()
    self.save_weights(folder=fol,W=W,b=b)
    with open(fol+"/softmax", "w") as h:
        h.write(str(self.daughter.softmax))
    with open(fol+"/loss", "w") as h:
        h.write(self.loss_formula)     
    np.savetxt(fol+'/power',[self.power]) 
    with open(fol+'/history.pickle', 'wb') as handle:
        pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(fol+'/loss_history.pickle', 'wb') as handle:
        pickle.dump(self.loss_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def histogram(sess,parent,daughter,folder):
    b_size = 2000
    x,y = parent.predict_from_matrix(batch_size=b_size)
    yd = sess.run(daughter.forward_pass(x_input=x))
    with open(folder+'/'+str(daughter.architecture)+'_histogram.txt', 'w') as f:
        for i in range(parent.architecture[-1]):
            bin_range = np.linspace(0.0,3.0/parent.architecture[-1],10)
            h,b = np.histogram(np.array(yd)[:,i],bins=bin_range)
            f.write("%s\n" % ('logit '+str(i)+': '+str(h/b_size)))
            

def weights_read(folder,file):
    fil_name=folder+'/'+file
    with open(fil_name, 'r') as f:
        reader = csv.reader(f)
        weights = list(reader)
    return weights

def get_weights_and_biases(folder,file):
    weights_temp = weights_read(folder=folder,file=file)
    weights = []
    for row in weights_temp:
        nwrow = []
        for r in row:
            nwrow.append(eval(r))
        weights.append(nwrow)
    temp=[]
    for mat in weights:
        temp.append(np.array(mat))
    weights = temp
    return weights

def load_model(name,folder='.'):
    fol = folder+'/'+name
    
    W = get_weights_and_biases(folder=fol,file='W.csv')
    W = [t.astype(np.float64) for t in W]
    b = get_weights_and_biases(folder=fol,file='b.csv')
    b = [t.astype(np.float64) for t in b]
    arc = [len(W[i]) for i in range(len(W))]
    arc.append(len(W[len(W)-1][1]))
    with open (fol+'/softmax', "r") as myfile:
        data = myfile.readlines()
    softmax = eval(data[0])
    with open (fol+'/loss', "r") as myfile:
        data = myfile.readlines()
    loss = data[0]
    power = float(np.loadtxt(fol+'/power'))
    with open(fol+'/history.pickle', 'rb') as handle:
        his = pickle.load(handle)
    with open(fol+'/loss_history.pickle', 'rb') as handle:
        l_his = pickle.load(handle)
    
    model = Model(architecture=arc,W=W,b=b,softmax=softmax,loss=loss,power=power)
    model.history = his
    model.loss_history = l_his
    
    return model

        

