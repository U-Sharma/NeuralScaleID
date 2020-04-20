import tensorflow as tf
import numpy as np
import time
import random
import os
import csv
import tables
import sys
import psutil
tf.compat.v1.disable_eager_execution()

class Teacher: #even if use_database = True, still supply W,b (for saving parent weights), architecture, scale. database_dir must have a filesize.txt
  def __init__(self,architecture=None,scale=None,reduce=1,W=None,b=None,bias_sd=0.00,weights_init='gaussian',softmax=True,database_dir='/dev/tmp',use_database=False,create_database=False):
    
    if (W is None or b is None) and (architecture is None):
        raise ValueError('enter either weights (W and b) or architecture')
    
    if (W is None and b is not None) or (b is None and W is not None):
        raise ValueError('enter either both W and b or neither')
    
    self.architecture = architecture
    if W is not None and b is not None:
        if architecture is not None: print('WARNING: overriding architecture from class argument with that compatible with weights')
        arc = [len(W[i]) for i in range(len(W))]
        arc.append(len(W[len(W)-1][1]))
        self.architecture = arc
        self.W,self.b = W,b
    if W is None: self.W,self.b = self.create_global_variables(bias_scale=bias_sd,weights_init=weights_init)
        
    self.scale = scale
    if scale is None: self.scale = np.ones(self.architecture[0]).astype(int)
        
    if len(self.scale)!=self.architecture[0]:
        raise ValueError('length of scale is not equal to the input size of teacher network') 
    
    self.max_features = len(self.scale)
    self.softmax = softmax
    
    if self.softmax and (self.architecture[-1]==1): raise ValueError('using softmax with one output. set softmax=False in class arguments')
    
    if create_database:
        if not os.path.isdir(database_dir): os.mkdir(database_dir)
        self.filename = database_dir+"/"+'data'+str(np.sum(self.scale))+'.h5'
        self.filesize = int(200000*4*10/reduce)
        self.weights_save_teacher(folder=database_dir,W=self.W,b=self.b)
        print('creating database file...',end='\r')
        self.create_data_file()
        if not os.path.exists(database_dir+'/filesize.txt'): np.savetxt(database_dir+'/filesize.txt',np.array([self.filesize]))

    if use_database:
        print('WARNING: using preexisting database. method predict() may not be reliable')
        self.filename = database_dir+'/data'+str(np.sum(self.scale))+'.h5'
        self.filesize = int(np.loadtxt(database_dir+'/filesize.txt'))
        if reduce<1: print("ERROR!!! reduce<1 leads to overflow with use_database=True");sys.exit();
        self.filesize = int(self.filesize/reduce)
        print('loading database...            ',end='\r')
        self.create_data_matrix_from_file()
        print('database loaded.                      ')
    #self.create_data_matrix() #For some reason this routine is much slower (factor of 2-4) comapred to first storing data to file and then fetching it. Weird. Maybe .h5 files are extremely efficient
    
  def weight_variable(self, shape, weights_init='gaussian'):
    N = shape[0]
    temp = np.sqrt(N)
    sd = 1/temp
    np.random.seed()
    if weights_init=='gaussian': initial = np.random.normal(size=shape,scale=sd)
    if weights_init=='uniform': initial = (np.random.rand(shape[0],shape[1])-0.5)*sd*np.sqrt(12)
    return initial

  def bias_variable(self, shape, scale=0.00, weights_init='uniform'):
    np.random.seed()
    if weights_init=='gaussian': initial = np.random.normal(size=shape,scale=scale)
    if weights_init=='uniform': initial = (np.random.rand(shape[0])-0.5)*scale*np.sqrt(12)
    return initial

  def create_global_variables(self,bias_scale=0.00,weights_init='gaussian'):  #Note that this returns a list of numpy arrays
    Wcoef = []
    bcoef = []
    architecture=self.architecture
    for i in range(len(architecture)-1):
      W = self.weight_variable(shape=[architecture[i],architecture[i+1]],weights_init=weights_init)
      Wcoef.append(W)
      b = self.bias_variable(shape=[architecture[i+1]],scale=bias_scale,weights_init=weights_init) #we don't use bias_variable() because we want random values for weight rather than constant 0.1
      bcoef.append(b)
    return Wcoef,bcoef

  def generate_input(self,batch_size=1):
    height = self.max_features
    x = np.random.rand(batch_size,height)-0.5
    #list = temp.tolist()
    return x

  def generate_input_sphere(self,batch_size=1): #uniformly on the surface of a sphere
    height = np.sum(self.scale)
    zeros_ct = self.max_features-height
    temp = np.random.normal(size=(batch_size,height))
    norm = np.transpose([np.linalg.norm(temp,axis=1)])
    x = np.divide(temp,norm)
    x_ext = np.zeros(shape=(batch_size,zeros_ct))
    x = np.concatenate((x,x_ext),axis=1)
    return x

  def create_data_file(self):
    total_lines = self.filesize
    repeats = 1000
    batch_size = int(total_lines/repeats)
    self.filesize = batch_size*repeats
    filename = self.filename
    #initial creation of the data file (if statement is added for multiprocessing)
    if not os.path.exists(filename):
      ROW_SIZE = self.architecture[0]+self.architecture[-1]
      f = tables.open_file(filename, mode='w')
      atom = tables.Float64Atom()
      array_c = f.create_earray(f.root, 'data', atom, (0, ROW_SIZE))
      f.close()
      f = tables.open_file(filename, mode='a') #file opened for concatenation
      for i in range(repeats):
            print(int(i/repeats*100),"% complete (data file generation)          ",end="\r")
            x_input = self.generate_input(batch_size=batch_size)
            y_output = self.predict(x_input)
            data_rows = np.concatenate((x_input,y_output),axis=1)
            f.root.data.append(data_rows)
      f.close()
    else:
        print('Data file exists. Waiting for 200 seconds before proceeding.')
        time.sleep(200) 
    #data file has been created and closed


  def predict_from_file(self,batch_size=1,res=None): #takes in batch_size and returns x_input,y_input from a pre-saved file
    #f = tables.open_file(self.filename, mode='r')
    with tables.open_file(self.filename, mode='r') as f:
        try:       
            if res is None: res = random.sample(range(self.filesize), batch_size)
            data = f.root.data[res,:]
        except KeyError:
            pass
        else: #return x,y
            return data[:,:self.architecture[0]],data[:,self.architecture[0]:]
    
  def create_data_matrix_from_file(self): #this function lets one process create a data_file and import the matrix from that file to each process
    length = self.filesize//100
    self.data_matrix_x = np.zeros(shape=(length*100,self.max_features))
    self.data_matrix_y = np.zeros(shape=(length*100,self.architecture[-1]))
    total_m = int(np.round(psutil.virtual_memory().total / (1024.0 ** 3)))
    for i in range(100):
        used_m = int(np.round(psutil.virtual_memory().used / (1024.0 ** 3)))
        print('loading data',i,'% ',' memory total:',total_m,'G  used:',used_m,'G                   ',end='\r')
        length = self.filesize//100
        x,y = self.predict_from_file(res=np.arange(i*length,(i+1)*length))
        self.data_matrix_x[i*length:(i+1)*length,:] = x
        self.data_matrix_y[i*length:(i+1)*length,:] = y
        #if i==0:
        #    self.data_matrix_x,self.data_matrix_y = x,y
        #else:           
        #    self.data_matrix_x = np.concatenate((self.data_matrix_x,x),axis=0)
        #    self.data_matrix_y = np.concatenate((self.data_matrix_y,y),axis=0)
    
    
  def predict_from_matrix(self,batch_size=1):
    length = len(self.data_matrix_x)
    res = random.sample(range(length), batch_size)
    return self.data_matrix_x[res,:],self.data_matrix_y[res,:]
    
  def predict(self,x_input): #The original predict. Takes x_input (a whole batch) of form [[],[],[],...] and gives output [[],[],[],...]
    architecture = self.architecture
    Wcoef = self.W
    bcoef = self.b
    y_int = x_input #the intermediate output (output of layers). Initialized to input value
    y_int = np.multiply(y_int,self.scale)
    for i in range(len(architecture)-1):
      y = np.matmul(y_int,Wcoef[i])+bcoef[i]
      if i==len(architecture)-2:
            if self.softmax:
                y = np.exp(6*y)
                y_output = y/np.transpose([np.sum(y,axis=1)])
            else:
                y_output = 6*y
      y_int = np.maximum(y,0)
    return y_output

  def predict_one_layer(self,x_input,layer=-1,relu=False): #layers can be labelled 1,2,3... starting from 1st hidden or -1,-2,.. starting from prefinal backwards
    architecture=self.architecture
    Wcoef = self.W
    bcoef = self.b
    if layer>0: pos = range(layer)
    elif layer==0: return self.predict(x_input)
    else: pos = range(len(architecture)-1+layer)
    y_int = x_input #the intermediate output (output of layers). Initialized to input value
    for i in pos:
        y = np.matmul(y_int,Wcoef[i])+bcoef[i]
        y_int = np.maximum(y,0)
        if relu: y_output = y_int
        else: y_output = y #NOTE: the output is taken pre relu because y_int (and not y) the output of relu.
    return y_output  

  def predict_layers(self,x_input,layer=None,relu=False):
    if layer is None:
        chkpt_dict={}
        for i in range(1,len(self.architecture)):
            chkpt_dict['layer'+str(i)] = self.predict_one_layer(x_input,layer=i,relu=relu)
        return chkpt_dict
    else:
        return self.predict_one_layer(x_input, layer=layer,relu=relu)

  def weights_save_teacher(self,folder,W,b,scale=None,arc=None,fname=None):
    W_new = []
    b_new = []
    for elem in W:
        temp = elem.tolist()
        W_new.append(temp)
    for elem in b:
        temp = elem.tolist()
        b_new.append(temp)
    name_W = folder+'/'+'W.csv'
    name_b = folder+'/'+'b.csv'
    with open(name_W, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(W_new)
    with open(name_b, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(b_new)

        

        
def weights_read(folder,file):
    fil_name=folder+'/'+file
    with open(fil_name, 'r') as f:
        reader = csv.reader(f)
        weights = list(reader)
    return weights

def load_weights(file,folder='.'):
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
