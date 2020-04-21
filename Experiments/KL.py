"""
This file loads data from a teacher database folder 'teac600dataBIG_20amb'. The database contains files data2.h5,data3.h5 etc generated from teacher with 2,3 etc features respectively. The teacher used to generate the database has
architecture [20,600,600,2]. The database was generated with reduce=1/20, i.e. 1.6*10**8 datapoints (refer to example.ipynb).

Make a 'data' directory for the data to be saved.

Adjust the number of multiprocessing threads according to the capacity of the machine. To run without multiprocessing, run method main_f  rather than main_f_multiP (uncomment main_f, comment out main_f_multiP). main_f takes argument p, which is a tuple (n,r), where n is the width of hidden layers in student (architecture=[20,n,n,2]) and r is the run (Integer. Useful when doing multiple runs. Can be set to any arbitrary positive integer if doing only one run)

This file takes number of features as input from the command line e.g. python3 KL.py --f=12. In this example the code would load file data12.h5 from folder 'teac600dataBIG_20amb'. 

In the above example, a folder 'features12' would be created in 'data', with subfolders 'losses' and 'models'. Trained models are saved in 'models' as model{n}_{r}, where n is the hidden layer width and r is the index of the run, as discussed above. Similarly loss{n}_{r}.pkl files are saved in 'losses'
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pickle
import teacher
import NeuralNet as student
import argparse
import multiprocessing

parser = argparse.ArgumentParser(description='example: --f=15 for 15 features')
parser.add_argument("--f", default='15', type=int, help="enter --f=num_fea (integer)")
args = parser.parse_args()
f = args.f

#f = 15
database_dir = 'teac600dataBIG_20amb'
batch_size=400
train_steps = 400*1000
epochs = 1

#os.mkdir('data')

fol = 'data/features{}'.format(f)
os.mkdir(fol)
os.mkdir(fol+'/losses')
os.mkdir(fol+'/models')
f_name = os.path.basename(__file__)
shutil.copyfile(f_name, 'data/SourceCode.txt')


#Loading the W.csv and b.csv files from database folder
W = student.get_weights_and_biases(folder=database_dir,file='W.csv')
b = student.get_weights_and_biases(folder=database_dir,file='b.csv')

#deducing the architecture of the teacher from W and b
arc = [len(W[i]) for i in range(len(W))]
arc.append(len(W[len(W)-1][1]))

max_fea = arc[0]
k=[1 if i<f else 0 for i in range(max_fea) ]

print('teacher architecture:',arc)

teach = teacher.Teacher(scale=k,architecture=arc,W=W,b=b,use_database=True,reduce=1,database_dir=database_dir) 

def run_model(n,train_steps=50*1000,verbose=0,lr=0.001,anneal=True,plot=False,batch_size=32,epochs=1):

    model = student.Model([20,n,n,2],softmax=True,loss='KL')
    
    loc = 0
    
    bs = 200 #batch_size
    tt = 200000 #training time (steps)
    lr= 0.01
    history = model.train(teach.data_matrix_x[loc:loc+bs*tt],teach.data_matrix_y[loc:loc+bs*tt],learning_rate=lr,batch_size=bs,verbose=verbose,epochs=1)
    loc += bs*tt
    
    bs = 1000 #batch_size
    tt = 20000 #training time (steps)
    lr= 0.01
    history = model.train(teach.data_matrix_x[loc:loc+bs*tt],teach.data_matrix_y[loc:loc+bs*tt],learning_rate=lr,batch_size=bs,verbose=verbose,epochs=1)
    loc += bs*tt
    
    bs = 4000 #batch_size
    tt = 20000 #training time (steps)
    lr= 0.001
    history = model.train(teach.data_matrix_x[loc:loc+bs*tt],teach.data_matrix_y[loc:loc+bs*tt],learning_rate=lr,batch_size=bs,verbose=verbose,epochs=1)
    loc += bs*tt
    
    
  
    x1,y1 = teach.predict_from_matrix(batch_size=10000)
    loss = model.evaluate(x1,y1)
    
    if plot:
        student.graph_sample(parent,model,folder='plot')

    #return [loss,model,model.count_params()-n-1]
    return loss,model

def main_f(p):
    (n,r) = p
    lr = 0.01
    
    print("f:",f," n:",n," r:",r,end='\r')
    l,m = run_model(n=n,lr=lr,train_steps=train_steps,batch_size=batch_size,epochs=epochs)
    print("f:",f," n:",n," r:",r,'loss:',l,'      ')
    
    m.save_model(location=fol+'/models',name='model'+str(n)+'_'+str(r))
    with open(fol+'/losses/loss'+str(n)+'_'+str(r)+'.pkl', 'wb') as handle:
        pickle.dump(l, handle)
    
        
        
def main_f_multiP(run):
    #breaking down n_list into 2 parts. Reason: even tho this file evaluates only one num_fea, I want to run several num_fea in parallel using gnu parallel. Thus, cpu usage shoots too high if you try too many n's at once. Aim is to try ~9 num_fea values at once, therefore step=1 and 3 values of n would give 27 inputs, or 36 threads with 4 n's. I dont want to exceed those manuy threads on a 96 cpu machine.
    if f%2==0:
        n_list = [int(22*(1.3)**i) for i in range(0,5)]    
        r_list = np.arange(run*step,(run+1)*step)
        n_repeat = np.repeat(n_list,len(r_list))
        r_repeat = np.tile(r_list,len(n_list))
        pool_list = list(zip(n_repeat,r_repeat))    
        pool = multiprocessing.Pool(processes=len(pool_list))
        pool.map(main_f,pool_list)
        pool.close()
        
        n_list = [int(22*(1.3)**i) for i in range(5,7)]    
        r_list = np.arange(run*step,(run+1)*step)
        n_repeat = np.repeat(n_list,len(r_list))
        r_repeat = np.tile(r_list,len(n_list))
        pool_list = list(zip(n_repeat,r_repeat))    
        pool = multiprocessing.Pool(processes=len(pool_list))
        pool.map(main_f,pool_list)
        pool.close()
        
        
    else:               
        n_list = [int(22*(1.3)**i) for i in range(5,7)]  
        n_list.reverse()
        r_list = np.arange(run*step,(run+1)*step)
        n_repeat = np.repeat(n_list,len(r_list))
        r_repeat = np.tile(r_list,len(n_list))
        pool_list = list(zip(n_repeat,r_repeat))    
        pool = multiprocessing.Pool(processes=len(pool_list))
        pool.map(main_f,pool_list)
        pool.close()
        
        n_list = [int(22*(1.3)**i) for i in range(0,5)]   
        n_list.reverse()
        r_list = np.arange(run*step,(run+1)*step)
        n_repeat = np.repeat(n_list,len(r_list))
        r_repeat = np.tile(r_list,len(n_list))
        pool_list = list(zip(n_repeat,r_repeat))    
        pool = multiprocessing.Pool(processes=len(pool_list))
        pool.map(main_f,pool_list)
        pool.close()

        
step = 1
for run in range(3):
    main_f_multiP(run)
    
    
#main_f((12,0))
