"""
This file loads data from a teacher database folder 'teac600dataL2_20amb'. The teacher used to generate the database has
architecture [20,600,600,1]. The database was generated with softmax=False and reduce=1/20, i.e. 1.6*10**8 datapoints (refer to example.ipynb).

Make a 'data' directory for the data to be saved.

Adjust the number of multiprocessing threads according to the capacity of the machine. To run without multiprocessing, run method main_f  rather than main_f_multiP (uncomment main_f, comment out main_f_multiP). main_f takes argument p, which is a tuple (n,r), where n is the width of hidden layers in student (architecture=[20,n,n,1]) and r is the run (Integer. Useful when doing multiple runs. Can be set to any arbitrary positive integer if doing only one run)

This file takes number of features as input from the command line e.g. python3 L2.py --f=12. In this example the code would load file data12.h5 from folder 'teac600dataL2_20amb'. 

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
parser.add_argument("--f", default='3', type=int, help="enter --f=num_fea (integer)")
args = parser.parse_args()
f = args.f

batch_size=400
train_steps = 400*1000
epochs = 1

#os.mkdir('data')

fol = 'data/features{}'.format('3plus3plus3')
os.mkdir(fol)
os.mkdir(fol+'/losses')
os.mkdir(fol+'/models')
f_name = os.path.basename(__file__)
shutil.copyfile(f_name, fol+'/SourceCode.txt')



teach1 = teacher.Teacher(architecture=[9,600,600,1],scale=[1,1,1,0,0,0,0,0,0],use_database=True,reduce=1,database_dir='database_X_3',softmax=False)
teach2 = teacher.Teacher(architecture=[9,600,600,1],scale=[0,0,0,1,1,1,0,0,0],use_database=True,reduce=1,database_dir='database_Y_3',softmax=False)
teach3 = teacher.Teacher(architecture=[9,600,600,1],scale=[0,0,0,0,0,0,1,1,1],use_database=True,reduce=1,database_dir='database_Z_3',softmax=False)

print("final teachers arch:",teach1.architecture,teach2.architecture,teach3.architecture)
print(teach1.scale)
print(teach2.scale)
print(teach3.scale)

def run_model(n,train_steps=50*1000,verbose=2,lr=0.001,anneal=True,plot=False,batch_size=32,epochs=1):

    model = student.Model([9,n,n,1],softmax=False,loss='power')
    
    dm_x = np.concatenate((teach1.data_matrix_x[:,0:3],teach2.data_matrix_x[:,3:6],teach3.data_matrix_x[:,6:9]),axis=1)
    dm_y = teach1.data_matrix_y + teach2.data_matrix_y + teach3.data_matrix_y
    
    loc = 0
    
    bs = 200 #batch_size
    tt = 200000 #training time (steps)
    lr= 0.01
    history = model.train(dm_x[loc:loc+bs*tt],dm_y[loc:loc+bs*tt],learning_rate=lr,batch_size=bs,verbose=verbose,epochs=1)
    loc += bs*tt
    
    bs = 1000 #batch_size
    tt = 20000 #training time (steps)
    lr= 0.01
    history = model.train(dm_x[loc:loc+bs*tt],dm_y[loc:loc+bs*tt],learning_rate=lr,batch_size=bs,verbose=verbose,epochs=1)
    loc += bs*tt
    
    bs = 4000 #batch_size
    tt = 20000 #training time (steps)
    lr= 0.001
    history = model.train(dm_x[loc:loc+bs*tt],dm_y[loc:loc+bs*tt],learning_rate=lr,batch_size=bs,verbose=verbose,epochs=1)
    loc += bs*tt
    
    
  
    x1,y1 = teach1.predict_from_matrix(batch_size=10000)
    x2,y2 = teach2.predict_from_matrix(batch_size=10000)
    x3,y3 = teach3.predict_from_matrix(batch_size=10000)
    x1 = np.concatenate((x1[:,0:3],x2[:,3:6],x3[:,6:9]),axis=1)
    y1 = y1+y2+y3
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

        n_list = [int(12*(1.3)**i) for i in range(0,7)]    
        r_list = np.arange(run*step,(run+1)*step)
        n_repeat = np.repeat(n_list,len(r_list))
        r_repeat = np.tile(r_list,len(n_list))
        pool_list = list(zip(n_repeat,r_repeat))    
        pool = multiprocessing.Pool(processes=len(pool_list))
        pool.map(main_f,pool_list)
        pool.close()

        
step = 3
for run in range(5):
    main_f_multiP(run)
    
    
#main_f((12,0))

"""
parser = argparse.ArgumentParser(description='p = (n,r)')
parser.add_argument("--p", default='(2,1)', help="enter tuple (n,r)")
args = parser.parse_args()
p = eval(args.p)

main_f(p)
"""
