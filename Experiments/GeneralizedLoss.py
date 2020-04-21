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
parser.add_argument("--p", default='1', type=int, help="enter --f=num_fea (integer)")
args = parser.parse_args()
power = args.p

f = 8
database_dir = 'teac600dataL2_20amb'
batch_size=400
train_steps = 400*1000
epochs = 1

#os.mkdir('data')

fol = 'data/features{}_{}'.format(f,power)
os.mkdir(fol)
os.mkdir(fol+'/losses')
os.mkdir(fol+'/models')
f_name = os.path.basename(__file__)
shutil.copyfile(f_name, fol+'/SourceCode.txt')



arc = [20,600,600,1]

max_fea = arc[0]
k = [1 if i<f else 0 for i in range(max_fea) ]

print('teacher architecture:',arc)

teach = teacher.Teacher(scale=k,architecture=arc,use_database=True,reduce=1,database_dir=database_dir,softmax=False) 
print("final teacher arch:",teach.architecture)

def run_model(n,train_steps=50*1000,verbose=2,lr=0.001,anneal=True,plot=False,batch_size=32,epochs=1):

    model = student.Model([20,n,n,1],softmax=False,loss='power',power=power)
    
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
    
    bs = 400 #batch_size
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
    if f<7: n_i = 8
    if f>6 and f<11: n_i = 12
    if f>10: n_i = 22
        
    if f%2==0:
        n_list = [int(n_i*(1.3)**i) for i in range(0,5)]    
        r_list = np.arange(run*step,(run+1)*step)
        n_repeat = np.repeat(n_list,len(r_list))
        r_repeat = np.tile(r_list,len(n_list))
        pool_list = list(zip(n_repeat,r_repeat))    
        pool = multiprocessing.Pool(processes=len(pool_list))
        pool.map(main_f,pool_list)
        pool.close()
        
        n_list = [int(n_i*(1.3)**i) for i in range(5,7)]    
        r_list = np.arange(run*step,(run+1)*step)
        n_repeat = np.repeat(n_list,len(r_list))
        r_repeat = np.tile(r_list,len(n_list))
        pool_list = list(zip(n_repeat,r_repeat))    
        pool = multiprocessing.Pool(processes=len(pool_list))
        pool.map(main_f,pool_list)
        pool.close()
        
        
    else:               
        n_list = [int(n_i*(1.3)**i) for i in range(5,7)]  
        n_list.reverse()
        r_list = np.arange(run*step,(run+1)*step)
        n_repeat = np.repeat(n_list,len(r_list))
        r_repeat = np.tile(r_list,len(n_list))
        pool_list = list(zip(n_repeat,r_repeat))    
        pool = multiprocessing.Pool(processes=len(pool_list))
        pool.map(main_f,pool_list)
        pool.close()
        
        n_list = [int(n_i*(1.3)**i) for i in range(0,5)]   
        n_list.reverse()
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
