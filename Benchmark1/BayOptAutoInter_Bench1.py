import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
from ann_functions import getModel, kCrossVal, transfBestparam, kCrossValGP
from time import perf_counter
import tensorflow as tf

start = perf_counter()
seed = 10

start = perf_counter()
#Benchmark1
highfid = lambda x: (6.*x-2.)**2*np.sin(12.*x-4.)
lowfid  = lambda x: 0.5*highfid(x) + 10*(x-0.5) + 5.
Nhf = 5
Nlf = 32
N = Nhf + Nlf
xhf = np.linspace(0,1,Nhf)
xlf = np.linspace(0,1,Nlf)
xtot = np.concatenate((xhf,xlf))
Nepo = 1000

#Discontinuous Benchmark           
#lowfid = lambda x: (0.5*(6.*x-2.)**2*np.sin(12.*x-4)+10.*(x-0.5)-5.)*(x<0.5) + (3+0.5*(6.*x-2)**2*np.sin(12.*x-4)+10*(x-0.5)-5.)*(x>0.5)
#highfid = lambda x: (2*lowfid(x)- 20*x+20)*(x<0.5) + (4+2*lowfid(x)- 20*x+20)*(x>0.5)
#Nhf = 6
#Nlf = 32
#N = Nhf + Nlf
#xhf = np.linspace(0,1,Nhf)
#xlf = np.linspace(0,1,Nlf)
#NepoLF = 5000
#NepoHF = 1200
            
#Nonlinear correlation
#lowfid = lambda x: np.sin(8*pi*x)
#highfid = lambda x: (x-np.sqrt(2))*lowfid(x)**2
#Nhf = 15
#Nlf = 64
#N = Nhf + Nlf
#xhf = np.linspace(0,1,Nhf)
#xlf = np.linspace(0,1,Nlf)
#NepoLF = 3000
#NepoHF = 3000

#Normalization
x_test = np.linspace(0,1,1000)
x_big = np.linspace(0,1,1000000)
hfmean = np.mean(highfid(x_big))
lfmean = np.mean(lowfid(x_big))
yhf = highfid(xhf)
ylf = lowfid(xlf)
yhf_test = highfid(x_test) - hfmean
ylf_test = lowfid(x_test) - lfmean
yhf = yhf - hfmean
ylf = ylf - lfmean

K.clear_session()
name = 'Inter'
MAX_EVAL = 30
bayes_trials = Trials()
opt_list = ['Adam','Adamax','standardadam']
kernel_list = ['uniform','glorot_uniform']
aux_dic = {'opt': opt_list, 'kernel_init': kernel_list}
space = {
     'alpha' : hp.loguniform('alpha',np.log(0.0001),np.log(0.1)),
     'nodes' : hp.qloguniform('nodes',np.log(4),np.log(128),2),
     'epochs': hp.quniform('epochs',1,3,1),
     'l2weight' : hp.loguniform('l2weight',np.log(0.0001),np.log(1)),
     'lr': hp.loguniform('lr',np.log(0.0001),np.log(0.1)),
     'kernel_init': hp.choice('kernel_init',kernel_list),
     'opt' : hp.choice('opt',opt_list)}
x_train = np.concatenate((xhf,xlf))
yhf_train = np.concatenate((yhf,np.full(Nlf,-10)))
ylf_train = np.concatenate((np.full(Nhf,-10),ylf))

def objective(params):
    K.clear_session()
    print(params)
    CVres = kCrossValGP(Nhf,Nlf,Nepo,xhf,yhf,xlf,ylf,params,name)
    return {'loss' : CVres, 'params':params, 'status': STATUS_OK}

best_params = fmin(fn = objective,
                   space = space,
                   algo = tpe.suggest,
                   max_evals = MAX_EVAL,
                   trials = bayes_trials)

transfBestparam(best_params,aux_dic)
print('best parameters: ', best_params)
finalModel = getModel(best_params,name)
  
x_train = np.concatenate((xhf,xlf))
yhf_train = np.concatenate((yhf,np.full(Nlf,-10)))
ylf_train = np.concatenate((np.full(Nhf,-10),ylf))
tf.random.set_seed(seed)
hist = finalModel.fit(x_train,[yhf_train,ylf_train],epochs=Nepo*int(best_params['epochs']),batch_size=N,verbose=0, validation_data=(x_test,[yhf_test,ylf_test]))
y_test = finalModel.predict(x_test)

stop = perf_counter()
elapsed = stop - start
print('Elapsed time: ', elapsed)

plt.figure()
plt.subplot(2,1,1)
plt.plot(x_test,y_test[0],'k', label = 'HF pred')
plt.plot(xhf,yhf,'ro', label = 'HF train data')
plt.plot(x_test,highfid(x_test) - hfmean,'--k', label = 'HF exact')
plt.legend()
plt.title('High fidelity model ')
plt.subplot(2,1,2)
plt.plot(x_test,y_test[1],'k', label = 'HF solution')
plt.plot(xlf,ylf,'ro', label = 'LF train data')
plt.plot(x_test,lowfid(x_test) - lfmean,'--k', label = 'LF exact')
plt.legend()
plt.title('Low fidelity model')

plt.figure()
plt.subplot(2,1,1)
plt.plot(hist.history['HF_loss'],color='red',label='HF train mse')
plt.plot(hist.history['LF_loss'],color='black',label='LF train mse')
plt.legend()
plt.yscale('log')
plt.subplot(2,1,2)
plt.plot(hist.history['val_HF_loss'],color='red', label='HF valid mse')
plt.plot(hist.history['val_LF_loss'],color='black', label = 'LF valid mse')
plt.legend()
plt.yscale('log')
