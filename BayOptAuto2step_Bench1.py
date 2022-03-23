import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
from ann_functions import getModel, kCrossVal, transfBestparam
from time import perf_counter
import tensorflow as tf

start = perf_counter()
seed = 10

#Benchmark1
highfid = lambda x: (6.*x-2.)**2*np.sin(12.*x-4.)
lowfid  = lambda x: 0.5*highfid(x) + 10*(x-0.5) + 5.
Nhf = 5
Nlf = 32
N = Nhf + Nlf
xhf = np.linspace(0,1,Nhf)
xlf = np.linspace(0,1,Nlf)
NepoLF = 1000
NepoHF = 500

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



### Normalization
x_big = np.linspace(0,1,1000000)
hfmean = np.mean(highfid(x_big))
lfmean = np.mean(lowfid(x_big))
x_test = np.linspace(0,1,1000)
Yhf = highfid(xhf)
Ylf = lowfid(xlf)
Yhf = Yhf - hfmean
Ylf = Ylf - lfmean

### LF
K.clear_session()
bestLF_params = {'lr' : 0.0255, 'kernel_init' : 'glorot_uniform', 'opt' : 'Adam'}
modelLF = getModel(bestLF_params,'LF')
tf.random.set_seed(seed)
histLF = modelLF.fit(xlf,Ylf,epochs=NepoLF,batch_size=Nlf,verbose=0)
yLF = modelLF.predict(x_test)[:,0]


### 2-step
xhf_help = modelLF.predict(xhf)[:,0]
xhf = np.vstack((xhf,xhf_help)).transpose()

x_test_help = modelLF.predict(x_test)[:,0]
x_test_in = np.vstack((x_test,x_test_help)).transpose()
y_test_in = highfid(x_test) - hfmean

name = '2step'
MAX_EVAL = 30
bayes_trials = Trials()
opt_list = ['Adam','Adamax']
kernel_list = ['uniform','glorot_uniform']
aux_dic = {'opt': opt_list, 'kernel_init': kernel_list}
space = {
     'nodes' : scope.int(hp.qloguniform('nodes',np.log(16),np.log(128),2)),
     'l2weight' : hp.loguniform('l2weight',np.log(0.0001),np.log(0.1)),
     'lr': hp.loguniform('lr',np.log(0.0001),np.log(0.1)),
     'kernel_init': hp.choice('kernel_init',kernel_list),
     'opt' : hp.choice('opt',opt_list)}

def objective(params):
    K.clear_session()
    print(params)
    CVres = kCrossVal(Nhf,NepoHF,xhf,Yhf,params,name)
    return {'loss' : CVres, 'params':params, 'status': STATUS_OK}

best_params = fmin(fn = objective,
                   space = space,
                   algo = tpe.suggest,
                   max_evals = MAX_EVAL,
                   trials = bayes_trials)

K.clear_session()
transfBestparam(best_params,aux_dic)
print('best parameters: ', best_params)
finalModel = getModel(best_params,name)
hist = finalModel.fit(xhf,Yhf,epochs=NepoHF,batch_size=Nhf,verbose=0, validation_data=[x_test_in,y_test_in])
yHF = finalModel.predict(x_test_in)

stop = perf_counter()
elapsed = stop - start
print('Elapsed time: ', elapsed)

### Plots
plt.figure()
plt.plot(x_test,yLF,'k', label = 'pred LF')
plt.plot(xlf,Ylf,'ro', label = 'LF training data')
plt.plot(x_test,lowfid(x_test) - lfmean,'--k', label = 'exact LF')
plt.legend()
plt.title('Low fidelity model')


plt.figure()
plt.plot(x_test,yHF,'k', label = 'pred HF')
plt.plot(xhf[:,0],Yhf,'ro', label = 'HF training data')
plt.plot(x_test,highfid(x_test) - hfmean,'--k', label = 'exact HF')
plt.legend()
plt.title('High fidelity model - 2-step')

plt.figure()
plt.subplot(2,1,1)
plt.plot(hist.history['loss'],color='red',label='HF train mse')
plt.plot(histLF.history['loss'],color='black',label='LF train mse')
plt.legend()
plt.yscale('log')
plt.subplot(2,1,2)
plt.plot(hist.history['val_loss'],color='red', label = 'HF validation mse')
plt.legend()
plt.yscale('log')