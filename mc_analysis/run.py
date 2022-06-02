import os
from itertools import product 
import warnings
  
# adding a single entry into warnings filter
warnings.simplefilter('error', UserWarning)


m_models = [20,50,100]
n_instances = [100,300,100]

mc = [100]

lr = [1]
epochs = [10000]
n_iters = [0,1000,5000,10000]

#os.system('pip install birt-gd')
for i, j in zip(m_models, n_instances):
    for k in n_iters:
        print()
        print('python mc.py -mc {} -l {} -e {} -m {} -i {} -t {}'.format(str(mc[0]),str(lr[0]), str(epochs[0]), str(i), str(j), str(k)))
        os.system('python mc.py -mc {} -l {} -e {} -m {} -i {} -t {}'.format(str(mc[0]),str(lr[0]), str(epochs[0]), str(i), str(j), str(k)))
