import os
from itertools import product 
import warnings
from tqdm import tqdm  
# adding a single entry into warnings filter
warnings.simplefilter('error', UserWarning)


m_models = [20,50,100]
n_instances = [100,300,100]

mc = [100]

lr = [1]
epochs = [10000]
n_iters = [0,1000,5000,10000]

for model in tqdm(['beta3','beta3fixed']):
    for i in tqdm(range(len(m_models)), desc="Iteration" ):
        for k in n_iters:
            print()
            print('python3 mc.py -mc {} -l {} -e {} -m {} -i {} -t {} -b {}'.format(str(mc[0]),str(lr[0]), str(epochs[0]), str(m_models[i]), str(n_instances[i]), str(k),model))
            os.system('python3 mc.py -mc {} -l {} -e {} -m {} -i {} -t {} -b'.format(str(mc[0]),str(lr[0]), str(epochs[0]), str(m_models[i]), str(n_instances[i]), str(k),model))
