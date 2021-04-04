import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ks_2samp
import os 

class Datasets:
    def __init__(self, transform=None, name=None, n_models=20, n_instances=100, random_seed=-1, scale=1):
        self.transform = transform
        self.name = name
        self.n_models = n_models
        self.n_instances = n_instances
        self.n_seed = random_seed
        self.scale = scale
    
    def _irt(self, thi, delj, aj):
        alphaij, betaij = [], []
        for d, a in zip(delj, aj):
            for t in thi:
                alphaij.append( (t/d)**(a) )
                betaij.append( ( ((1 - t)/(1 - d))**(a)) )

        y_est = np.array(alphaij)/(np.array(alphaij) + np.array(betaij))
        return y_est

    def data_generate(self, scale):
        np.random.seed(self.n_seed)
        self._thi = np.random.beta(1,1,size = self.n_models)
        self._delj = np.random.beta(1,1,size = self.n_instances)
        self._aj = np.random.normal(1,self.scale, size = self.n_instances)

        alphaij, betaij = [], []

        X, y = [], []
        j, i  = 0, 0
        for delj, aj in zip(self._delj, self._aj):
            for thi in self._thi:
                alphaij.append( (thi/delj)**(aj) )
                betaij.append( ((1 - thi)/(1 - delj))**(aj) )
                X.append((j ,i))
                i += 1
            i = 0
            j += 1

        for alpha,beta in zip(alphaij, betaij):
            y.append( np.random.beta(alpha,beta, size=1)[0] )

        #print(X)
        return X, y
    
    def mc_write(self, param, path, **kwargs):
        path_generate = path
        #self.write(param, path)
        if not os.path.exists(path_generate):
            os.makedirs(path_generate)
        if kwargs['mc'] == 0:
            df = pd.DataFrame(
                {
                    'p.value_thi': ks_2samp(param['thi'],param['_thi']),
                    'p.value_delj': ks_2samp(param['delj'],param['_delj']),
                    'p.value_aj': ks_2samp(param['aj'],param['_aj']),
                    'corr_thi_to_pred_thi': pearsonr(param['thi'],param['_thi']),
                    'corr_delj_to_pred_delj': pearsonr(param['delj'],param['_delj']),
                    'corr_aj_to_pred_aj': pearsonr(param['aj'],param['_aj'])
                }
            )

            df_infs = pd.DataFrame(
                {
                    'dataset_name': [param['dataset_name']],
                    'learning_rate': [param['learning_rate']],
                    'mc_iterations': [param['mc_iterations']],
                    'n_epochs': [param['epochs']],
                    'n_batchs': [param['batchs']],
                    'n_models': [len(param['thi'])],
                    'n_instances': [len(param['delj'])]
                }
            )

            df.to_csv( path_generate + '/generate_data_mc_{}.csv'.format(param['mc_iterations']) )
            df_infs.to_csv( path_generate + '/generate_data_mc_{}_infs.csv'.format(param['mc_iterations']) )
        else:
            df = pd.read_csv(path_generate + '/generate_data_mc_{}.csv'.format(param['mc_iterations']), index_col=0)

            df.loc[ kwargs['mc'] ] = [
                ks_2samp(param['thi'],param['_thi'])[1],
                ks_2samp(param['delj'],param['_delj'])[1],
                ks_2samp(param['aj'],param['_aj'])[1],

                pearsonr(param['thi'],param['_thi'])[0],
                pearsonr(param['delj'],param['_delj'])[0],
                pearsonr(param['aj'],param['_aj'])[0]
            ]
            
            df.to_csv(path_generate + '/generate_data_mc_{}.csv'.format(param['mc_iterations']))

    def data_split(self):
        X, y = [], []
        for row in range(self.data.shape[0]):
            for col in range(self.data.shape[1]):
                X.append((row,col))
                y.append(self.data.iloc[row,col])
        return X, y