import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ks_2samp, wilcoxon, ttest_ind
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
        
    def data_generate_write(self, X, y, params):
        names = ['mc_results/params', 'mc_results/data', params['n_iter']]
        data = pd.DataFrame( np.reshape(y, (X[-1][0] + 1, X[-1][1] + 1)) )
        df_abilities = pd.DataFrame(params['_thi'], columns=['_abilities'])
        df_j = pd.DataFrame(
            {
                "_difficulties": params['_delj'],
                "_discrimination": params['_aj']
            }
        )
        if (not os.path.exists(names[0]) ):
            os.makedirs(names[0])
        if (not os.path.exists(names[1]) ):
            os.makedirs(names[1])

        data.to_csv(names[1] + '/generate_data_iter_mc{}_i{}_m{}.csv'.format(names[-1],params['n_instances'],params['n_models']))
        df_abilities.to_csv(names[0] + '/generate_abilities_iter_mc{}_i{}_m{}.csv'.format(names[-1],params['n_instances'],params['n_models']))
        df_j.to_csv(names[0] + '/generate_diff_disc_iter_mc{}_i{}_m{}.csv'.format(names[-1],params['n_instances'],params['n_models']))
        return data

    def RSE(self, y_true, y_pred):
    	return sum( (y_pred - y_true)**(2) )/sum( (y_true - np.mean(y_true))**(2) )
    	
    def mc_write(self, param, path, **kwargs):
        path_generate = path
        #self.write(param, path)
        if not os.path.exists(path_generate):
            os.makedirs(path_generate)

        a_aj = np.logical_and( param["aj"] < 0 ,  param["_aj"] > 0)
        b_aj = np.logical_and(param["aj"] > 0,  param["_aj"] < 0)

        aj_changed_sign = sum(np.logical_or(a_aj,b_aj))/self.n_instances

        if kwargs['mc'] == 0:
            data = {
                'p.value_thi.ks': [ks_2samp(param['thi'],param['_thi']).pvalue],
                'p.value_delj.ks': [ks_2samp(param['delj'],param['_delj']).pvalue],
                'p.value_aj.ks': [ks_2samp(param['aj'],param['_aj']).pvalue],

                #'p.value_thi.wilcoxon': [wilcoxon(param['thi'],param['_thi']).pvalue],
                #'p.value_delj.wilcoxon': [wilcoxon(param['delj'],param['_delj']).pvalue],
                #'p.value_aj.wilcoxon': [wilcoxon(param['aj'],param['_aj']).pvalue],
                
                'RSE_thi': [self.RSE(param['thi'], param['_thi'])],
                'RSE_delj': [self.RSE(param['delj'], param['_delj'])],
                'RSE_aj': [self.RSE(param['aj'], param['_aj'])],

                'corr_thi_to_pred_thi': pearsonr(param['thi'],param['_thi'])[0],
                'corr_delj_to_pred_delj': pearsonr(param['delj'],param['_delj'])[0],
                'corr_aj_to_pred_aj': pearsonr(param['aj'],param['_aj'])[0],

                'aj_sign_changed': [aj_changed_sign],

                'time_stamp': param['time_stamp']
            }
            df = pd.DataFrame(data)

            # df_infs = pd.DataFrame(
            #     {
            #         'dataset_name': [param['dataset_name']],
            #         'learning_rate': [param['learning_rate']],
            #         'mc_iterations': [param['mc_iterations']],
            #         'n_epochs': [param['epochs']],
            #         'n_batchs': [param['batchs']],
            #         'n_models': [len(param['thi'])],
            #         'n_instances': [len(param['delj'])]
            #     }
            # )

            df.to_csv(path_generate + '/generate_data_mc{}_m{}_i{}_e{}_t{}_lr{}.csv'.format(
                param['mc_iterations'], param['n_models'], param['n_instances'], param['epochs'],
                param['n_inits'], param['learning_rate']
                )
                )
            #df_infs.to_csv( path_generate + '/generate_data_mc_{}_infs.csv'.format(param['mc_iterations']) )
        else:
            df = pd.read_csv(path_generate + '/generate_data_mc{}_m{}_i{}_e{}_t{}_lr{}.csv'.format(
                param['mc_iterations'], param['n_models'], param['n_instances'], param['epochs'],
                param['n_inits'], param['learning_rate']
                ),
                index_col=0)

            df.loc[ kwargs['mc'] ] = [
                ks_2samp(param['thi'],param['_thi']).pvalue,
                ks_2samp(param['delj'],param['_delj']).pvalue,
                ks_2samp(param['aj'],param['_aj']).pvalue,

                #wilcoxon(param['thi'],param['_thi']).pvalue,
                #wilcoxon(param['delj'],param['_delj']).pvalue,
                #wilcoxon(param['aj'],param['_aj']).pvalue,
                
                self.RSE(param['thi'], param['_thi']),
                self.RSE(param['delj'], param['_delj']),
                self.RSE(param['aj'], param['_aj']),

                pearsonr(param['thi'],param['_thi'])[0],
                pearsonr(param['delj'],param['_delj'])[0],
                pearsonr(param['aj'],param['_aj'])[0],

                aj_changed_sign,
                
                param['time_stamp'][0]
            ]

            df.to_csv(path_generate + '/generate_data_mc{}_m{}_i{}_e{}_t{}_lr{}.csv'.format(
                param['mc_iterations'], param['n_models'], param['n_instances'], param['epochs'],
                param['n_inits'], param['learning_rate']
                )
                )
