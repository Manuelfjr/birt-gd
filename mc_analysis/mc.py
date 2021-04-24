import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
from DataTreat import Datasets
from birt import BIRTSGD

# Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs all the code''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--scale', dest='scale',
                        type=float,
                        default=1,
                        help= '''Scale to generate dataset.''')  
    parser.add_argument('-mc', '--mc_interations', dest='mc_interations',
                        type=int,
                        default=1,
                        help= '''Number of Monte Carlo iterations. If -mc 1, 
                                 isn't a Monte Carlo process.''')
    parser.add_argument('-r', '--random_seed', dest='random_seed',
                        type=int,
                        default=-1,
                        help= '''Set seed. It's int. If -r -1, generate random seed.''')                 
    parser.add_argument('-l', '--learning_rate', dest='learning_rate',
                        type=float,
                        default=0.1,
                        help= '''Learning rate to gradient descent.''')
    parser.add_argument('-e', '--epochs', dest='epochs',
                        type=int,
                        default=20,
                        help= '''Epochs to gradient descent.''')
    parser.add_argument('-b', '--batch_size', dest='batch_size',
                        type=int,
                        default=5,
                        help= '''Number of batchs to gradient descent.''')
    parser.add_argument('-m', '--n_models', dest='n_models',
                        type=int,
                        default=20,
                        help= '''Number of models to IRTSGD.''')
    parser.add_argument('-i', '--n_instances', dest='n_instances',
                        type=int,
                        default=100,
                        help= '''Number of instances.''')
    parser.add_argument('-t', '--n_inits', dest='n_inits',
                        type=int,
                        default=10,
                        help= '''Number of initializations.''')
    parser.add_argument('-w', '--n_workers', dest='n_workers',
                        type=int,
                        default=-1,
                        help= '''Number of cpu workers.''')  

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    mc = vars(args)['mc_interations']

    MC_PATH = 'mc_results'

    if vars(args)['random_seed'] == -1:
        #random_seed = list(np.random.randint(low=0, high=( (2)**(32) - 1 ), size=mc))
        random_seed = list(np.arange(0,mc))
    else:
        random_seed = [vars(args)['random_seed']]

    for n_iter in tqdm(range(mc)):
        Data = Datasets(
                n_models = vars(args)['n_models'], n_instances = vars(args)['n_instances'],
                random_seed = random_seed[n_iter], scale = vars(args)['scale']
                )
        
        X, y = Data.data_generate(scale = vars(args)['scale'])
        Data.data_generate_write(
            X, 
            y,
            params={ 
            '_thi': Data._thi,
            '_delj': Data._delj,
            '_aj': Data._aj,
            'n_iter':n_iter
            }
        )
        n_models, n_instances = vars(args)['n_models'], vars(args)['n_instances']
        
        irt = BIRTSGD(
            learning_rate = vars(args)['learning_rate'], 
            epochs = vars(args)['epochs'],
            n_models = n_models, n_instances = n_instances,
            batch_size = vars(args)['batch_size'], 
            n_inits = vars(args)['n_inits'], 
            n_workers = vars(args)['n_workers'], 
            random_seed = random_seed[n_iter],
            fixed_discrimination=False
        )

        _thi, _delj, _aj = irt.fit(X, y).get_params()

        # from IPython import embed
        # embed()
        #generated
        params = {
            'dataset_name': 'generated_data_mc',
            'thi': Data._thi,
            'delj': Data._delj,
            'aj': Data._aj,
            'y': y,
            '_thi': _thi,
            '_delj': _delj,
            '_aj': _aj,
            'mc_iterations': mc,
            'n_models': vars(args)['n_models'],
            'n_instances': vars(args)['n_instances'],
            'epochs': vars(args)['epochs'],
            'n_inits': vars(args)['n_inits'],
            'learning_rate': vars(args)['learning_rate'],
            'batchs': vars(args)['batch_size']
            }
        
        Data.mc_write(param = params, path = MC_PATH, mc = n_iter)

