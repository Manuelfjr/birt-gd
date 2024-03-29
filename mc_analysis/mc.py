import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
from DataTreat import Datasets
from birt import Beta4, Beta3
import time


# Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs all the code''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--scale', dest='scale',
                        type=float,
                        default=1,
                        help= '''Scale to generate dataset.''')
    parser.add_argument('-b', '--beta', dest='beta',
                        type=str,
                        default="beta4",
                        help= '''Which beta.''')
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
    #parser.add_argument('-b', '--batch_size', dest='batch_size',
    #                    type=int,
    #                    default=5,
    #                    help= '''Number of batchs to gradient descent.''')
    parser.add_argument('-m', '--n_respondents', dest='n_respondents',
                        type=int,
                        default=20,
                        help= '''Number of models to IRTSGD.''')
    parser.add_argument('-i', '--n_items', dest='n_items',
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
    parser.add_argument('-p', '--path', dest='path',
                        type=str,
                        default="mc_results",
                        help= '''Name of path of results.''')  

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    mc = vars(args)['mc_interations']

    MC_PATH = vars(args)["path"]

    if vars(args)['random_seed'] == -1:
        #random_seed = list(np.random.randint(low=0, high=( (2)**(32) - 1 ), size=mc))
        random_seed = list(np.arange(0,mc))
    else:
        random_seed = [vars(args)['random_seed']]

    for n_iter in tqdm(range(mc)):
        Data = Datasets(
                n_respondents = vars(args)['n_respondents'], n_items = vars(args)['n_items'],
                random_seed = random_seed[n_iter], scale = vars(args)['scale']
                )
        
        X, y = Data.data_generate(scale = vars(args)['scale'])
        responses = Data.data_generate_write(
            X, 
            y,
            params={ 
            '_thi': Data._thi,
            '_delj': Data._delj,
            '_aj': Data._aj,
            'n_iter':n_iter,
            'n_respondents': X[-1][1] + 1,
            'n_items': X[-1][0] + 1,
            }
        )
# b4 = Beta4(
#        learning_rate=1, 
#        epochs=5000,
#        n_respondents=pij.shape[1], 
#        n_items=pij.shape[0],
#        n_inits=1000, 
#        n_workers=-1,
#        random_seed=1,
#        tol=10**(-8),
#        set_priors=False
#    )
        
        n_respondents, n_items = vars(args)['n_respondents'], vars(args)['n_items']
        
        start = time.time()
        if vars(args)['beta'] == 'beta4':
            irt = Beta4(
                learning_rate = vars(args)['learning_rate'], 
                epochs = vars(args)['epochs'],
                n_respondents = n_respondents, 
                n_items = n_items,
                n_inits = vars(args)['n_inits'], 
                n_workers = vars(args)['n_workers'], 
                random_seed = random_seed[n_iter],
                set_priors=True,
                tol = 10 **(-8)
            )
        elif vars(args)['beta'] == 'beta3':
            irt = Beta3(
                learning_rate = vars(args)['learning_rate'], 
                epochs = vars(args)['epochs'],
                n_respondents = n_respondents, 
                n_items = n_items,
                n_inits = vars(args)['n_inits'], 
                n_workers = vars(args)['n_workers'], 
                random_seed = random_seed[n_iter],
                #set_priors=False,
                tol = 10 **(-8)
            )
            
        elif (vars(args)['beta']=='beta3fixed'):
            irt = Beta3(
                learning_rate = vars(args)['learning_rate'], 
                epochs = vars(args)['epochs'],
                n_respondents = n_respondents, 
                n_items = n_items,
                n_inits = 0, 
                n_workers = vars(args)['n_workers'], 
                random_seed = random_seed[n_iter],
                #set_priors=False,
                tol = 10 **(-8)
            )

        irt.fit(responses.values)
        done = time.time()
        time_stamp = done - start

        _thi, _delj, _aj = irt.get_params()

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
            'n_respondents': vars(args)['n_respondents'],
            'n_items': vars(args)['n_items'],
            'epochs': vars(args)['epochs'],
            'n_inits': vars(args)['n_inits'],
            'learning_rate': vars(args)['learning_rate'],
            'time_stamp': [time_stamp],
            'model': vars(args)['beta']
            #'batchs': vars(args)['batch_size']
            }
        
        Data.mc_write(param = params, path = MC_PATH, mc = n_iter)

