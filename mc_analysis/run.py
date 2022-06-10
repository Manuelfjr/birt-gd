import os
from itertools import product 
import warnings
import argparse
from tqdm import tqdm  
# adding a single entry into warnings filter
warnings.simplefilter('error', UserWarning)

# Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs all the code''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--n_models', dest='n_models',
                        type=int,nargs="+",required=True,
                        default=20,
                        help= '''Number of models to IRTSGD.''')
    parser.add_argument('-i', '--n_instances', dest='n_instances',
                        type=int,nargs="+",required=True,
                        default=100,
                        help= '''Number of instances.''')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    m_models = vars(args)['n_models']
    n_instances = vars(args)['n_instances']

    mc = [100]

    lr = [1]
    epochs = [10000]
    n_iters = [0,1000,5000,10000]

    for model in tqdm(['beta4','beta3','beta3fixed']):
        for i in tqdm(range(len(m_models)), desc="Iteration" ):
            for k in n_iters:
                print() 
                print('python3 mc.py -mc {} -l {} -e {} -m {} -i {} -t {} -b {}'.format(str(mc[0]),str(lr[0]), str(epochs[0]), str(m_models[i]), str(n_instances[i]), str(k),str(model)))
                os.system('python3 mc.py -mc {} -l {} -e {} -m {} -i {} -t {} -b {}'.format(str(mc[0]),str(lr[0]), str(epochs[0]), str(m_models[i]), str(n_instances[i]), str(k),str(model)))
