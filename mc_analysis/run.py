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
    parser.add_argument('-m', '--n_respondents', dest='n_respondents',
                        type=int,nargs="+",required=True,
                        default=20,
                        help= '''Number of respondents to IRTSGD.''')
    parser.add_argument('-i', '--n_items', dest='n_items',
                        type=int,nargs="+",required=True,
                        default=100,
                        help= '''Number of items.''')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    m_models = vars(args)['n_respondents']
    n_items = vars(args)['n_items']
    print(m_models, n_items)
    if (len(m_models) == 1)and(len(n_items) == 1):
        MC_PATH = "mc_i{}_m{}_results".format(n_items[0],m_models[0])
    else:
        MC_PATH = '.'
    mc = [100]

    lr = [1]
    epochs = [10000]
    n_iters = [1000]#[0,1000,5000,10000]

    #for model in tqdm(['beta4','beta3','beta3fixed']):
    for model in tqdm(['beta3','beta3fixed']):
        for i in tqdm(range(len(m_models)), desc="Iteration" ):
            for k in n_iters:
                print() 
                print('python3 mc.py -mc {} -l {} -e {} -m {} -i {} -t {} -b {} -p {}'.format(str(mc[0]),str(lr[0]), str(epochs[0]), str(m_models[i]), str(n_items[i]), str(k),str(model), MC_PATH) )
                os.system('python3 mc.py -mc {} -l {} -e {} -m {} -i {} -t {} -b {} -p {}'.format(str(mc[0]),str(lr[0]), str(epochs[0]), str(m_models[i]), str(n_items[i]), str(k),str(model), MC_PATH))
