from multiprocessing import Process, Queue, Value

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import numpy as np


class BIRTGD:
    """Beta3-IRT with Gradient descent.

    Read more in the https://github.com/Manuelfjr/BIRTGD .

    Parameters:
    -------------------------------------------------------

    learning_rate : float, default=0.1
        It's a learning rate to Gradient descent.

    epochs : int, default=20
        Numbers of epochs to fit the model.

    n_models : int, default=20
        Numbers of models to be evaluated.

    n_instances : int, default=100
        Numbers of instances to fit the models.

    random_seed : int, default=1
        Determines a random state to generation initial kicks.
        
    n_inits : int, default=10
        Determines the number of initializations.
        
    n_workers : int, default=-1
        Determines the number of CPUs to use.

    
    Attributes
    -------------------------------------------------------
    self.abilities : ResourceVariable of shape (n_models,)
        It is the optimized estimate for the abilitie parameter.
    
    self.difficulties : ResourceVariable of shape (n_instances,)
        It is the optimized estimate for the difficulties parameter.

    self.discriminations : ResourceVariable of shape (n_instances,)
        It is the optimized estimate for the discrimination parameter.


    Notes
    -------------------------------------------------------


    Example
    -------------------------------------------------------
    >>> from birt import BIRTGD
    >>> data = pd.DataFrame({'a': [0.99,0.89,0.87], 'b': [0.32,0.25,0.45]})
    >>> bgd = BIRTGD(n_models=2, n_instances=3, random_seed=1)
    >>> bgd.fit(data)
    100%|██████████| 5000/5000 [00:22<00:00, 219.50it/s]
    <birt.BIRTGD at 0x7f6131326c10>
    >>> bsgd.abilities
    array([0.90438306, 0.27729774], dtype=float32)
    >>> bgd.difficulties
    array([0.3760659, 0.5364428, 0.34256178], dtype=float32)
    >>> bgd.discriminations
    array([1.6690203, 0.9951777, 0.65577406], dtype=float32)
    """
    def __init__(
        self, learning_rate=1, 
        epochs=5000, n_models=20, 
        n_instances=100,
        n_inits=1000, n_workers=-1,
        random_seed=1
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.n_models = n_models
        self.n_instances = n_instances
        self.n_seed = random_seed
        self.n_inits = n_inits
        self.n_workers = n_workers
        self._params = {
            'learning_rate': learning_rate,
            'epochs': epochs,
            'n_models': n_models,
            'n_instances': n_instances
        }
    
    def get_params(self):
        """Get parameters

        Parameters
        -------------------------------------------------------
        self : self objetc

        Returns
        -------------------------------------------------------
        tuple of (abilities, difficulties, discrimination).
        """
        return self.abilities, self.difficulties, self.discriminations
    
    def fit(self, X, y=None):
        """Compute BIRT Gradient Descent
        
        Parameters
        -------------------------------------------------------
        X : list, array or tensor of shape (n_instances,n_models)
            

        Returns
        -------------------------------------------------------
        self
            Adjusted values of the parameters
        """
        self.pij = X
        queue = Queue()

        args = (
            queue, X, self.n_models, 
            self.n_instances,
            self.epochs, self.lr, self.n_seed, 
            self.n_inits
        )

        p = Process(target=_fit, args=list(args))
        p.start()
        abi, dif, dis = queue.get()
        p.join()

        self.abilities = abi
        self.difficulties = dif
        self.discriminations = dis
        X, Y, self.pijhat = _split(self.pij, self.predict(), (len(self.discriminations), len(self.abilities)))
        self.score(X, Y)
        return self
    
    def irt(self, abilities, difficulties, discriminations):
        """calculating a probability (i) deduction from (j) using Beta3-IRT

        Parameters
        -------------------------------------------------------
        abilities : 
                Optimized ability estimation by gradient descent using Beta3-IRT

        difficulties : 
                Optimized difficulties estimation by gradient descent using Beta3-IRT

        discriminations : 
                Optimized discriminations estimation by gradient descent using Beta3-IRT

        Returns
        -------------------------------------------------------
        Calculate E[pij | abilities, difficulties, discriminations]
        """
        alphaij = (abilities/difficulties)**(discriminations)
        betaij = ( ((1 - abilities)/(1 - difficulties))**(discriminations) )
        return alphaij/(alphaij + betaij)

    def predict(self, X=None):
        """Predict E[pij | abilities, difficulties, discrimination] = pij
            
        Parameters
        -------------------------------------------------------
        X : 
            None

        Returns
        -------------------------------------------------------
        return 
                y_pred = E[pij | abilities, difficulties, discrimination]
        """
        discriminations = self.discriminations 

        y_pred = []
        for d,a in zip(self.difficulties, discriminations):
            for t in self.abilities:
                y_pred.append(
                    self.irt(
                        abilities = t, 
                        difficulties = d, 
                        discriminations = a
                    )
                )

        return y_pred
    
    def score(self, X=None, Y=None):
        """Score is Pseudo-R2
            
        Parameters
        -------------------------------------------------------
        self.

        X : array.
            Contains pij adjusted.

        Y : array.
            Contains pij real.

        Returns
        -------------------------------------------------------
            Calculate the Pseudo-R2 based a OLS model.
        """
        model = LinearRegression()
        model.fit(X,Y)
        self.score = model.score(X, Y)
        return self.score

    def summary(self, show=False):
        """Summary of results
            
        Parameters
        -------------------------------------------------------
        self.

        show : boolean.
            Show hyperparams (abilities, difficulties and discrimination).

        Returns
        -------------------------------------------------------
            print summarize of the hyperparams and p_ij.
        """
        abi, dif, dis = self.get_params()
        
        out = '''
        ESTIMATES
        -----
                        | Min      1Qt      Median   3Qt      Max      Std.Dev
        Ability         | {0:.5f}  {1:.5f}  {2:.5f}  {3:.5f}  {4:.5f}  {5:.5f}
        Difficulty      | {6:.5f}  {7:.5f}  {8:.5f}  {9:.5f}  {10:.5f}  {11:.5f}
        Discrimination  | {12:.5f}  {13:.5f}  {14:.5f}  {15:.5f}  {16:.5f}  {17:.5f}
        pij             | {18:.5f}  {19:.5f}  {20:.5f}  {21:.5f}  {22:.5f}  {23:.5f}
        -----
        Pseudo-R2       | {24:.5f}
        '''.format(
                    np.min(abi), np.quantile(abi,0.25),np.median(abi), np.quantile(abi,0.75),np.max(abi), np.std(abi),
                    np.min(dif), np.quantile(dif,0.25),np.median(dif), np.quantile(dif,0.75),np.max(dif), np.std(dif),
                    np.min(dis), np.quantile(dis,0.25),np.median(dis), np.quantile(dis,0.75),np.max(dis), np.std(dis),
                    np.min(self.pijhat.values), np.quantile(self.pijhat.values,0.25),np.median(self.pijhat.values), np.quantile(self.pijhat.values,0.75),np.max(self.pijhat.values), np.std(self.pijhat.values),
                    self.score
                )
        if show == False:
            print(out)
        else:
            hyper = '''abilities: {},\n\t\tdifficulti1es: {},\n\t\tdiscriminations: {}\n\t-----
                    '''.format(abi, dif, dis)
            print(out + hyper)

    def plot(self, xaxis=None, yaxis=None, kwargs={}, ann=False):
        """Plot's of results
            
        Parameters
        -------------------------------------------------------
        xaxis : {'discrimination','difficulties','abilities'}.
                x axis of plot.
        yaxis : {'discrimination','difficulties', 'average_response', 'average_item'}
                y axis of plot.
        kwargs: dict.
            other Matplotlib library arguments.
        ann   : boolean.
            data points with annotations

        Returns
        -------------------------------------------------------
            Scatterplot of xaxis vs yaxis.
        """
        pij = pd.DataFrame(np.reshape(self.predict(), (len(self.discriminations), len(self.abilities))))

        sns.set_style('darkgrid')
        plt.figure(figsize=(13,6))
        if (xaxis == 'discrimination') and (yaxis == 'difficulty'):
            if ann == False:
                sns.scatterplot(x=self.discriminations, y=self.difficulties, **kwargs)
            else:
                sns.scatterplot(x=self.discriminations, y=self.difficulties, **kwargs)
                for i in range(pij.shape[0]):
                    plt.text(x = self.discriminations[i]+0.007, y = self.difficulties[i]+0.007, s='n{}'.format(i+1))
        elif (yaxis == 'discrimination') and (xaxis == 'difficulty'):
            if ann == False:
                sns.scatterplot(x=self.difficulties, y=self.discriminations, **kwargs)
            else:
                sns.scatterplot(x=self.difficulties, y=self.discriminations, **kwargs)
                for i in range(pij.shape[0]):
                    plt.text(x = self.difficulties[i]+0.007, y = self.discriminations[i]+0.007, s='n{}'.format(i+1))
        elif (xaxis == 'ability') and (yaxis == 'average_response'):
            if ann == False:
                sns.scatterplot(x=self.abilities, y=pij.apply(np.mean,axis=0), **kwargs)
            else:
                sns.scatterplot(x=self.abilities, y=pij.apply(np.mean,axis=0), **kwargs)
                for i in range(pij.shape[1]):
                    plt.text(x = self.abilities[i]+0.003, y =pij.apply(np.mean,axis=0)[i]+0.003, s='{}'.format(self.pij.columns[i]))
        elif (xaxis == 'difficulty') and (yaxis == 'average_item'):
            if ann == False:
                sns.scatterplot(x=self.difficulties, y=pij.apply(np.mean,axis=1), **kwargs)
            else:
                sns.scatterplot(x=self.difficulties, y=pij.apply(np.mean,axis=1), **kwargs)
                for i in range(pij.shape[0]):
                    plt.text(x = self.difficulties[i]+0.004, y =pij.apply(np.mean,axis=1)[i]+0.004, s='n{}'.format(i+1))
        elif xaxis == yaxis:
            raise ValueError('xaxis and yaxis are the same')
        elif xaxis not in ['discrimination','difficulties','abilities'] or yaxis not in ['discrimination','difficulties', 'average_response', 'average_item']:
            raise ValueError(f'{xaxis} or {yaxis} doesnt exists')
        else:
            raise ValueError(f'plotting {xaxis} vs {yaxis} is impossible.')
        plt.title(f'{xaxis} vs {yaxis}')
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
    
    def boxplot(self, x=None, y=None, kwargs={}):
        """Boxplot's of results
            
        Parameters
        -------------------------------------------------------
        x : ['discrimination','difficulties','abilities', None].
                x axis of plot.
        y : ['discrimination','difficulties','abilities', None]
                y axis of plot.
        kwargs: dict.
            other Matplotlib library arguments.

        Returns
        -------------------------------------------------------
            Boxplot of x axis vs y axis.
        """

        abi = pd.DataFrame({'abilities': self.abilities})
        dif_dis = pd.DataFrame({'difficulties': self.difficulties, 'discriminations': self.discriminations})
        sns.set_style('darkgrid')
        plt.figure(figsize=(13,6))
        if x == 'abilities':
            if y ==None:
                sns.boxplot(x=x,y=y, data=abi, **kwargs)
            elif y == 'discriminations' or y == 'difficulties':
                raise ValueError(f'the length of x is different from y.')
            elif y == 'abilities':
                raise ValueError(f'both X and Y are the same')
        elif x == 'difficulties':
            if y == None or y == 'discriminations':
                sns.boxplot(x=x,y=y, data=dif_dis, **kwargs)
            elif y == 'difficulties':
                raise ValueError(f'both X and Y are the same')
            elif y == 'abilities':
                raise ValueError(f'the length of x is different from y. ({dif_dis.shape[0], abi.shape[0]})')
        elif x == 'discriminations':
            if y == None or y == 'difficulties':
                sns.boxplot(x=x,y=y, data=dif_dis, **kwargs)
            elif y == 'discriminations':
                raise ValueError(f'both X and Y are the same')
            elif y == 'abilities':
                raise ValueError(f'the length of x is different from y. ({dif_dis.shape[0], abi.shape[0]})')
        elif x == None:
            if y == 'abilities':
                sns.boxplot(x=x,y=y, data=abi, **kwargs)
            elif y == 'difficulties' or y == 'discriminations':
                sns.boxplot(x=x,y=y, data=dif_dis, **kwargs)
            elif y == 'None':
                raise ValueError(f'both X and Y are None')
        elif y == None and (x not in ['abilities', 'difficulties', 'discriminations']):
            raise ValueError(f'y is None but {x} doesnot exist')
        elif x == None and (y not in ['abilities', 'difficulties', 'discriminations']):
            raise ValueError(f'x is None but {y} doesnot exist')
        elif x != None and y != None:
            if (x == 'abilities' and y == 'difficulties') or (x == 'difficulties' and y == 'abilities'):
                raise ValueError(f'the length of x is different from y.')
            

def _loss(y_true, y_pred):
    """Calculate Binary Cross Entropy (loss function)

    Parameters
    -------------------------------------------------------
    y_true : tf.Tensor 
            Real valued Tensor containing the ground truth

    y_pred : tf.Tensor
            Real valued Tensor containing predictions

    Returns 
    -------------------------------------------------------
    EagerTensor with loss functions value
    """
    loss = tf.keras.losses.BinaryCrossentropy()
    return loss(y_true, y_pred)

def _irt(thi, delj, aj, bj):
    """Calculating a probability (i) deduction from (j) using Beta3-IRT

    Parameters
    -------------------------------------------------------
    thi : 
            Parameter to be estimated from the abilitie for Beta3-IRT

    delj : 
            Parameter to be estimated from the difficulty for Beta3-IRT

    aj : 
            Other parameter of discrimination to be estimated to Beta3-IRT.

    bj : 
            Other parameter of discrimination to be estimated to Beta3-IRT.

    Returns
    -------------------------------------------------------
    Calculate E[pij | abilities, difficulties, discrimination = aj * bj]
    """

    sig_thi, sig_delj = tf.math.sigmoid(thi), tf.math.sigmoid(delj)
    tanh_bj, soft_aj = tf.math.tanh(bj), tf.math.softplus(aj)

    term1 = sig_delj / (1 - sig_delj)
    term2 = (1 - sig_thi) / sig_thi
    mult = tf.tensordot(term1, term2, axes=1)

    est = 1 / (1 + mult ** (tanh_bj * soft_aj))
    return est

def _fit(*args):
    """Train BIRT Gradient Descent

    Parameters
    -------------------------------------------------------
    args : tuple containing 
            (queue, X, n_models, n_instances, epochs, lr, random_seed, n_inits)

    Returns
    -------------------------------------------------------
    self
    """

    (queue, X, n_models, n_instances, epochs, lr, random_seed, n_inits) = args

    np.random.seed(random_seed)
    tf.random.set_seed(
        random_seed
    )

    X = tf.Variable(
        X, 
        trainable=False, dtype=tf.float32
    )


    thi = tf.Variable(
        np.random.normal(0,1, size=(1, n_models)), 
        trainable=True, dtype=tf.float32
    )
    
    delj = tf.Variable(
        np.random.normal(0,1, size=(n_instances, 1)), 
        trainable=True, dtype=tf.float32
    )
    
    bj = tf.Variable(
        np.ones((n_instances, 1)) * 0.5096851, 
        trainable=True, dtype=tf.float32
    )

    aj = tf.Variable(   
        np.ones((n_instances, 1)) * 2.002374, 
        trainable=True, dtype=tf.float32
    )
    
    t = 0
    for _ in tqdm(range(epochs)):
        if t < n_inits:
            variables = [
                thi,
                delj
            ]
        else: 
            variables = [
                thi,
                delj,
                aj,
                bj
            ]
        
        with tf.GradientTape() as g:
            g.watch(variables)
            pred = _irt(thi = thi, delj = delj, aj = aj, bj = bj)

            current_loss = _loss(
                X, pred
            )
        gradients = g.gradient(current_loss, variables)
        if t < n_inits:
            _thi, _delj = gradients
        else:
            _thi, _delj, _aj, _bj = gradients
            aj.assign_sub(tf.math.scalar_mul(lr, _aj))
            bj.assign_sub(tf.math.scalar_mul(lr, _bj))
        thi.assign_sub(tf.math.scalar_mul(lr, _thi))
        delj.assign_sub(tf.math.scalar_mul(lr, _delj))

        t += 1
        
    
    abilities = tf.math.sigmoid(thi).numpy().flatten()
    difficulties = tf.math.sigmoid(delj).numpy().flatten()
    discriminations = tf.math.tanh(bj) * tf.math.softplus(aj)
    discriminations = discriminations.numpy().flatten()

    parameters = (abilities, difficulties, discriminations)

    queue.put(parameters)

def _split(*args):
    (pij, predict, shape) = args
    pijhat = pd.DataFrame( np.reshape(predict, (shape[0], shape[1])) )

    shape = np.shape(pij)
    try:
        Y, X = np.reshape(pij.values,(shape[0]*shape[1], 1)), np.reshape(pijhat.values,(shape[0]*shape[1], 1))
    except:
        Y, X = np.reshape(pij,(shape[0]*shape[1], 1)), np.reshape(pijhat.values,(shape[0]*shape[1], 1))
    return Y, X, pijhat


#m, n = 2, 150
#t1 = np.random.beta(1,0.1,size = 1)
#t2 = np.random.beta(1,5,size = 1)
#d = np.random.beta(1,1,size = n)
#a = np.random.normal(1,1,size = n)

#alpha_1 = (t1/d)**(a)
#beta_1 = ((1 - t1)/(1 - d))**(a)

#alpha_2 = (t2/d)**(a)
#beta_2 = ((1 - t2)/(1 - d))**(a)

#data = pd.DataFrame({'a': np.random.beta(alpha_1, beta_1, n), 'b': np.random.beta(alpha_2, beta_2, n)})
#data = pd.DataFrame({'a': [0.12,0.3,0.2], 'b': [0.98,0.89,0.7899]})

#ep = 5000
#birt = BIRTGD(learning_rate=0.1, epochs=ep, n_instances=data.shape[0], n_models=data.shape[1], n_inits=1000)
#birt.fit(data)
#birt.summary(show = False)
#birt.plot(xaxis='discrimination', yaxis='difficulties', kwargs={'palette': 'Greys'}, ann=True)
#birt.plot(xaxis='difficulties', yaxis='discrimination', kwargs={'palette': 'Greys'}, ann=True)
#birt.plot(xaxis='abilities', yaxis='average_response', kwargs={'palette': 'Greys'}, ann=True)
#birt.plot(xaxis='difficulties', yaxis='average_item', kwargs={'palette': 'Greys'}, ann=True)

#print((t1,t2), birt.abilities)
