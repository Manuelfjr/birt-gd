import os
from multiprocessing import cpu_count, Pool

import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

class BIRTSGD:
    """Beta3-IRT with Gradient descent.

    Read more in the https://github.com/Manuelfjr/BIRTSGD .

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

    batch_size : int, default=5
        Batch size for each iteration of the model fit. (batch_size <= n_instances).

    random_seed : int, default=1
        Determines a random state to generation initial kicks.
    
    fixed_discrimination : Boolean, default=False
        Determines if discrimination is 1 (discrimination=1). If False, discrimination=1.
    
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
    >>> from BIRTSGD.birt import BIRTSGD
    >>> X = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    >>> Y = [0.98,0.81,0.12,0.567,0.76,0.9]
    >>> bsgd = BIRTSGD(n_models=3, n_instances=2, random_seed=1)
    >>> bsgd.fit(X,Y)
    100%|██████████| 20/20 [00:00<00:00, 52.58it/s]
    <birt.BIRTSGD at 0x7f6ce2555f50>
    >>> bsgd.abilities
    array([0.78665066, 0.5025896 , 0.545207  ], dtype=float32)
    >>> bsgd.difficulties
    array([0.25070453, 0.46883535], dtype=float32)
    >>> bsgd.discriminations
    array([0.09374281, 1.4122988 ], dtype=float32)
    """
    def __init__(
        self, learning_rate=0.1, 
        epochs=20, n_models=20, 
        n_instances=100, batch_size=5, 
        n_inits=10, n_workers=-1,
        random_seed=1, fixed_discrimination=False
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.n_models = n_models
        self.n_instances = n_instances
        self.batch_size = batch_size
        self.n_seed = random_seed
        self.n_inits = n_inits
        self.n_workers = n_workers
        self.fixed_discrimination = fixed_discrimination
        self._params = {
            'learning_rate': learning_rate,
            'epochs': epochs,
            'n_models': n_models,
            'n_instances': n_instances,
            'batch_size':  batch_size
        }

    def _transform(self, data):
        """Separates the data set into X, with a list of tuples and Y with a list of p_ {ij}
            
        Parameters
        -------------------------------------------------------
        data :
                all adataset (DataFrame type), like:
                model_00  | model_01  | model_02  | ... | model_n
                -------------------------------------------------
                p00       | p10       | p20       | ... | pi0  
                -------------------------------------------------
                p01       | p11       | p21       | ... | pi1  
                -------------------------------------------------
                p02       | p12       | p22       | ... | pi2
                -------------------------------------------------
                ...
                ...
                ...
                -------------------------------------------------
                p0j       | p1j       | p2j       | ... | pij  
        
        Returns
        -------------------------------------------------------
        X : 
            a list with set of tuples (n_instance=i, n_model=j), like:
            [
                (0, 0),
                (0, 1),
                (0, 2),
                ...
                (0, i),
                (1, 0),
                (1, 1),
                (1, 2),
                ... 
                (1, i),
                ...
                (j, i)
            ]
        y : 
            a list with p_{ij} ( len(y) == n_instance * n_model ), like:
            [
                p00,
                p01,
                p02,
                ...,
                p0i,
                p10,
                p11,
                p12,
                ...,
                p1i,
                ...,
                pij
            ]
        """
        X, y = [], []
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                X.append((row,col))
                y.append(data.iloc[row,col])
        return X, y

    def get_params(self):
        """Get parameters

        Parameters
        -------------------------------------------------------
        self : self objetc

        Returns
        -------------------------------------------------------
        tuple of (abilities, difficulties, discrimination) or (abilities, difficulties),
        if self.fixed_discrimination == True.
        """
        if self.fixed_discrimination == False:
            return self.abilities, self.difficulties, self.discriminations
        else:
            return self.abilities, self.difficulties
    
    def fit(self, X, y=None):
        """Compute BIRT Gradient Descent
        
        Parameters
        -------------------------------------------------------
        X : list, array or tensor of shape (n_instances*n_models)

            X = [
                (0_instance, 0_model), 
                (0_instance, 1_model), 
                (0_instance, 2_model), 
                ...
                (0_instance, n_model),
                (1_instance, 0_model),
                (1_instance, 1_model),
                ...
                (n_instance, n_model)
                ]

            Training datasets to BIRTSGD model.
        
        y : list, array or tensor of shape (n_instances*n_models,). default: y==None
            if y == None, use self._transform tu split the data in X and y

        Returns
        -------------------------------------------------------
        self
            Adjusted values ​​of the parameters
        """
        if y == None:
            X, y = self._transform(X)

        args = [
            (
                X, y, self.n_models, 
                self.n_instances,
                self.epochs, self.batch_size, 
                self.lr, seed, self.fixed_discrimination,None
            ) for seed in range(
                self.n_seed, self.n_seed + self.n_inits
            )
        ]

        if self.n_workers == -1:
            n_workers = cpu_count()

        if self.n_workers == 1:
            map_f = map
        else:
            if n_workers > len(args):
                n_workers = len(args)

            p = Pool(n_workers)
            map_f = p.map

        results = map_f(_fit, args)

        abi, dif, dis_b, dis_a = [], [], [], []
        for res in results:
            abi.append(res[0])
            dif.append(res[1])
            if self.fixed_discrimination == False:
                dis_b.append(res[2])
                dis_a.append(res[3])

        abi = tf.math.reduce_mean(tf.stack(abi, axis=1), axis=1)
        dif = tf.math.reduce_mean(tf.stack(dif, axis=1), axis=1)
        if self.fixed_discrimination == False:
            dis_b = tf.math.reduce_mean(tf.stack(dis_b, axis=1), axis=1)
            dis_a = tf.math.reduce_mean(tf.stack(dis_a, axis=1), axis=1)
        
        if self.fixed_discrimination == False:
            if self.n_inits > 1:
                abi, dif, dis_b, dis_a = _fit((
                    X, y, self.n_models, 
                    self.n_instances,
                    self.epochs, self.batch_size, 
                    self.lr, self.n_seed + self.n_inits, self.fixed_discrimination, 
                    [abi, dif, dis_b, dis_a]
                ))
            self.abilities = tf.math.sigmoid(abi).numpy()
            self.difficulties = tf.math.sigmoid(dif).numpy()
            self.discriminations = tf.math.tanh(dis_b) * tf.math.softplus(dis_a)
            self.discriminations = self.discriminations.numpy()
        else:
            abi, dif = _fit((
                X, y, self.n_models, 
                self.n_instances,
                self.epochs, self.batch_size, 
                self.lr, self.n_seed + self.n_inits, self.fixed_discrimination,[abi, dif]
            ))
            self.abilities = tf.math.sigmoid(abi).numpy()
            self.difficulties = tf.math.sigmoid(dif).numpy()

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
        if self.fixed_discrimination == False:
            discriminations = self.discriminations 
        else:
            discriminations = [1]*len(self.difficulties)

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

def _loss(y_true, y_pred):
    """Calculate Binary Cross Entropy (loss function)

    Parameters
    -------------------------------------------------------
    y_true : tf.Tensor 
            It's a real value to y

    y_pred : tf.Tensor
            It's a real value to y predict

    Returns 
    -------------------------------------------------------
    EagerTensor with loss functions value
    """
    loss = tf.keras.losses.BinaryCrossentropy()
    return loss(y_true, y_pred)

def _irt(thi, delj, aj, bj):
    """calculating a probability (i) deduction from (j) using Beta3-IRT

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

    alphaij = (sig_thi/sig_delj)**(tanh_bj*soft_aj)
    betaij = ( ((1 - sig_thi)/(1 - sig_delj))**(tanh_bj*soft_aj) )
    y_est = alphaij/(alphaij + betaij)
    return y_est

def _irt_fixed_discrimination(thi, delj):
    """calculating a probability (i) deduction from (j) using Beta3-IRT

    Parameters
    -------------------------------------------------------
    thi : 
            Parameter to be estimated from the abilitie for Beta3-IRT

    delj : 
            Parameter to be estimated from the difficulty for Beta3-IRT

    Returns
    -------------------------------------------------------
    Calculate E[pij | abilities, difficulties, discrimination = 1]
    """
    sig_thi, sig_delj = tf.math.sigmoid(thi), tf.math.sigmoid(delj)

    alphaij = (sig_thi/sig_delj)**(1)
    betaij = ( ((1 - sig_thi)/(1 - sig_delj))**(1) )
    y_est = alphaij/(alphaij + betaij)
    return y_est

def _fit(args):
    """Train BIRT Gradient Descent

    Parameters
    -------------------------------------------------------
    batches : batchs containing X and Y

    Returns
    -------------------------------------------------------
    self
    """

    (X, y, n_models, n_instances, epochs, batch_size, lr, random_seed, fixed_discrimination, params) = args

    np.random.seed(random_seed)
    tf.random.set_seed(
        random_seed
    )

    if not params:

        thi = tf.Variable(
            np.random.normal(0,1, size=n_models), 
            trainable=True, dtype=tf.float32
        )
        
        delj = tf.Variable(
            np.random.normal(0,1, size=n_instances), 
            trainable=True, dtype=tf.float32
        )
        if fixed_discrimination == False:
            bj = tf.Variable(
                np.abs(np.random.normal(1, 1, size=n_instances)), 
                trainable=True, dtype=tf.float32
            )

            aj = tf.Variable(
                np.random.normal(1, 1, size=n_instances), 
                trainable=True, dtype=tf.float32
            )
    
    else:
        thi = tf.Variable(
            params[0], 
            trainable=True, dtype=tf.float32
        )
        
        delj = tf.Variable(
            params[1], 
            trainable=True, dtype=tf.float32
        )
        if fixed_discrimination == False:
            bj = tf.Variable(
                params[2], 
                trainable=True, dtype=tf.float32
            )

            aj = tf.Variable(
                params[3], 
                trainable=True, dtype=tf.float32
            )

    dataset = tf.data.Dataset.from_tensor_slices(
        np.hstack((X, np.array(y).reshape(-1, 1)))
    ).shuffle(len(X), reshuffle_each_iteration=True)

    if fixed_discrimination == False:
        variables = [
                thi,
                delj,
                aj,
                bj
            ]
        
        for _ in tqdm(range(epochs)):
            batches = dataset.batch(batch_size).as_numpy_iterator()
            for batch in batches:

                with tf.GradientTape() as g:
                    g.watch(variables)  # Precisa dizer para a tape observar as variaveis
                    t = tf.gather(thi, batch[:,1].astype(int))
                    d = tf.gather(delj, batch[:,0].astype(int))
                    a = tf.gather(aj, batch[:,0].astype(int))
                    b = tf.gather(bj, batch[:,0].astype(int))
                    y_pred = _irt(thi = t, delj = d, aj = a, bj = b)

                    current_loss = _loss(
                        batch[:, 2].astype(np.float32), y_pred
                    )
                _thi, _delj, _aj, _bj = g.gradient(current_loss, variables)
                
                thi.scatter_sub(tf.math.scalar_mul(lr, _thi))
                delj.scatter_sub(tf.math.scalar_mul(lr, _delj))
                aj.scatter_sub(tf.math.scalar_mul(lr, _aj))
                bj.scatter_sub(tf.math.scalar_mul(lr, _bj))

        # abilities, difficulties = tf.math.sigmoid(thi).numpy(), tf.math.sigmoid(delj).numpy()
        # discriminations = tf.math.tanh(bj).numpy()*tf.math.softplus(aj).numpy()
        
        parameters = thi, delj, bj, aj
    else:
        variables = [
                thi,
                delj
            ]
        
        for _ in tqdm(range(epochs)):
            batches = dataset.batch(batch_size).as_numpy_iterator()
            for batch in batches:

                with tf.GradientTape() as g:
                    g.watch(variables)  # Precisa dizer para a tape observar as variaveis
                    t = tf.gather(thi, batch[:,1].astype(int))
                    d = tf.gather(delj, batch[:,0].astype(int))
                    y_pred = _irt_fixed_discrimination(thi = t, delj = d)

                    current_loss = _loss(
                        batch[:, 2].astype(np.float32), y_pred
                    )
                _thi, _delj = g.gradient(current_loss, variables)
                
                thi.scatter_sub(tf.math.scalar_mul(lr, _thi))
                delj.scatter_sub(tf.math.scalar_mul(lr, _delj))

        # abilities, difficulties = tf.math.sigmoid(thi).numpy(), tf.math.sigmoid(delj).numpy()
        # discriminations = tf.math.tanh(bj).numpy()*tf.math.softplus(aj).numpy()
        
        parameters = thi, delj

    return parameters
    
        
    
   
