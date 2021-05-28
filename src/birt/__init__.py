import os
from multiprocessing import Process, Queue

import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


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
    >>> from BIRTGD.birt import BIRTGD
    >>> X = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    >>> Y = [0.98,0.81,0.12,0.567,0.76,0.9]
    >>> bgd = BIRTGD(n_models=3, n_instances=2, random_seed=1)
    >>> bgd.fit(X,Y)
    100%|██████████| 20/20 [00:00<00:00, 52.58it/s]
    <birt.BIRTGD at 0x7f6ce2555f50>
    >>> bsgd.abilities
    array([0.78665066, 0.5025896 , 0.545207  ], dtype=float32)
    >>> bgd.difficulties
    array([0.25070453, 0.46883535], dtype=float32)
    >>> bgd.discriminations
    array([0.09374281, 1.4122988 ], dtype=float32)
    """
    def __init__(
        self, learning_rate=0.1, 
        epochs=20, n_models=20, 
        n_instances=100,
        n_inits=10, n_workers=-1,
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
            Adjusted values ​​of the parameters
        """
        
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

# def _irt_fixed_discrimination(thi, delj):
#     """calculating a probability (i) deduction from (j) using Beta3-IRT

#     Parameters
#     -------------------------------------------------------
#     thi : 
#             Parameter to be estimated from the abilitie for Beta3-IRT

#     delj : 
#             Parameter to be estimated from the difficulty for Beta3-IRT

#     Returns
#     -------------------------------------------------------
#     Calculate E[pij | abilities, difficulties, discrimination = 1]
#     """
#     sig_thi, sig_delj = tf.math.sigmoid(thi), tf.math.sigmoid(delj)

#     term1 = sig_delj / (1 - sig_delj)
#     term2 = (1 - sig_thi) / sig_thi
#     mult = tf.tensordot(term1, term2, axes=1)

#     est = 1 / (1 + mult)
#     return est

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
