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

    n_batchs : int, default=5
        Batch numbers for each iteration of the model fit. (n_batchs <= n_instances)

    random_seed : int, default=1
        Determines a random state to generation initial kicks.

    
    Attributes
    -------------------------------------------------------
    self._thi : ResourceVariable of shape (n_models,)
        It's a initial kick to ability parameter .
    
    self._delj : ResourceVariable of shape (n_instances,)
        It's a initial kick to difficulties parameter.

    self._aj : ResourceVariable of shape (n_instances,)
        It's a initial kick to other parameter of discrimination (Discrimination is: self._aj*self._bj).
    
    self._bj : ResourceVariable of shape (n_instances,)
        It's a initial kick to other parameter of discrimination (Discrimination is: self._aj*self._bj).
    
    self._abilities : ResourceVariable of shape (n_models,)
        It is the optimized estimate for the abilitie parameter.
    
    self._difficulties : ResourceVariable of shape (n_instances,)
        It is the optimized estimate for the difficulties parameter.

    self._discrimination : ResourceVariable of shape (n_instances,)
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
    >>> bsgd._abilities
    array([0.78665066, 0.5025896 , 0.545207  ], dtype=float32)
    >>> bsgd._difficulties
    array([0.25070453, 0.46883535], dtype=float32)
    >>> bsgd._discrimination
    array([0.09374281, 1.4122988 ], dtype=float32)
    """
    def __init__(self, learning_rate=0.1, epochs=20, n_models=20, n_instances=100, n_batchs=5, random_seed=1):
        self.lr = learning_rate
        self.epoch = epochs
        self.n_models = n_models
        self.n_instances = n_instances
        self.n_batchs = n_batchs
        self.n_seed = random_seed
        self._params = {
            'learning_rate': learning_rate,
            'epochs': epochs,
            'n_models': n_models,
            'n_instances': n_instances,
            'n_batchs':  n_batchs
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
        tuple of (abilities, difficulties, discrimination)
        """
        return self._abilities, self._difficulties, self._discrimination
    
    def _loss(self, y_true, y_pred):
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
    
    def _irt(self, thi, delj, aj, bj):
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
    
    def irt(self, abilities, difficulties, discrimination):
        """calculating a probability (i) deduction from (j) using Beta3-IRT

        Parameters
        -------------------------------------------------------
        abilities : 
                Optimized ability estimation by gradient descent using Beta3-IRT

        difficulties : 
                Optimized difficulties estimation by gradient descent using Beta3-IRT

        discrimination : 
                Optimized discrimination estimation by gradient descent using Beta3-IRT

        Returns
        -------------------------------------------------------
        Calculate E[pij | abilities, difficulties, discrimination]
        """
        alphaij = (abilities/difficulties)**(discrimination)
        betaij = ( ((1 - abilities)/(1 - difficulties))**(discrimination) )
        return alphaij/(alphaij + betaij)
    
    def _train(self, batches):
        """Train BIRT Gradient Descent

        Parameters
        -------------------------------------------------------
        batches : batchs containing X and Y

        Returns
        -------------------------------------------------------
        self
        """
        variables = [
            self._thi,
            self._delj,
            self._aj,
            self._bj
        ]

        for batch in batches:

            with tf.GradientTape() as g:
                g.watch(variables)  # Precisa dizer para a tape observar as variaveis
                t = tf.gather(self._thi, batch[:,1].astype(int))
                d = tf.gather(self._delj, batch[:,0].astype(int))
                a = tf.gather(self._aj, batch[:,0].astype(int))
                b = tf.gather(self._bj, batch[:,0].astype(int))
                y_pred = self._irt(thi = t, delj = d, aj = a, bj = b)

                current_loss = self._loss(
                    batch[:, 2].astype(np.float32), y_pred
                )
            _thi, _delj, _aj, _bj = g.gradient(current_loss, variables)

            self._thi.scatter_sub(tf.math.scalar_mul(self.lr, _thi))
            self._delj.scatter_sub(tf.math.scalar_mul(self.lr, _delj))
            self._aj.scatter_sub(tf.math.scalar_mul(self.lr, _aj))
            self._bj.scatter_sub(tf.math.scalar_mul(self.lr, _bj))
        return self
    
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

        np.random.seed(self.n_seed)
        tf.random.set_seed(
            self.n_seed
        )

        self._thi = tf.Variable(
            np.random.normal(0,1, size=self.n_models), 
            trainable=True, dtype=tf.float32
        )
        
        self._delj = tf.Variable(
            np.random.normal(0,1, size=self.n_instances), 
            trainable=True, dtype=tf.float32
        )

        self._aj = tf.Variable(
            np.random.normal(1, 1, size=self.n_instances), 
            trainable=True, dtype=tf.float32
        )

        self._bj = tf.Variable(
            np.abs(np.random.normal(1, 1, size=self.n_instances)), 
            trainable=True, dtype=tf.float32
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            np.hstack((X, np.array(y).reshape(-1, 1)))
        ).shuffle(len(X), reshuffle_each_iteration=True)
    
        costs = []
        
        for _ in tqdm(range(self.epoch)):
            self._train(dataset.batch(self.n_batchs).as_numpy_iterator())
        
        self._abilities, self._difficulties = tf.math.sigmoid(self._thi).numpy(), tf.math.sigmoid(self._delj).numpy()
        self._discrimination = tf.math.tanh(self._bj).numpy()*tf.math.softplus(self._aj).numpy()
        
        return self

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

        y_pred = []
        for d,a in zip(self._difficulties, self._discrimination):
            for t in self._abilities:
                y_pred.append(
                    self.irt(
                        abilities = t, 
                        difficulties = d, 
                        discrimination = a
                    )
                )

        return y_pred
