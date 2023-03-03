from multiprocessing import Process, Queue, Value

import tensorflow as tf
#import tensorflow_probability as tfp
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning

import seaborn as sns
import pandas as pd
import numpy as np

class _viz:
    def __init__(self, abi, dif, dis):
        self.abilities = abi
        self.difficulties = dif
        self.discriminations = dis

    def _split(self, *args):
        (pij, predict, shape) = args
        self.pij = pij
        self.pijhat = pd.DataFrame( np.reshape(predict, (shape[0], shape[1])) )

        shape = np.shape(self.pij)
        try:
            Y, X = np.reshape(self.pij.values,(shape[0]*shape[1], 1)), np.reshape(self.pijhat.values,(shape[0]*shape[1], 1))
        except:
            Y, X = np.reshape(self.pij,(shape[0]*shape[1], 1)), np.reshape(self.pijhat.values,(shape[0]*shape[1], 1))
        return Y, X, self.pijhat

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
    
    def irt(self, abilities, difficulties, discriminations):
        """calculating a probability (i) deduction from (j)
        Parameters
        -------------------------------------------------------
        abilities : 
                Optimized ability estimation by gradient descent
        difficulties : 
                Optimized difficulties estimation by gradient descent
        discriminations : 
                Optimized discriminations estimation by gradient descent
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
        y_pred = []
        for d,a in zip(self.difficulties, self.discriminations):
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

    def plot(self, xaxis=None, yaxis=None, kwargs={}, ann=False, font_size=12, font_ann_size=12):
        """Plot's of results
            
        Parameters
        -------------------------------------------------------
        xaxis : {'discrimination','difficulty','ability'}.
                x axis of plot.
        yaxis : {'discrimination','difficulty', 'average_response', 'average_item'}
                y axis of plot.
        kwargs: dict.
            other Matplotlib library arguments
        ann   : boolean.
            data points with annotations
        font_size : int.
            font size of x and y labels
        font_ann_size : int.
            font size of ann points
        Returns
        -------------------------------------------------------
            Scatterplot of xaxis vs yaxis.
        """
        pij = pd.DataFrame(np.reshape(self.predict(), (len(self.discriminations), len(self.abilities))))

        sns.set_style('darkgrid')
        plt.figure(figsize=(13,6))
        if (xaxis == 'discrimination') and (yaxis == 'difficulty'):
            sns.scatterplot(x=self.discriminations, y=self.difficulties, **kwargs)
            if ann == True:
                for i in range(pij.shape[0]):
                    plt.text(x = self.discriminations[i]+0.007, y = self.difficulties[i]+0.007, s='n{}'.format(i+1), fontsize=font_ann_size)
        elif (yaxis == 'discrimination') and (xaxis == 'difficulty'):
            sns.scatterplot(x=self.difficulties, y=self.discriminations, **kwargs)
            if ann == True:
                sns.scatterplot(x=self.difficulties, y=self.discriminations, **kwargs)
                for i in range(pij.shape[0]):
                    plt.text(x = self.difficulties[i]+0.007, y = self.discriminations[i]+0.007, s='n{}'.format(i+1), fontsize=font_ann_size)
        elif (xaxis == 'ability') and (yaxis == 'average_response'):
            sns.scatterplot(x=self.abilities, y=pij.apply(np.mean,axis=0), **kwargs)
            if ann == True:
                sns.scatterplot(x=self.abilities, y=pij.apply(np.mean,axis=0), **kwargs)
                for i in range(pij.shape[1]):
                    plt.text(x = self.abilities[i]+0.003, y =pij.apply(np.mean,axis=0)[i]+0.003, s='{}'.format(self.pij.columns[i]), fontsize=font_ann_size)
        elif (xaxis == 'difficulty') and (yaxis == 'average_item'):
            sns.scatterplot(x=self.difficulties, y=pij.apply(np.mean,axis=1), **kwargs)
            if ann == True:
                for i in range(pij.shape[0]):
                    plt.text(x = self.difficulties[i]+0.004, y =pij.apply(np.mean,axis=1)[i]+0.004, s='n{}'.format(i+1), fontsize=font_ann_size)
        elif (xaxis == 'discrimination') and (yaxis == 'average_item'):
            sns.scatterplot(x=self.discriminations, y=pij.apply(np.mean,axis=1), **kwargs)
            if ann == True:
                for i in range(pij.shape[0]):
                    plt.text(x = self.discriminations[i]+0.004, y =pij.apply(np.mean,axis=1)[i]+0.004, s='n{}'.format(i+1), fontsize=font_ann_size)
        elif xaxis == yaxis:
            raise ValueError('xaxis and yaxis are the same')
        elif xaxis not in ['discrimination','difficulty','ability'] or yaxis not in ['discrimination','difficulty', 'average_response', 'average_item']:
            raise ValueError(f'{xaxis} or {yaxis} doesnt exists')
        else:
            raise ValueError(f'plotting {xaxis} vs {yaxis} is impossible.')
        plt.title(f'{xaxis} vs {yaxis}', fontsize=font_size)
        plt.xlabel(xaxis, fontsize=font_size)
        plt.ylabel(yaxis, fontsize=font_size)
    
    def boxplot(self, x=None, y=None, kwargs={}, font_size=12):
        """Boxplot's of results
            
        Parameters
        -------------------------------------------------------
        x : ['discrimination','difficulty','ability', None].
                x axis of plot.
        y : ['discrimination','difficulty','ability', None]
                y axis of plot.
        kwargs: dict.
            other Matplotlib library arguments.
        font_size : int.
            font size of x and y labels
        Returns
        -------------------------------------------------------
            Boxplot of x axis vs y axis.
        """
        abi = pd.DataFrame({'ability': self.abilities})
        dif_dis = pd.DataFrame({'difficulty': self.difficulties, 'discrimination': self.discriminations})
        sns.set_style('darkgrid')
        plt.figure(figsize=(13,6))
        if x == 'abilities':
            if y == None:
                bx = sns.boxplot(x=x,y=y, data=abi, **kwargs)
            elif y == 'discrimination' or y == 'difficulty':
                raise ValueError(f'the length of x is different from y.')
            elif y == 'ability':
                raise ValueError(f'both X and Y are the same')
        elif x == 'difficulty':
            if y == None or y == 'discrimination':
                bx = sns.boxplot(x=x,y=y, data=dif_dis, **kwargs)
            elif y == 'difficulty':
                raise ValueError(f'both X and Y are the same')
            elif y == 'ability':
                raise ValueError(f'the length of x is different from y. ({dif_dis.shape[0], abi.shape[0]})')
        elif x == 'discrimination':
            if y == None or y == 'difficulty':
                bx = sns.boxplot(x=x,y=y, data=dif_dis, **kwargs)
            elif y == 'discrimination':
                raise ValueError(f'both X and Y are the same')
            elif y == 'ability':
                raise ValueError(f'the length of x is different from y. ({dif_dis.shape[0], abi.shape[0]})')
        elif x == None:
            if y == 'ability':
                bx = sns.boxplot(x=x,y=y, data=abi, **kwargs)
            elif y == 'difficulty' or y == 'discrimination':
                bx = sns.boxplot(x=x,y=y, data=dif_dis, **kwargs)
            elif y == 'None':
                raise ValueError(f'both X and Y are None')
        elif y == None and (x not in ['ability', 'difficulty', 'discrimination']):
            raise ValueError(f'y is None but {x} doesnot exist')
        elif x == None and (y not in ['ability', 'difficulty', 'discrimination']):
            raise ValueError(f'x is None but {y} doesnot exist')
        elif x != None and y != None:
            if (x == 'ability' and y == 'difficulty') or (x == 'difficulty' and y == 'ability'):
                raise ValueError(f'the length of x is different from y.')
        bx.set_ylabel(y,fontsize=font_size)
        bx.set_xlabel(x,fontsize=font_size)

class _irt(object):
    def irt_four(self, thi, delj, aj, bj, set_priors=True):
        """Calculating a probability (i) deduction from (j) using Beta4-IRT
        Parameters
        -------------------------------------------------------
        thi : 
                Parameter to be estimated from the abilitie for Beta4-IRT
        delj : 
                Parameter to be estimated from the difficulty for Beta4-IRT
        aj : 
                Other parameter of discrimination to be estimated to Beta4-IRT.
        bj : 
                Other parameter of discrimination to be estimated to Beta4-IRT.
        Returns
        -------------------------------------------------------
        Calculate E[pij | abilities, difficulties, discrimination = aj * bj]
        """
        sig_thi, sig_delj = tf.math.sigmoid(thi), tf.math.sigmoid(delj)
        soft_aj = tf.math.softplus(aj)

        if set_priors==True:
            tanh_bj = bj
        else:
            tanh_bj = tf.math.tanh(bj)

        term1 = sig_delj / (1 - sig_delj)
        term2 = (1 - sig_thi) / sig_thi
        mult = tf.tensordot(term1, term2, axes=1)

        est = 1 / (1 + mult ** (tanh_bj * soft_aj))
        return est

    def fit_four(self, *args):
        """Train BIRT with Gradient Descent
        Parameters
        -------------------------------------------------------
        args : tuple containing 
                (queue, X, n_respondents, n_items, epochs, lr, random_seed, n_inits, tol)
        Returns
        -------------------------------------------------------
        self
        """
        (queue, X, n_respondents, n_items, epochs, lr, random_seed, n_inits, tol) = args

        np.random.seed(random_seed)
        tf.random.set_seed(
            random_seed
        )

        X = tf.Variable(
            X, 
            trainable=False, dtype=tf.float32
        )

        thi = tf.Variable(
            np.random.normal(0,1, size=(1, n_respondents)), 
            trainable=True, dtype=tf.float32
        )
        
        delj = tf.Variable(
            np.random.normal(0,1, size=(n_items, 1)), 
            trainable=True, dtype=tf.float32
        )
        
        bj = tf.Variable(
            np.ones((n_items, 1)) * 0.5096851, 
            trainable=True, dtype=tf.float32
        )

        aj = tf.Variable(   
            np.ones((n_items, 1)) * 2.002374, 
            trainable=True, dtype=tf.float32
        )
        
        t = 0
        self.current_loss = 0
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
                pred = self.irt_four(thi = thi, delj = delj, aj = aj, bj = bj, set_priors=False)
                old_loss = self.current_loss
                self.current_loss = _loss(
                    X, pred
                )
                
            if np.abs(old_loss - self.current_loss) < tol:
                print(f'Model converged at the {t}th epoch')
                break

            gradients = g.gradient(self.current_loss, variables)
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

    def fit_priors(self, *args):
        """Train BIRT with Gradient Descent
        Parameters
        -------------------------------------------------------
        args : tuple containing 
                (queue, X, n_respondents, n_items, epochs, lr, random_seed, n_inits, tol)
        Returns
        -------------------------------------------------------
        self
        """
        (queue, X, n_respondents, n_items, epochs, lr, random_seed, n_inits, tol) = args

        np.random.seed(random_seed)
        tf.random.set_seed(
            random_seed
        )
        
        abil = np.mean(X, axis=0)
        thi = tf.Variable(
            np.log(-abil / (abil - 1)).reshape((1, n_respondents)),
            # np.random.normal(0,1, size=(1, n_respondents)), 
            trainable=True, dtype=tf.float32
        )

        diff = 1 - np.mean(X, axis=1)        
        delj = tf.Variable(
            np.log(-diff / (diff - 1)).reshape((n_items, 1)),
            # np.random.normal(0,1, size=(n_items, 1)), 
            trainable=True, dtype=tf.float32
        )
        
        cor = np.zeros((n_items, 1))
        for i in range(len(X)):
            p_i = self.pij[i]#P[i]
            cor[i, 0] = np.corrcoef(abil, p_i)[0,1]       

        cor[np.isnan(cor)] = 0

        bj = tf.Variable(
            np.log((-cor - 1) / (cor - 1)) / 2, 
            trainable=False, dtype=tf.float32
        )
        
        aj = tf.Variable(   
            # np.random.normal(0, 1, size=(n_items, 1)), clip
            np.log(np.exp(np.ones((n_items, 1))) - 1),
            trainable=True, dtype=tf.float32
        )

        X = tf.constant(
            X, dtype=tf.float32
        )
        
        t = 0
        self.current_loss = 0
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
                    aj
                ]
            
            with tf.GradientTape() as g:
                g.watch(variables)
                pred = self.irt_four(thi = thi, delj = delj, aj = aj, bj = bj)
                old_loss = self.current_loss
                self.current_loss = _loss(
                    X, pred
                )

            if np.abs(old_loss - self.current_loss) < tol:
                print(f'Model converged at the {t}th epoch')
                break

            gradients = g.gradient(self.current_loss, variables)
            if t < n_inits:
                _thi, _delj = gradients
            else:
                _thi, _delj, _aj = gradients
                aj.assign_sub(tf.math.scalar_mul(lr, _aj))
                # bj.assign_sub(tf.math.scalar_mul(lr, _bj))
            thi.assign_sub(tf.math.scalar_mul(lr, _thi))
            delj.assign_sub(tf.math.scalar_mul(lr, _delj))

            t += 1
        
        abilities = tf.math.sigmoid(thi).numpy().flatten()
        difficulties = tf.math.sigmoid(delj).numpy().flatten()
        discriminations = tf.math.tanh(bj) * tf.math.softplus(aj)
        discriminations = discriminations.numpy().flatten()

        parameters = (abilities, difficulties, discriminations)

        queue.put(parameters)

    def irt_three(self, thi, delj, aj):
        """Calculating a probability (i) deduction from (j) using Beta3-IRT
        Parameters
        -------------------------------------------------------
        thi : 
                Parameter to be estimated from the abilitie for Beta3-IRT
        delj : 
                Parameter to be estimated from the difficulty for Beta3-IRT
        aj : 
                Other parameter of discrimination to be estimated to Beta3-IRT.
        Returns
        -------------------------------------------------------
        Calculate E[pij | abilities, difficulties, discrimination = aj * bj]
        """
        sig_thi, sig_delj = tf.math.sigmoid(thi), tf.math.sigmoid(delj)
        param_aj = tf.convert_to_tensor(aj)

        term1 = sig_delj / (1 - sig_delj)
        term2 = (1 - sig_thi) / sig_thi
        mult = tf.tensordot(term1, term2, axes=1)

        est = 1 / (1 + mult ** (param_aj))
        return est

    def fit_three(self, *args):
        """Train with Gradient Descent
        Parameters
        -------------------------------------------------------
        args : tuple containing 
                (queue, X, n_respondents, n_items, epochs, lr, random_seed, n_inits, tol)
        Returns
        -------------------------------------------------------
        self
        """
        (queue, X, n_respondents, n_items, epochs, lr, random_seed, n_inits, tol) = args

        np.random.seed(random_seed)
        tf.random.set_seed(
            random_seed
        )

        X = tf.constant(
            X
        )

        thi = tf.Variable(
            np.random.normal(0,1, size=(1, n_respondents)), 
            trainable=True, dtype=tf.float32
        )
        
        delj = tf.Variable(
            np.random.normal(0,1, size=(n_items, 1)), 
            trainable=True, dtype=tf.float32
        )
        
        if n_inits == 0:
            aj = tf.Variable(   
                np.random.normal(1,1, size=(n_items, 1)), 
                trainable=True, dtype=tf.float32
            )
        else:
            aj = tf.Variable(   
                np.ones((n_items, 1)), 
                trainable=True, dtype=tf.float32
            )
        
        t = 0
        self.current_loss = 0 
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
                    aj
                ]
            
            with tf.GradientTape() as g:
                g.watch(variables)
                pred = self.irt_three(thi = thi, delj = delj, aj = aj)
                old_loss = self.current_loss
                self.current_loss = _loss(
                    X, pred
                )

            if np.abs(old_loss - self.current_loss) < tol:
                print(f'Model converged at the {t}th epoch')
                break

            gradients = g.gradient(self.current_loss, variables)
            if t < n_inits:
                _thi, _delj = gradients
            else:
                _thi, _delj, _aj = gradients
                aj.assign_sub(tf.math.scalar_mul(lr, _aj))

            thi.assign_sub(tf.math.scalar_mul(lr, _thi))
            delj.assign_sub(tf.math.scalar_mul(lr, _delj))

            t += 1
        
        abilities = tf.math.sigmoid(thi).numpy().flatten()
        difficulties = tf.math.sigmoid(delj).numpy().flatten()
        discriminations = tf.convert_to_tensor(aj).numpy().flatten()

        parameters = (abilities, difficulties, discriminations)

        queue.put(parameters)

    def irt(self, abilities, difficulties, discriminations):
        """calculating a probability (i) deduction from (j)
        Parameters
        -------------------------------------------------------
        abilities : 
                Optimized ability estimation by gradient descent
        difficulties : 
                Optimized difficulties estimation by gradient descent
        discriminations : 
                Optimized discriminations estimation by gradient descent
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
        y_pred = []
        for d,a in zip(self.difficulties, self.discriminations):
            for t in self.abilities:
                y_pred.append(
                    self.irt(
                        abilities = t, 
                        difficulties = d, 
                        discriminations = a
                    )
                )

        return y_pred

class Beta4(_viz,_irt):
    """Beta4-IRT with Gradient descent.
    Read more in the https://github.com/Manuelfjr/BIRTGD .
    Parameters:
    -------------------------------------------------------
    learning_rate : float, default=0.1
        It's a learning rate to Gradient descent.
    epochs : int, default=20
        Numbers of epochs to fit the model.
    n_respondents : int, default=20
        Numbers of models to be evaluated.
    n_items : int, default=100
        Numbers of instances to fit the models.
    random_seed : int, default=1
        Determines a random state to generation initial kicks.
        
    n_inits : int, default=10
        Determines the number of initializations.
        
    n_workers : int, default=-1
        Determines the number of CPUs to use.
    
    tol : float, default=10**(-5)
        Tolerance to converge epochs.
    Attributes
    -------------------------------------------------------
    self.abilities : ResourceVariable of shape (n_respondents,)
        It is the optimized estimate for the abilitie parameter.
    
    self.difficulties : ResourceVariable of shape (n_items,)
        It is the optimized estimate for the difficulties parameter.
    self.discriminations : ResourceVariable of shape (n_items,)
        It is the optimized estimate for the discrimination parameter.
    Notes
    -------------------------------------------------------
    Example
    -------------------------------------------------------
    >>> from birt import BIRTGD
    >>> data = pd.DataFrame({'a': [0.99,0.89,0.87], 'b': [0.32,0.25,0.45]})
    >>> bgd = BIRTGD(n_respondents=2, n_items=3, random_seed=1)
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
        epochs=10000, n_respondents=20, 
        n_items=100,
        n_inits=1000, n_workers=-1,
        random_seed=1,
        tol=10**(-5),
        set_priors=True
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.n_respondents = n_respondents
        self.n_items = n_items
        self.n_seed = random_seed
        self.n_inits = n_inits
        self.n_workers = n_workers
        self.tol = tol
        self.set_priors = set_priors
        self._params = {
            'learning_rate': learning_rate,
            'epochs': epochs,
            'n_respondents': n_respondents,
            'n_items': n_items
        }
    
    def fit(self, X, y=None):
        """Compute Beta^4-IRT with Gradient Descent
        
        Parameters
        -------------------------------------------------------
        X : list, array or tensor of shape (n_items,n_respondents)
            
        Returns
        -------------------------------------------------------
        self
            Adjusted values of the parameters
        """

        X = np.clip(X, (10)**(-4), 1 - (10)**(-4))
        self.pij = X
        queue = Queue()

        args = (
            queue, X, self.n_respondents, 
            self.n_items,
            self.epochs, self.lr, self.n_seed, 
            self.n_inits, self.tol
        )

        if self.set_priors:
            p = Process(target=super().fit_priors, args=list(args))
        else:
            p = Process(target=super().fit_four, args=list(args))
        p.start()
        abi, dif, dis = queue.get()
        p.join()
        
        super().__init__(abi, dif, dis)

        self.abilities = abi
        self.difficulties = dif
        self.discriminations = dis
        X, Y, self.pijhat = self._split(self.pij, super().predict(), (len(self.discriminations), len(self.abilities)))
        self.score(X, Y)
        return self


class Beta3(_viz,_irt):
    """Beta3-IRT with Gradient descent.
    Read more in the https://github.com/Manuelfjr/BIRTGD .
    Parameters:
    -------------------------------------------------------
    learning_rate : float, default=0.1
        It's a learning rate to Gradient descent.
    epochs : int, default=20
        Numbers of epochs to fit the model.
    n_respondents : int, default=20
        Numbers of models to be evaluated.
    n_items : int, default=100
        Numbers of instances to fit the models.
    random_seed : int, default=1
        Determines a random state to generation initial kicks.
        
    n_inits : int, default=10
        Determines the number of initializations.
        
    n_workers : int, default=-1
        Determines the number of CPUs to use.
    tol : float, default=10**(-5)
        Tolerance to converge epochs.
    
    Attributes
    -------------------------------------------------------
    self.abilities : ResourceVariable of shape (n_respondents,)
        It is the optimized estimate for the abilitie parameter.
    
    self.difficulties : ResourceVariable of shape (n_items,)
        It is the optimized estimate for the difficulties parameter.
    self.discriminations : ResourceVariable of shape (n_items,)
        It is the optimized estimate for the discrimination parameter.
    Notes
    -------------------------------------------------------
    Example
    -------------------------------------------------------
    >>> from birt import BTHREE
    >>> data = pd.DataFrame({'a': [0.99,0.89,0.87], 'b': [0.32,0.25,0.45]})
    >>> bgd = BTHREE(n_respondents=2, n_items=3, random_seed=1)
    >>> bgd.fit(data)
    100%|██████████| 5000/5000 [00:22<00:00, 219.50it/s]
    <birt.BTHREE at 0x7f6131326c10>
    >>> bsgd.abilities
    array([0.90438306, 0.27729774], dtype=float32)
    >>> bgd.difficulties
    array([0.3760659, 0.5364428, 0.34256178], dtype=float32)
    >>> bgd.discriminations
    array([1.6690203, 0.9951777, 0.65577406], dtype=float32)
    """
    def __init__(
        self, learning_rate=1, 
        epochs=10000, n_respondents=20, 
        n_items=100,
        n_inits=1000, n_workers=-1,
        random_seed=1, tol=10**(-5)
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.n_respondents = n_respondents
        self.n_items = n_items
        self.n_seed = random_seed
        self.n_inits = n_inits
        self.n_workers = n_workers
        self.tol = tol
        self._params = {
            'learning_rate': learning_rate,
            'epochs': epochs,
            'n_respondents': n_respondents,
            'n_items': n_items
        }
    
    def fit(self, X, y=None):
        """Compute BIRT Gradient Descent
        
        Parameters
        -------------------------------------------------------
        X : list, array or tensor of shape (n_items,n_respondents)
            
        Returns
        -------------------------------------------------------
        self
            Adjusted values of the parameters
        """
        self.pij = X
        queue = Queue()

        args = (
            queue, X, self.n_respondents, 
            self.n_items,
            self.epochs, self.lr, self.n_seed, 
            self.n_inits, self.tol
        )

        p = Process(target=super().fit_three, args=list(args))
        p.start()
        abi, dif, dis = queue.get()
        p.join()

        super().__init__(abi, dif, dis)

        self.abilities = abi
        self.difficulties = dif
        self.discriminations = dis
        X, Y, self.pijhat = self._split(self.pij, self.predict(), (len(self.discriminations), len(self.abilities)))
        self.score(X, Y)
        return self
 
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