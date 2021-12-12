[![license: MIT](https://img.shields.io/badge/license-MIT-red.svg?&logo=license&color=blue)](https://github.com/Manuelfjr/birt-gd/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-birtgd-blue?&logo)](https://github.com/Manuelfjr/birt-sgd)
[![Author](https://img.shields.io/badge/author-manuelfjr-blue?&logo=github)](https://github.com/Manuelfjr)
[![Author2](https://img.shields.io/badge/author-tmfilho-blue?&logo=github)](https://github.com/tmfilho)

<!-- PyPi Status
![PyPI - Status](https://img.shields.io/pypi/status/pandas)
-->

<!--
![Py Coverage](https://s3.amazonaws.com/assets.coveralls.io/badges/coveralls_94.png)
-->

<!-- PyPi Downloads
[![PyPi - Downloads](https://pypip.in/d/pandas/badge.png?&color=blue&logo=python)](https://pypi.org/project/pandas/#files)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/scikit-learn?style=flat)](https://pypi.org/project/pandas/#files)
-->

<!-- Latest PyPI version
[![Latest PyPI version](https://img.shields.io/pypi/v/pandas?logo=pypi)](https://pypi.python.org/pypi/pandas)
-->

<!-- Release
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/pandas-dev/pandas)](https://GitHub.com/pandas-dev/pandas/releases/)

[![GitHub release](https://img.shields.io/github/release/Manuelfjr/birt-sgd.svg)](https://GitHub.com/Manuelfjr/birt-sgd/releases/)
-->

<!-- Static download of pepy
[![Downloads](https://static.pepy.tech/personalized-badge/pandas?period=total&units=international_system&left_color=grey&right_color=red&left_text=downloads)](https://pepy.tech/project/pandas)
-->

<!-- Github downloads
[![Github All Releases](https://img.shields.io/github/downloads/pandas-dev/pandas/total.svg?&logo=github&color=blue)]()
-->

<!-- Lines of code
![Lines of code](https://img.shields.io/tokei/lines/github/Manuelfjr/birt-sgd)
-->

<!-- Code size
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/pandas-dev/pandas)
-->

<!-- Github contributors
![GitHub contributors](https://img.shields.io/github/contributors/pandas-dev/pandas)
-->

<!--
[![Downloads](https://pepy.tech/badge/pandas)](https://pepy.tech/project/pandas)    
-->

# [birt-gd](https://pypi.org/project/birt-gd/)

**BIRTGD** is an implementation of Beta3-irt using gradient descent.

The model expects to receive two sets of data, *X* being a list or array containing tuples of indices, where the first index references the instance *j* and the second index of the tuple references the model *i*, thus, *Y* will be a list or array where each input will be p<sub>ij</sub> ~ &Beta;(&alpha;<sub>ij</sub>, &beta;<sub>ij</sub>), the probability of the *i* model correctly classifying the *j* model. Being, 

p<sub>ij</sub> ~ &Beta;(&alpha;<sub>ij</sub>, &beta;<sub>ij</sub>)

&alpha;<sub>ij</sub> = F<sub>&alpha;</sub>(&theta;<sub>i</sub>, &delta;<sub>j</sub>, a<sub>j</sub>) = (&theta;<sub>i</sub>/&delta;<sub>j</sub>)<sup>a<sub>j</sub></sup>

&beta;<sub>ij</sub> = F<sub>&beta;</sub>(&theta;<sub>i</sub>, &delta;<sub>j</sub>, a<sub>j</sub>) = ( (1 - &theta;<sub>i</sub>)/(1 - &delta;<sub>j</sub>) )<sup>a<sub>j</sub></sup>

&theta;<sub>i</sub> ~ &Beta;(1,1), &delta;<sub>j</sub> ~ &Beta;(1,1), a<sub>j</sub> ~ N(1, &sigma;<sup>2</sup><sub>0</sub>)

where,

E[p<sub>ij</sub> | &theta;<sub>i</sub>, &delta;<sub>j</sub>, a<sub>j</sub>] = (&alpha;<sub>ij</sub>)/( &alpha;<sub>ij</sub> + &beta;<sub>ij</sub>) = 1/(1 + ( (&delta;<sub>j</sub>)/(1 - &delta;<sub>j</sub>) )<sup>a<sub>j</sub></sup> &#xd7; ( (&theta;<sub>i</sub>)/(1 - &theta;<sub>i</sub>) )<sup> - a<sub>j</sub></sup> )

# Installation
## Dependencies 
birt-sgd requires:
- Python (>=3.8.5)
- numpy (>=1.19.5)
- tqdm (>=4.59.0)
- tensorflow (>=2.4.1)
- pandas (>=1.2.3)
- seaborn (>=0.11.0)
- matplotlib (>=3.3.2)
- scikit-learn (>=0.23.2)

## User installation

```bash
pip install birt-gd
```

## Source code 
You can check the code with 
```bash
git clone https://github.com/Manuelfjr/birt-gd
```

# Usage
Import the **BIRTGD's class**

```py
>>> from birt import BIRTGD
```

```py
>>> data = pd.DataFrame({'a': [0.99,0.89,0.87], 'b': [0.32,0.25,0.45]})
```

```py
>>> bgd = BIRTGD(n_models=2, n_instances=3, random_seed=1)
>>> bgd.fit(data)
100%|██████████| 5000/5000 [00:22<00:00, 219.50it/s]
<birt.BIRTGD at 0x7f6131326c10>
```

```py 
>>> bgd.abilities
array([0.90438306, 0.27729774], dtype=float32)
```

```py
>>> bgd.difficulties
array([0.3760659 , 0.5364428 , 0.34256178], dtype=float32)
```

```py
>>> bgd.discriminations
array([1.6690203 , 0.9951777 , 0.65577406], dtype=float32)
```

# Summary data

How to use the summary feature:

* **Generate data**
```py
import numpy as np
import pandas as pd
from birt import BIRTGD
import matplotlib.pyplot as plt

m, n = 5, 20
np.random.seed(1)
abilities = [np.random.beta(1,i) for i in ([0.1, 10] + [1]*(m-2))]
difficulties = [np.random.beta(1,i) for i in [10, 5] + [1]*(n-2)]
discrimination = list(np.random.normal(1,1, size=n))
pij = pd.DataFrame(columns=range(m), index=range(n))

i,j = 0,0
for theta in abilities:
  for delta, a in zip(difficulties, discrimination):
    alphaij = (theta/delta)**(a)
    betaij = ((1-theta)/(1 - delta))**(a)
    pij.loc[j,i] = np.random.beta(alphaij, betaij, size=1)[0]
    j+=1
  j = 0
  i+=1
```

* **Fitting the model**
```py
birt = BIRTGD(n_models=pij.shape[1],
             n_instances=pij.shape[0],
             learning_rate=1,
             epochs=5000,
             n_inits=1000)
birt.fit(pij)
```


* **Score (Pseudo-$R^2$)**
```py
birt.score
```
```py
0.9038145665424927
```


* **Summary**
```py
birt.summary()
```
```py

        ESTIMATES
        -----
                        | Min      1Qt      Median   3Qt      Max      Std.Dev
        Ability         | 0.00010  0.22148  0.63389  0.73353  0.92040  0.33960
        Difficulty      | 0.01745  0.28047  0.63058  0.84190  0.98624  0.31635
        Discrimination  | 0.31464  1.28330  1.61493  2.22936  4.44645  1.02678
        pij             | 0.00000  0.02219  0.35941  0.86255  0.99993  0.40210
        -----
        Pseudo-R2       | 0.90381
        

```

# Using Scatterplot Feature

```py
birt.plot(xaxis='discrimination',yaxis='difficulty', ann=True, kwargs={'color': 'red'})
plt.show()
```

<img alt = "assets/dis_diff_ex.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/dis_diff_ex.png">

```py
birt.plot(xaxis='difficulty',yaxis='average_item', ann=True, kwargs={'color': 'red'})
plt.show()
```

<img alt = "assets/diff_av_ex2.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/diff_av_ex2.png">


```py
birt.plot(xaxis='ability',yaxis='average_response', ann=False)
plt.show()
```

<img alt = "assets/ab_av_ex3.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ab_av_ex3.png">

# Using Boxplot Feature

```py
birt.boxplot(y='abilities',kwargs={'linewidth': 4})
```

<img alt = "assets/ab_av_ex4.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ex4.png">


```py
birt.boxplot(x='difficulties')
```
<img alt = "assets/ab_av_ex5.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ex5.png">

```py
birt.boxplot(y='discriminations')
```

<img alt = "assets/ab_av_ex6.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ex6.png">


# Help and Support
## Communication

- E-mail: [ferreira.jr.ufpb@gmail.com]()
- Site: [https://manuelfjr.github.io/](https://manuelfjr.github.io/)

# License
[MIT License](https://github.com/Manuelfjr/birt-sgd/blob/main/LICENSE)

Copyright (c) 2021 Manuel Ferreira Junior

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
