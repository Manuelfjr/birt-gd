[![license: MIT](https://img.shields.io/badge/license-MIT-red.svg?&logo=license&color=blue)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-birtsgd-blue?&logo)](https://github.com/Manuelfjr/birt-sgd)
[![Author](https://img.shields.io/badge/author-manuelfjr-blue?&logo=github)](https://github.com/Manuelfjr)
[![Author2](https://img.shields.io/badge/author-tmfilho-blue?&logo=github)](https://github.com/tmfilho)

<!-- PyPi Status
![PyPI - Status](https://img.shields.io/pypi/status/pandas)
-->

<!-- PyPi Downloads
[![PyPi - Downloads](https://pypip.in/d/pandas/badge.png?&color=blue&logo=python)](https://pypi.org/project/pandas/#files)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/scikit-learn?style=flat)](https://pypi.org/project/pandas/#files)
-->

<!-- Latest PyPI version
[![Latest PyPI version](https://img.shields.io/pypi/v/birt-sgd?logo=pypi)](https://pypi.python.org/pypi/birt-sgd)
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

# birt-sgd
**BIRTSGD** is an implementation of Beta3-irt using gradient descent.

The model expects to receive two sets of data, X being a list or array containing tuples of indices, where the first index references the instance __j__ and the second index of the tuple references the model __i__, thus, Y will be a list or array where each input will be $p_{ij}$ ~ $\Beta(\alpha_{ij}, \beta_{ij})$, the probability of the __i__ model correctly classifying the __j__ model, being:
<!--
$$
p_{ij} \sim Beta(\alpha_{ij}, \beta_{ij}),
$$
$$
\alpha_{ij} = F_{\alpha}(\theta_{i}, \delta_{j}, a_{j}) = \bigg(\frac{\theta_{i}}{\delta_{j}}\bigg)^{a_{j}},
$$
$$
\beta_{ij} = F_{\beta}(\theta_{i}, \delta_{j}, a_{j}) = \bigg(\frac{1 - \theta_{i}}{1 - \delta_{j}}\bigg)^{a_{j}},
$$
$$
\theta_{i} \sim B(1,1), \delta_{j} \sim B(1,1), a_{j} \sim N(1, \sigma^{2}_{0})
$$
where,
$$
E[p_{ij} | \theta_i,\delta_j,a_j] = \frac{\alpha_{ij}}{\alpha_{ij} + \beta_{ij}} = \frac{1}{1 - \big(\frac{\delta_{j}}{1 - \delta_{j}}\big)^{a_{j}}\cdot \big(\frac{\theta_{i}}{1 - \theta_{i}}\big)^{ - a_{j}} }
$$
-->
# Installation
## Dependencies 
birt-sgd requires:
- Python (>=3.8.5)
- numpy (>=1.19.5)
- tqdm (>=4.59.0)
- tensorflow (>=2.4.1)
- pandas (>=1.2.3)

## User installation

```bash
pip install birt-sgd
```

## Source code 
You can check the code with 
```bash
git clone https://github.com/Manuelfjr/birt-sgd
```

# Usage
Import the **BIRTSGD's class**

```py
>>> from birt import BIRTSGD
```

```py
>>> X = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
>>> Y = [0.98,0.81,0.12,0.567,0.76,0.9]
```

```py
>>> bsgd = BIRTSGD(n_models=3, n_instances=2, random_seed=1)
>>> bsgd.fit(X,Y)
100%|██████████| 20/20 [00:00<00:00, 52.81it/s]
<birt.BIRTSGD at 0x7f6ce2555f50>
```

```py 
>>> bsgd._thi
array([0.78665066, 0.50258964, 0.545207  ], dtype=float32)
```

```py
>>> bsgd._delj
array([0.25070453, 0.46883532], dtype=float32)
```

```py
>>> bsgd._aj
array([0.25051177, 2.3821855 ], dtype=float32)
```

```py
>>> bsgd._bj
array([0.37420523, 0.59285855], dtype=float32)  
```

# Help and Support
## Communication

- E-mail: [ferreira.jr.ufpb@gmail.com]()
- Site: [https://manuelfjr.github.io/](https://manuelfjr.github.io/)