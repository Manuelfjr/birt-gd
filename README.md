---
mathjax: true
---

[![license: MIT](https://img.shields.io/badge/license-MIT-red.svg?&logo=license&color=blue)](https://github.com/Manuelfjr/birt-sgd/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-birtsgd-blue?&logo)](https://github.com/Manuelfjr/birt-sgd)
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

# [birt-sgd](https://test.pypi.org/project/birt-sgd/)
**BIRTSGD** is an implementation of Beta3-irt using gradient descent.

The model expects to receive two sets of data, *X* being a list or array containing tuples of indices, where the first index references the instance *j* and the second index of the tuple references the model *i*, thus, *Y* will be a list or array where each input will be p<sub>ij</sub> ~ &Beta;(&alpha;<sub>ij</sub>, &beta;<sub>ij</sub>), the probability of the *i* model correctly classifying the *j* model. Being, 

p<sub>ij</sub> ~ &Beta;(&alpha;<sub>ij</sub>, &beta;<sub>ij</sub>)

&alpha;<sub>ij</sub> = F<sub>&alpha;</sub>(&theta;<sub>i</sub>, &delta;<sub>j</sub>, a<sub>j</sub>) = (&theta;<sub>ij</sub>/&delta;<sub>ij</sub>)<sup>a<sub>j</sub></sup>

&beta;<sub>ij</sub> = F<sub>&beta;</sub>(&theta;<sub>i</sub>, &delta;<sub>j</sub>, a<sub>j</sub>) = ( (1 - &theta;<sub>ij</sub>)/(1 - &delta;<sub>ij</sub>) )<sup>a<sub>j</sub></sup>

&theta;<sub>i</sub> ~ &Beta;(1,1), &delta;<sub>j</sub> ~ &Beta;(1,1), a<sub>j</sub> ~ N(1, &sigma;<sup>2</sup><sub>0</sub>)

where,

E[p<sub>ij</sub> | &theta;<sub>i</sub>, &delta;<sub>j</sub>, a<sub>j</sub>] = (&alpha;<sub>ij</sub>)/( &alpha;<sub>ij</sub> + &beta;<sub>ij</sub>) = 1/(1 + ( (&delta;<sub>ij</sub>)/(1 - &delta;<sub>ij</sub>) )<sup>a<sub>ij</sub></sup> &#xd7; ( (&theta;<sub>ij</sub>)/(1 - &theta;<sub>ij</sub>) )<sup> - a<sub>ij</sub></sup> )
<!--
Being:   

<img src="https://latex.codecogs.com/svg.latex?&space;p_{ij} \sim B(\alpha_{ij}, \beta_{ij}), " title="p_{ij} \sim B(\alpha_{ij}, \beta_{ij}), " /><br>

<img src="https://latex.codecogs.com/svg.latex?&space;\alpha_{ij} = F_{\alpha}(\theta_{i}, \delta_{j}, a_{j}) = \bigg(\frac{\theta_{i}}{\delta_{j}}\bigg)^{a_{j}}, " title="\alpha_{ij} = F_{\alpha}(\theta_{i}, \delta_{j}, a_{j}) = \bigg(\frac{\theta_{i}}{\delta_{j}}\bigg)^{a_{j}}, " /><br>

<img src="https://latex.codecogs.com/svg.latex?&space;\beta_{ij} = F_{\beta}(\theta_{i}, \delta_{j}, a_{j}) = \bigg(\frac{1 - \theta_{i}}{1 - \delta_{j}}\bigg)^{a_{j}}," title="\beta_{ij} = F_{\beta}(\theta_{i}, \delta_{j}, a_{j}) = \bigg(\frac{1 - \theta_{i}}{1 - \delta_{j}}\bigg)^{a_{j}}, " /><br>

<img src="https://latex.codecogs.com/svg.latex?&space;\theta_{i} \sim B(1,1), \delta_{j} \sim B(1,1), a_{j} \sim N(1, \sigma^{2}_{0})," title="\theta_{i} \sim B(1,1), \delta_{j} \sim B(1,1), a_{j} \sim N(1, \sigma^{2}_{0}), " /><br>

where,

<img src="https://latex.codecogs.com/svg.latex?&space;E[p_{ij} | \theta_i,\delta_j,a_j] = \frac{\alpha_{ij}}{\alpha_{ij} + \beta_{ij}} = \frac{1}{1 - \big(\frac{\delta_{j}}{1 - \delta_{j}}\big)^{a_{j}}\cdot \big(\frac{\theta_{i}}{1 - \theta_{i}}\big)^{ - a_{j}} }," title="E[p_{ij} | \theta_i,\delta_j,a_j] = \frac{\alpha_{ij}}{\alpha_{ij} + \beta_{ij}} = \frac{1}{1 - \big(\frac{\delta_{j}}{1 - \delta_{j}}\big)^{a_{j}}\cdot \big(\frac{\theta_{i}}{1 - \theta_{i}}\big)^{ - a_{j}} }, " /><br>
-->

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

# License
[MIT License](https://github.com/Manuelfjr/birt-sgd/blob/main/LICENSE)

Copyright (c) 2021 Manuel

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