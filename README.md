[![PyPI version](https://badgen.net/pypi/v/birt-gd)](https://pypi.org/project/birt-gd/#history)
[![Total downloads](https://static.pepy.tech/badge/birt-gd)](https://pepy.tech/project/birt-gd)
[![PyPI - Downloads/month](https://img.shields.io/pypi/dm/birt-gd?style=flat-square&color=darkgreen)](https://pypi.org/project/birt-gd/)
[![PyPI - Downloads/week](https://img.shields.io/pypi/dw/birt-gd?style=flat-square&color=darkgreen)](https://pypi.org/project/birt-gd/)
[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Open issues](https://badgen.net/github/open-issues/manuelfjr/birt-gd)](https://github.com/Manuelfjr/birt-gd/issues?q=is%3Aopen+is%3Aissue)
[![Closed issues](https://badgen.net/github/closed-issues/manuelfjr/birt-gd)](https://github.com/Manuelfjr/birt-gd/issues?q=is%3Aissue+is%3Aclosed)

<a href="https://www.buymeacoffee.com/manuelfjr" target="_blank"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-5F7FFF?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>

# birt-gd

**BIRT** implements &beta;&sup3;-IRT and &beta;&#8308;-IRT (Beta Item Response Theory) fit by gradient descent in TensorFlow. Unlike classic IRT, which models binary correct/incorrect responses, Beta-IRT models a **continuous** response p<sub>ij</sub> &isin; (0, 1) &mdash; e.g. the probability that classifier/respondent *j* assigns to the correct class of item *i* &mdash; which makes it well suited to evaluating and comparing probabilistic classifiers, not just human test-takers.

## Table of contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Citation](#citation)
- [Support](#support)
- [License](#license)
- [Author](#author) / [Contributors](#contributors)

## Background

Given a matrix *X* of response probabilities p<sub>ij</sub> &sim; &Beta;(&alpha;<sub>ij</sub>, &beta;<sub>ij</sub>) &mdash; the probability of respondent *j* correctly classifying item *i* &mdash; the model estimates:

- **ability** (&theta;<sub>i</sub>) per respondent
- **difficulty** (&delta;<sub>j</sub>) per item
- **discrimination** (&omega;<sub>j</sub> / &beta;<sub>j</sub> for &beta;&#8308;-IRT, a single a<sub>j</sub> for &beta;&sup3;-IRT) per item

using:

&theta;<sub>i</sub> = &sigma;(t<sub>i</sub>), &nbsp; &delta;<sub>j</sub> = &sigma;(d<sub>j</sub>), &nbsp; &omega;<sub>j</sub> = softplus(o<sub>j</sub>), &nbsp; &beta;<sub>j</sub> = tanh(b<sub>j</sub>)

E[p<sub>ij</sub> | &theta;<sub>i</sub>, &delta;<sub>j</sub>, &omega;<sub>j</sub>, &beta;<sub>j</sub>] = 1 / (1 + (&delta;<sub>j</sub>/(1-&delta;<sub>j</sub>))<sup>&omega;<sub>j</sub>&beta;<sub>j</sub></sup> &times; (&theta;<sub>i</sub>/(1-&theta;<sub>i</sub>))<sup>-&omega;<sub>j</sub>&beta;<sub>j</sub></sup>)

&beta;&#8308;-IRT (`Beta4`) fits this with unconstrained gradient descent (link functions remove the bounded-parameter symmetry problem &beta;&sup3;-IRT has); `set_priors=True` (default) initializes abilities/difficulties from the data's own moments instead of random draws, which converges faster and more reliably. &beta;&sup3;-IRT (`Beta3`) is the earlier, single-discrimination-parameter model — kept for comparison/backwards compatibility. See [Citation](#citation) for the papers behind both.

## Installation

```bash
pip install birt-gd
```

### Requirements

- Python >= 3.10
- tensorflow ^2.18.0
- pandas ^2.2.3
- scikit-learn ^1.6.1
- matplotlib ^3.10.0
- seaborn ^0.13.2
- tqdm ^4.67.1

## Usage

```py
from birt import Beta4
import pandas as pd

data = pd.DataFrame({
    'a': [0.99, 0.89, 0.87, 0.50],
    'b': [0.32, 0.25, 0.45, 0.20],
    'c': [0.50, 0.50, 0.50, 0.50],
})

bgd = Beta4(
    learning_rate=1,
    epochs=10000,
    n_respondents=data.shape[1],
    n_items=data.shape[0],
    n_inits=1000,
    random_seed=1,
    tol=10**(-5),
    set_priors=True,
)
bgd.fit(data.values)

bgd.abilities        # array([0.626, 0.416, 0.474], dtype=float32)
bgd.difficulties     # array([0.456, 0.478, 0.442, 0.608], dtype=float32)
bgd.discriminations  # array([0.992, 1.000, 0.961, 0.792], dtype=float32)
bgd.score            # Pseudo-R2, e.g. 0.888
```

`Beta3` shares the same interface (drop `set_priors`, since &beta;&sup3;-IRT has a single discrimination parameter):

```py
from birt import Beta3

b3 = Beta3(learning_rate=1, epochs=10000, n_respondents=data.shape[1], n_items=data.shape[0])
b3.fit(data.values)
```

### Summary

```py
bgd.summary()
```

```
        ESTIMATES
        -----
                        | Min      1Qt      Median   3Qt      Max      Std.Dev
        Ability         | 0.00012  0.21369  0.57847  0.69513  0.93050  0.33468
        Difficulty      | 0.03876  0.27725  0.58860  0.84598  0.96604  0.30748
        Discrimination  | 0.25266  0.73648  1.04295  1.35130  2.09018  0.47445
        pij             | 0.00000  0.04613  0.40412  0.81140  0.99958  0.36590
        -----
        Pseudo-R2       | 0.88788
```

### Plots

`bgd.plot(xaxis=..., yaxis=..., ann=True, kwargs={'color': 'red'})` — scatter of any pair among `discrimination`, `difficulty`, `ability`, `average_response`, `average_item`.

`bgd.boxplot(x=..., y=..., kwargs={...})` — boxplot of `ability`, `difficulty` or `discrimination`.

<p float="left">
  <img src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/dis_diff_ex.png" width="32%" alt="discrimination vs difficulty scatterplot">
  <img src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/diff_av_ex2.png" width="32%" alt="difficulty vs average item scatterplot">
  <img src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ab_av_ex3.png" width="32%" alt="ability vs average response scatterplot">
</p>
<p float="left">
  <img src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ex4.png" width="32%" alt="ability boxplot">
  <img src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ex5.png" width="32%" alt="difficulty boxplot">
  <img src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ex6.png" width="32%" alt="discrimination boxplot">
</p>

More end-to-end examples: [example/00_example.ipynb](example/00_example.ipynb).

## Development

```bash
git clone https://github.com/Manuelfjr/birt-gd
cd birt-gd
poetry install
poetry shell
```

`mc_analysis/` holds the Monte Carlo simulation study used to validate the model; it ships with the repo but not with the PyPI package.

### Contributing

Issues and pull requests are welcome at [github.com/Manuelfjr/birt-gd](https://github.com/Manuelfjr/birt-gd). There's no test suite yet, so please describe how a change was verified (e.g. output of the [Usage](#usage) example) in the PR description.

## Citation

`birt-gd` is the reference implementation for the following papers — please cite the one matching the model you use (`Beta4` &rarr; &beta;&#8308;-IRT, `Beta3` &rarr; &beta;&sup3;-IRT):

```bibtex
@article{ferreirajunior2023beta4irt,
  title   = {{$\beta^4$-IRT}: A New {$\beta^3$-IRT} with Enhanced Discrimination Estimation},
  author  = {Ferreira-Junior, Manuel and Reinaldo, Jessica T. S. and Silva Filho, Telmo M. and Lima Neto, Eufrasio A. and Prudencio, Ricardo B. C.},
  journal = {arXiv preprint arXiv:2303.17731},
  year    = {2023}
}

@inproceedings{chen2019beta3irt,
  title     = {{$\beta^3$-IRT}: A New Item Response Model and its Applications},
  author    = {Chen, Yu and Silva Filho, Telmo and Prudencio, Ricardo B. C. and Diethe, Tom and Flach, Peter},
  booktitle = {Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2019}
}
```

## Support

- E-mail: [ferreira.jr.ufpb@gmail.com](mailto:ferreira.jr.ufpb@gmail.com)
- Site: [manuelfjr.github.io](https://manuelfjr.github.io/)

## License

[GNU General Public License v3.0](LICENSE) &copy; Manuel Ferreira Junior

## Author

<table>
  <tr>
    <td align="center" width="120"><a href="https://manuelfjr.github.io/"><img style="border-radius: 50%;" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/author.jpg" width="100px;" alt="Manuel Ferreira Junior"/><br /><sub><b>Manuel Ferreira Junior</b></sub></a></td>
  </tr>
</table>

## Contributors

<table>
  <tr>
    <td align="center" width="120"><a href="https://github.com/tmfilho"><img style="border-radius: 50%;" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/coauthor1.jpeg" width="100px;" alt="Telmo de Menezes e Silva Filho"/><br /><sub><b>Telmo de Menezes e Silva Filho</b></sub></a></td>
    <td align="center" width="120"><a href="https://flach.github.io/"><img style="border-radius: 50%;" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/coauthor5.jpg" width="100px;" alt="Peter Flach"/><br /><sub><b>Peter Flach</b></sub></a></td>
    <td align="center" width="120"><a href="http://lattes.cnpq.br/2984888073123287"><img style="border-radius: 50%;" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/coauthor3.jpg" width="100px;" alt="Ricardo Prudêncio"/><br /><sub><b>Ricardo Prud&ecirc;ncio</b></sub></a></td>
    <td align="center" width="120"><a href="http://lattes.cnpq.br/5580004940091667"><img style="border-radius: 50%;" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/coauthor4.jpg" width="100px;" alt="Eufrásio de Andrade Lima Neto"/><br /><sub><b>Eufr&aacute;sio de Andrade Lima Neto</b></sub></a></td>
  </tr>
</table>
