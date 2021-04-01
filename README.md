[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?&logo=license)](https://opensource.org/licenses/MIT)
[![Author](https://img.shields.io/badge/author-manuelfjr-blue?&logo=github)](https://github.com/Manuelfjr)
[![Author](https://img.shields.io/badge/author-tmfilho-blue?&logo=github)](https://github.com/tmfilho)
[![PyPI pyversions](https://img.shields.io/badge/python-v3.8.5-orange?&logo=python)](https://pypi.python.org/pypi/ansicolortags/)

# birt-sgd
**BIRTSGD:** Evaluation of clustering methods using Beta^3-IRT with descending gradient

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

# Usage
```py
from birt import BIRTSGD

birtsgd = BIRTSGD(
    learning_rate=0.1, epochs=20, 
    n_models=20, n_instances=100, 
    n_batchs=5, random_seed=1
    )
```

