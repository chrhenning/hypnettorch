# Hypernetworks for PyTorch

[This package](https://hypnettorch.readthedocs.io) contains utilities that make it easy to work with [hypernetworks](https://arxiv.org/abs/1609.09106) in [PyTorch](https://pytorch.org/).

## Installation

You can either install the latest package version via

```
python3 -m pip install hypnettorch
```

or, you directly install the current sources

```
python3 -m pip install git+https://github.com/chrhenning/hypnettorch
```

#### Installation for developers

If you actively develop the package, it is easiest to install it in `development mode`, such that all changes that are done to source files are directly visible when you use the package.

Clone the repository to a location of your choice

```
git clone https://github.com/chrhenning/hypnettorch.git
```

and move inside the cloned repo

```
cd ./hypnettorch
```

Now, you can simply **install** the package in `editable` mode, which will ensure that you can easily update the package sources (cf. [development mode](https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html))

```
pip3 install --editable . --user
```

Since the package was installed in `editable` mode, you can always update the sources simply by pulling the most recent code

```
git pull
```

You can **uninstall** the package at any point by running `python3 setup.py develop -u`.

## Usage

The basic functionalities of the package are quite intuitive and easy to use, e.g.,

```python
import torch
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP
mnet = MLP(n_in=8, no_weights=True)
hnet = HMLP(mnet.param_shapes)
weights = hnet.forward(cond_id=0)
inputs = torch.rand(32, 8)
mnet.forward(inputs, weights=weights)
```

There are several [tutorials](https://github.com/chrhenning/hypnettorch/tree/master/hypnettorch/tutorials). Check out the [getting started](https://github.com/chrhenning/hypnettorch/blob/master/hypnettorch/tutorials/getting_started.ipynb) tutorial when working with ``hypnettorch`` for the first time.

## Documentation

The documentation can be found [here](https://hypnettorch.readthedocs.io).

#### Note for developers

The documentation can be build using 

```
python3 setup.py build_sphinx
```

and opened via the file [index.html](docs/html/index.html).

## Citation

When using this package in your research project, please consider citing one of our papers for which this package has been developed.

```
@inproceedings{oshg2019hypercl,
title={Continual learning with hypernetworks},
author={Johannes von Oswald and Christian Henning and Jo{\~a}o Sacramento and Benjamin F. Grewe},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://arxiv.org/abs/1906.00695}
}
```

```
@inproceedings{ehret2020recurrenthypercl,
  title={Continual Learning in Recurrent Neural Networks},
  author={Benjamin Ehret and Christian Henning and Maria R. Cervera and Alexander Meulemans and Johannes von Oswald and Benjamin F. Grewe},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://arxiv.org/abs/2006.12109}
}
```

```
@inproceedings{posterior:replay:2021:henning:cervera,
title={Posterior Meta-Replay for Continual Learning}, 
      author={Christian Henning and Maria R. Cervera and Francesco D'Angelo and Johannes von Oswald and Regina Traber and Benjamin Ehret and Seijin Kobayashi and Benjamin F. Grewe and João Sacramento},
booktitle={Conference on Neural Information Processing Systems},
year={2021},
url={https://arxiv.org/abs/2103.01133}
}
```
