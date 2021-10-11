# Hypernetworks for PyTorch

This package combines utilities that make it easy to work with hypernetworks in [PyTorch](https://pytorch.org/).

## Installation

**TODO**

## Usage

```python
from hypnettorch.mnets import MLP
net = MLP()
```

## Documentation

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
      author={Christian Henning and Maria R. Cervera and Francesco D'Angelo and Johannes von Oswald and Regina Traber and Benjamin Ehret and Seijin Kobayashi and João Sacramento and Benjamin F. Grewe},
booktitle={Conference on Neural Information Processing Systems},
year={2021},
url={https://arxiv.org/abs/2103.01133}
}
```
