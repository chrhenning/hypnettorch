.. Hypernet Algorithms documentation master file, created by
   sphinx-quickstart on Thu Sep  5 10:29:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*hypnettorch* - Hypernetworks in PyTorch
========================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   Data Handlers <data.rst>
   Hypernets <hnets.rst>
   Hyperparameter Search <hpsearch.rst>
   Main Networks <mnets.rst>
   Utilities <utils.rst>
   Tutorials <tutorials.rst>
   Examples <examples.rst>

This package provides functionalities to easily work with hypernetworks in PyTorch. A hypernetwork :math:`h(\mathbf{e}, \theta)` is a neural network with parameters :math:`\theta` that generates the parameters :math:`\omega` of another neural network :math:`f(\mathbf{x}, \omega)`, called *main network*. These two network types require specialized implementations. For instance, a *main network* must have the ability to receive its own weights :math:`\omega` as additional input to the ``forward`` method (see subpackage :ref:`mnets <mnets-reference-label>`). A collection of different hypernetwork implementations can be found in subpackage :ref:`hnets <hnets-reference-label>`.

Installation
------------

See `here <https://github.com/chrhenning/hypnettorch#installation>`__.

Usage
-----

Check out the :ref:`tutorials <tutorials-reference-label>`, especially the `getting started <https://github.com/chrhenning/hypnettorch/blob/master/hypnettorch/tutorials/getting_started.ipynb>`__ tutorial.

You can also check out :ref:`example implementations <exmp-reference-label>` that make use of ``hypnettorch``.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
