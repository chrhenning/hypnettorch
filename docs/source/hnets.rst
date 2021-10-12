.. _hnets-reference-label:

Hypernetworks
*************

.. contents::

A `hypernetwork <https://arxiv.org/abs/1609.09106>`__ is a neural network that produces the weights of another network. As such, it can be seen as a specific type of main network (aka neural network). Therefore, each hypernetwork has a specific interface :class:`hnets.hnet_interface.HyperNetInterface` which is derived from the main network interface :class:`mnets.mnet_interface.MainNetInterface`. 

.. note::
    All hypernetworks in this subpackage implement the abstract interface :class:`hnets.hnet_interface.HyperNetInterface` to provide a consistent interface for users.

.. automodule:: hypnettorch.hnets.hnet_interface
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hypnettorch.hnets.chunked_deconv_hnet
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hypnettorch.hnets.chunked_mlp_hnet
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hypnettorch.hnets.deconv_hnet
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hypnettorch.hnets.hnet_container
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hypnettorch.hnets.hnet_helpers
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hypnettorch.hnets.hnet_perturbation_wrapper
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hypnettorch.hnets.mlp_hnet
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hypnettorch.hnets.structured_hmlp_examples
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hypnettorch.hnets.structured_mlp_hnet
    :members:
    :undoc-members:
    :show-inheritance:
