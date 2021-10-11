#!/usr/bin/env python3
# Copyright 2020 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :hnets/deconv_hnet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :04/16/2020
# @version        :1.0
# @python_version :3.6.10
"""
Deconvolutional Hypernetwork with Self-Attention Layers
-------------------------------------------------------

The module :mod:`hnets.deconv_hnet` implements a hypernetwork that uses
transpose convolutions (like the generator of a GAN) to generate weights.
Though, as convolutions usually suffer from only capturing local correlations
sufficiently, we incorporate the self-attention mechanism developed by

    Zhang et al., `Self-Attention Generative Adversarial Networks \
<https://arxiv.org/abs/1805.08318>`__, 2018.

See :class:`utils.self_attention_layer.SelfAttnLayerV2` for details on this
layer type.

The purpose of this network can be seen as the convolutional analogue of the
fully-connected :class:`hnets.mlp_hnet.HMLP`. Hence, it produces all weights in
one go; and does not utilize chunking to obtain better weight compression ratios
(a chunked version can be found in module :mod:`hnets.chunked_deconv_hnet`).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn

from hypnettorch.hnets.hnet_interface import HyperNetInterface
from hypnettorch.utils.self_attention_layer import SelfAttnLayerV2

class HDeconv(nn.Module, HyperNetInterface):
    """Implementation of a `deconvolutional full hypernet`.

    This is a convolutional network, employing transpose convolutions. The
    network structure is inspired by the
    `DCGAN <https://arxiv.org/abs/1511.06434>`__ generator structure, though,
    we are additionally using self-attention layers to model global
    dependencies.

    In general, each transpose convolutional layer will roughly double its
    input size. Though, we set the hard constraint that if the input size of
    a transpose convolutional layer would be smaller 4, then it doesn't change
    the size.

    The network allows to maintain a set of embeddings internally that can be
    used as conditional input (cmp. :class:`hnets.mlp_hnet.HMLP`).

    Args:
        (....): See constructor arguments of class
            :class:`hnets.mlp_hnet.HMLP`.
        hyper_img_shape (tuple): Since the network has a (de-)convolutional
            output layer, the output will be in an image-like shape. Therefore,
            it won't be possible to precisely produce the number of weights
            prescribed by ``target_shapes``. Therefore, the `hyper-image` size
            defined via this option has to be chosen big enough, i.e., the
            number of pixels must be greater equal than the number of weights to
            be produced. Remaining pixels will be discarded.

            This option has to be a tuple ``(width, height)``, denoting the
            internal output shape of the the hypernet. The number of output
            channels is assumed to be 1, except if specified otherwise via
            ``(width, height, channels)``.
        num_layers (int): The number of transpose convolutional layers including
            the initial fully-connected layer.
        num_filters (list, optional): List of integers of length
            ``num_layers-1``. The number of output channels in each hidden
            transpose conv. layer. By default, the number of filters in the
            last hidden layer will be ``128`` and doubled in every prior layer.

            Note:
                The output of the first layer (which is fully-connected) is here
                considered to be in the shape of an image tensor.
        kernel_size (int, tuple or list, optional): A single number, a tuple
            ``(k_x, k_y)`` or a list of scalars/tuples of length
            ``num_layers-1``. Specifying the kernel size in each convolutional
            layer.
        sa_units (tuple or list): List of integers, each representing the index
            of a layer in this network after which a self-attention unit should
            be inserted. For instance, index 0 represents the
            fully-connected layer. The last layer may not be chosen.
    """
    def __init__(self, target_shapes, hyper_img_shape, uncond_in_size=0,
                 cond_in_size=8, num_layers=5, num_filters=None, kernel_size=5,
                 sa_units=(1, 3), verbose=True, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_uncond_weights=False, no_cond_weights=False,
                 num_cond_embs=1, use_spectral_norm=False,
                 use_batch_norm=False):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        HyperNetInterface.__init__(self)

        if use_spectral_norm:
            raise NotImplementedError('Spectral normalization not yet ' +
                                      'implemented for this hypernetwork type.')

        raise NotImplementedError()

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed'):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.mlp_hnet.HMLP.forward`.

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """
        raise NotImplementedError()

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This network does not have any distillation targets.

        Returns:
            ``None``
        """
        return None

    def get_cond_in_emb(self, cond_id):
        """Get the ``cond_id``-th (conditional) input embedding.

        Args:
            cond_id (int): Determines which input embedding should be returned
                (the ID has to be between ``0`` and ``num_cond_embs-1``, where
                ``num_cond_embs`` denotes the corresponding constructor
                argument).

        Returns:
            (torch.nn.Parameter)
        """
        if self.conditional_params is None:
            raise RuntimeError('Input embeddings are not internally ' +
                               'maintained!')
        if not isinstance(cond_id, int) or cond_id < 0 or \
                cond_id >= len(self.conditional_params):
            raise RuntimeError('Option "cond_id" must be between 0 and %d!' \
                               % (len(self.conditional_params)-1))
        return self.conditional_params[cond_id]

if __name__ == '__main__':
    pass
