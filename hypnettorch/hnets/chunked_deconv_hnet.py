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
# @title          :hnets/chunked_deconv_hnet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :03/30/2020
# @version        :1.0
# @python_version :3.6.10
"""
Chunked Deconvolutional Hypernetwork with Self-Attention Layers
---------------------------------------------------------------

The module :mod:`hnets.chunked_deconv_hnet` implements a chunked version of the
transpose convolutional hypernetwork represented by class
:class:`hnets.deconv_hnet.HDeconv` (similar as to
:class:`hnets.chunked_mlp_hnet.ChunkedHMLP` represents a chunked version of the
full hypernetwork :class:`hnets.mlp_hnet.HMLP`).

Therefore, an instance of class :class:`ChunkedHDeconv` manages internally an
instance of class :class:`hnets.deconv_hnet.HDeconv`, which is invoked multiple
time with a different additional input (the so called `chunk embedding`) to
produce a chunk of the target weights at a time, which are later put together.
See description of module :mod:`hnets.chunked_mlp_hnet` for more details.

Note:
    This type of hypernetwork is completely agnostic to the architecture of the
    target network. The splits happen at arbitrary locations in the flattened
    target network weight vector.
"""
import numpy as np
import torch
import torch.nn as nn
from warnings import warn

from hypnettorch.hnets.hnet_interface import HyperNetInterface

class ChunkedHDeconv(nn.Module, HyperNetInterface):
    """Implementation of a `chunked deconvolutional hypernet`.

    The ``target_shapes`` will be flattened and split into chunks of size
    ``chunk_size = np.prod(hyper_img_shape)``. In total, there will be
    ``np.ceil(self.num_outputs/chunk_size)`` chunks, where the last chunk
    produced might contain a remainder that is discarded.

    Each chunk has it's own `chunk embedding` that is fed into the underlying
    hypernetwork.

    Note:
        It is possible to set ``uncond_in_size`` and ``cond_in_size`` to zero
        if ``cond_chunk_embs`` is ``True``.

    Attributes:
        (....): See attributes of class
            :class:`hnets.chunked_mlp_hnet.ChunkedHMLP`.

    Args:
        (....): See constructor arguments of class
            :class:`hnets.deconv_hnet.HDeconv`.
        chunk_emb_size (int): See constructor arguments of class
            :class:`hnets.chunked_mlp_hnet.ChunkedHMLP`.
        cond_chunk_embs (bool): See constructor arguments of class
            :class:`hnets.chunked_mlp_hnet.ChunkedHMLP`.
    """
    def __init__(self, target_shapes, hyper_img_shape, chunk_emb_size=8,
                 cond_chunk_embs=False, uncond_in_size=0, cond_in_size=8,
                 num_layers=5, num_filters=None, kernel_size=5, sa_units=(1, 3),
                 verbose=True, activation_fn=torch.nn.ReLU(), use_bias=True,
                 no_uncond_weights=False, no_cond_weights=False,
                 num_cond_embs=1, use_spectral_norm=False,
                 use_batch_norm=False):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        HyperNetInterface.__init__(self)

        self._hnet = None

        raise NotImplementedError()

    @property
    def num_chunks(self):
        """Getter for read-only attribute :attr:`num_chunks`."""
        raise NotImplementedError()

    @property
    def chunk_emb_size(self):
        """Getter for read-only attribute :attr:`chunk_emb_size`."""
        return self._chunk_emb_size

    @property
    def cond_chunk_embs(self):
        """Getter for read-only attribute :attr:`cond_chunk_embs`."""
        return self._cond_chunk_embs

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed'):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.chunked_mlp_hnet.ChunkedHMLP.forward`.

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """
        raise NotImplementedError()

    def get_cond_in_emb(self, cond_id):
        """Get the ``cond_id``-th (conditional) input embedding.

        Args:
            (....): See docstring of method
                :meth:`hnets.deconv_hnet.HDeconv.get_cond_in_emb`.

        Returns:
            (torch.nn.Parameter)
        """
        return self._hnet.get_cond_in_emb(cond_id)

    def get_chunk_emb(self, chunk_id=None, cond_id=None):
        """Get the ``chunk_id``-th chunk embedding.

        Args:
            (....): See docstring of method
                :meth:`hnets.chunked_mlp_hnet.ChunkedHMLP.get_chunk_emb`.

        Returns:
            (torch.nn.Parameter)
        """
        raise NotImplementedError()

if __name__ == '__main__':
    pass


