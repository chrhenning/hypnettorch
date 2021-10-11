#!/usr/bin/env python3
# Copyright 2019 Christian Henning
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
# title          :utils/local_conv2d_layer.py
# author         :ch
# contact        :henningc@ethz.ch
# created        :10/30/2019
# version        :1.0
# python_version :3.6.8
"""
2D-convolutional layer without weight sharing
---------------------------------------------

This module implements a biologically-plausible version of a convolutional layer
that does not use weight-sharing. Such a convnet is termed "locally-connected
network" in:

    `Bartunov et al., "Assessing the Scalability of Biologically-Motivated Deep
    Learning Algorithms and Architectures", NeurIPS 2018.
    <http://papers.nips.cc/paper/8148-assessing-the-scalability-of-biologically\
-motivated-deep-learning-algorithms-and-architectures>`_

.. autosummary::

    hypnettorch.utils.local_conv2d_layer.LocalConv2dLayer
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypnettorch.utils.torch_utils import init_params

class LocalConv2dLayer(nn.Module):
    r"""Implementation of a locally-connected 2D convolutional layer.

    Since this implementation of a convolutional layer doesn't use weight-
    sharing, it will have more parameters than a conventional convolutional
    layer such as :class:`torch.nn.Conv2d`.

    For example, consider a convolutional layer with kernel size ``[K, K]``,
    ``C_in`` input channels and ``C_out`` output channels, that has an output
    feature map size of ``[H, W]``. Each receptive field [#f1]_ will have its
    own weights, a parameter tensor of size ``K x K``. Thus, in total the layer
    will have ``C_out * C_in * H * W * K * K`` weights compared to
    ``C_out * C_in * K * K`` weights that a conventional
    :class:`torch.nn.Conv2d` would have.

    Consider the :math:`i`-th input feature map :math:`F^{(i)}`
    (:math:`1 \leq i \leq C_{\text{in}}`), the :math:`j`-th output feature map
    :math:`G^{(j)}` (:math:`1 \leq j \leq C_{\text{out}}`) and the pixel with
    coordinates :math:`(x,y)` in the :math:`j`-th output feature map
    :math:`G^{(j)}_{xy}` (:math:`1 \leq x \leq W` and :math:`1 \leq y \leq H`).

    We denote the filter weights of this pixel connecting to the :math:`i`-th
    input feature map by :math:`W_{xy}^{(i,j)} \in \mathbb{R}^{K \times K}`.
    The corresponding receptive field inside :math:`F^{(i)}` that is used to
    compute pixel :math:`G^{(j)}_{xy}` is denoted by
    :math:`\hat{F}^{(i)}(x,y) \in \mathbb{R}^{K \times K}`.

    The bias weights for feature map :math:`G^{(j)}` are denoted by
    :math:`B^{(j)}`, with a scalar weight :math:`B^{(j)}_{xy}` for pixel
    :math:`(x,y)`.

    Using this notation, the computation of this layer can be described by the
    following formula

    .. math::

        G^{(j)}_{xy} &= B^{(j)}_{xy} + \sum_{i=1}^{C_{\text{in}}} \text{sum}
        (W_{xy}^{(i,j)} \odot \hat{F}^{(i)}(x,y)) \\
        &= B^{(j)}_{xy} + \sum_{i=1}^{C_{\text{in}}} \langle W_{xy}^{(i,j)},
        \hat{F}^{(i)}(x,y) \rangle_F

    where :math:`\text{sum}(\cdot)` is the unary operator that computes the sum
    of all elements in a matrix, :math:`\odot` denotes the Hadamard product
    and :math:`\langle \cdot, \cdot \rangle_F` denotes the Frobenius inner
    product, which computes the sum of the entries of the Hadamard product
    between real-valued matrices.

    **Implementation details**

    Let :math:`N` denote the batch size. We can use the function
    :func:`torch.nn.functional.unfold` to split our input, which is of shape
    ``[N, C_in, H_in, W_in]``, into receptive fields ``F_hat`` of dimension
    ``[N, C_in * K * K, H * W]``. The receptive field :math:`\hat{F}^{(i)}(x,y)`
    would then correspond to :code:`F_hat[:, i * K*K:(i+1) * K*K, y*H + x]`,
    assuming that indices now start at ``0`` and not at ``1``.

    In addition, we have a weight tensor ``W`` of shape
    ``[C_out, C_in * K * K, H * W]``.

    Now, we can compute the element-wise product of receptive fields and their
    filters by introducing a slack dimension into the shape of ``F_hat`` (i.e.,
    ``[N, 1, C_in * K * K, H * W]``) and by using broadcasting. ``F_hat * W``
    will result into a tensor of shape ``[N, C_out, C_in * K * K, H * W]``.
    By summing over the third dimension ``dim=2`` and reshaping the output we
    retrieve the result of our local convolutional layer.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        in_height (int): Height of the input feature maps, assuming that input
            feature maps have shape ``[C_in, H, W]`` (omitting the batch
            dimension). This argument is necessary to compute the size of
            output feature maps, as we need a filter for each pixel in each
            output feature map.
        in_width (int): Width of input feature maps.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution.
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input.
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            There will be one scalar bias per filter.
        no_weights (bool): If ``True``, the layer will have no trainable
            weights. Hence, weights are expected to be passed to the
            :meth:`forward` method.

    .. rubric:: Footnotes

    .. [#f1] For each of the ``C_in`` input feature maps, there is one receptive
       field for each pixel in all ``C_out`` feature maps.
    """
    def __init__(self, in_channels, out_channels, in_height, in_width,
                 kernel_size, stride=1, padding=0, bias=True, no_weights=False):
        super(LocalConv2dLayer, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._in_height = in_height
        self._in_width = in_width
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._has_bias = bias
        self._no_weights = no_weights

        self._out_height = (in_height - kernel_size[0] + 2 * padding[0]) // \
            stride[0] + 1
        self._out_width = (in_width - kernel_size[1] + 2 * padding[1]) // \
            stride[1] + 1

        # Size of a single receptive field.
        rf_size = in_channels * kernel_size[0] * kernel_size[1]
        self._rf_size = rf_size
        # Number of pixels per output feature map.
        num_pix = self._out_height * self._out_width
        self._num_pix = num_pix

        self._weights = None
        self._param_shapes = [[out_channels, rf_size, num_pix]]
        if bias:
            self._param_shapes.append([out_channels, num_pix])

        if not no_weights:
            self._weights = nn.ParameterList()

            self.register_parameter('filters', nn.Parameter( \
                torch.Tensor(*self._param_shapes[0]), requires_grad=True))
            self._weights.append(self.filters)

            if bias:
                self.register_parameter('bias', nn.Parameter( \
                    torch.Tensor(*self._param_shapes[1]), requires_grad=True))
                self._weights.append(self.bias)

                init_params(self.filters, self.bias)
            else:
                self.register_parameter('bias', None)

                init_params(self.filters)

    @property
    def weights(self):
        """A list of all internal weights of this layer. If all weights are
        assumed to be generated externally, then this attribute will be
        ``None``.

        :type: torch.nn.ParameterList or None
        """
        return self._weights

    @property
    def param_shapes(self):
        """A list of list of integers. Each list represents the shape of a
        parameter tensor. Note, this attribute is independent of the attribute
        :attr:`weights`, it always comprises the shapes of all weight tensors as
        if the network would be stand-alone (i.e., no weights being passed to
        the :meth:`forward` method).

        :type: list
        """
        return self._param_shapes

    @property
    def out_height(self):
        """Height of the output feature maps.

        :type: int
        """
        return self._out_height

    @property
    def out_width(self):
        """Width of the output feature maps.

        :type: int
        """
        return self._out_width

    def forward(self, x, weights=None):
        """Compute output of local convolutional layer.

        Args:
            x: The input images of shape ``[N, C_in, H_in, W_in]``, where ``N``
                denotes the batch size..
            weights: Weights that should be used instead of the internally
                maintained once (determined by attribute :attr:`weights`). Note,
                if ``no_weights`` was ``True`` in the constructor, then this
                parameter is mandatory.

        Returns:
            The output feature maps of shape ``[N, C_out, H, W]``.
        """
        if self._no_weights and weights is None:
            raise ValueError('Layer was generated without weights. ' +
                             'Hence, "weights" option may not be None.')

        if weights is None:
            filters = self.filters
            bias = self.bias
        else:
            assert(len(weights) == len(self.param_shapes))
            for i, p in enumerate(weights):
                assert(np.all(np.equal(p.shape, self.param_shapes[i])))

            filters = weights[0]
            if self._has_bias:
                bias = weights[1]
            else:
                bias = None

        # Extract receptive fields.
        F_hat = F.unfold(x, self._kernel_size, padding=self._padding,
                         stride=self._stride)
        assert(np.all(np.equal(F_hat.shape[1:], 
                               [self._rf_size, self._num_pix])))

        # Insert extra dim for broadcasting.
        F_hat = F_hat.view(-1, 1, self._rf_size, self._num_pix)

        # Compute Frobenius inner product.
        G = (F_hat * filters).sum(dim=2).squeeze(dim=2)
        assert(np.all(np.equal(G.shape[1:],
                               [self._out_channels, self._num_pix])))

        if bias is not None:
            G = G + bias

        G = G.view(-1, self._out_channels, self._out_height, self._out_width)

        return G

if __name__ == '__main__':
    pass


