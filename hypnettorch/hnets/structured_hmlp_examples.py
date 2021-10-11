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
# @title          :hnets/structured_hmlp_examples.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :05/02/2020
# @version        :1.0
# @python_version :3.6.10
"""
Example Instantiations of a Structured Chunked MLP - Hypernetwork
-----------------------------------------------------------------

The module :mod:`hnets.structured_hmlp_examples` provides helpers for example
instantiations of :class:`hnets.structured_mlp_hnet.StructuredHMLP`.

Functions in this module typically take a given main network and produce the
constructor arguments ``chunk_shapes``, ``num_per_chunk`` and ``assembly_fct``
of class :class:`hnets.structured_mlp_hnet.StructuredHMLP`.

Note:
    These examples should be used with care. They are meant as inspiration and
    might not cover all possible usecases.

.. autosummary::

    hypnettorch.hnets.structured_hmlp_examples.resnet_chunking
    hypnettorch.hnets.structured_hmlp_examples.wrn_chunking
"""
import math
import numpy as np
import torch
from warnings import warn

from hypnettorch.mnets.resnet import ResNet
from hypnettorch.mnets.wide_resnet import WRN

def resnet_chunking(net, gcd_chunking=False):
    r"""Design a structured chunking for a ResNet.

    A resnet as implemented in class :class:`mnets.resnet.ResNet` consists
    roughly of 5 parts:

    - An input convolutional layer with weight shape ``[C_1, C_in, 3, 3]``
    - 3 blocks of ``2*n`` convolutional layers each where the first layer has
      shape ``[C_i, C_j, 3, 3]`` with :math:`i \in \{2, 3, 4\}` and
      :math:`j \equiv i-1` and the remaining ``2*n-1`` layers have a weight
      shape of ``[C_i, C_i, 3, 3]``.
    - A final fully connected layer of shape ``[n_classes, n_hidden]``.

    Each layer may additionally have a bias vector and (if batch normalization
    is used) a scale and shift vector.

    For instance, if a resnet with biases and batchnorm is used and the first
    layer will be produced as one structured chunk, then the first chunk shape
    (see return value ``chunk_shapes``) will be:
    ``[[C_1, C_in, 3, 3], [C_1], [C_1], [C_1]]``.

    This function will chunk layer wise (i.e., a chunk always comprises up to
    4 elements: weights tensor, bias vector, batchnorm scale and shift). By
    default, layers with the same shape are grouped together. Hence, the
    standard return value contains 8 chunk shapes (input layer, first layer of
    each block, remaining layers of each block (which all have the same shape)
    and the fully-connected output layer). Therefore, the return value
    ``num_per_chunk`` would be as follows:
    ``[1, 1, 2*n-1, 1, 2*n-1, 1, 2*n-1, 1]``.

    Args:
        net (mnets.resnet.ResNet): The network for which the structured chunking
            should be devised.
        gcd_chunking (bool): If ``True``, the layers within the 3 resnet blocks
            will be produced by 4 chunks. Therefore, the greatest common divisor
            (gcd) of the feature sizes ``C_1, C_2, C_3, C_4`` is computed and
            the 6 middle ``chunk_shapes`` produced by default are replaced by 4
            chunk shapes ``[[C_gcd, C_i, 3, 3], [C_gcd]]`` (assuming no
            batchnorm is used). Note, the first and last entry of
            ``chunk_shapes`` will remain unchanged by this option.

            Hence, ``len(num_per_chunk) = 6`` in this case.

    Returns:
        (tuple): Tuple containing the following arguments that can be passed
        to the constructor of class
        :class:`hnets.structured_mlp_hnet.StructuredHMLP`.

        - **chunk_shapes** (list)
        - **num_per_chunk** (list)
        - **assembly_fct** (func)
    """
    if not isinstance(net, ResNet):
        raise ValueError('Function expects resnet as argument ""net".')

    if net._use_context_mod:
        raise NotImplementedError('This function doesn\'t handle context-mod ' +
                                  'layers yet!')

    # TODO Update implementation.
    #if net._param_shapes_meta is not None:
    #    warn('Note, at the time of implementation of this function, the ' +
    #         'resnet attribute "param_shapes_meta" was not yet implemented. ' +
    #         'Hence, this function implementation should be updated.')

    has_bn = net._use_batch_norm
    has_bias = net.has_bias
    n = net._n
    filter_sizes = net._filter_sizes

    num_layers = 6*n + 2
    factor = 1
    sub = 0
    if has_bias:
        factor += 1
    if has_bn:
        factor += 2
        sub = 2
    assert len(net.param_shapes) == factor * num_layers - sub

    if gcd_chunking:
        # Note, each of the `6*n` layers in the middle can be made up of
        # several chunks. We know that 1 layer has `C1` as input channel
        # dimension, 2n layers have `C2` and `C3` as input channel dimension and
        # 2n-1 layers have `C4` as input channel dimension. Though, depending on
        # the gcd, multiple chunks are required to produce the weights of 1
        # layer.
        num_per_chunk = [1, None, None, None, None, 1]
    else:
        num_per_chunk = [1, 1, 2*n-1, 1, 2*n-1, 1, 2*n-1, 1]

    chunk_shapes = []
    assembly_fct = None

    # Note, if batchnorm is used, then the first 2 * (6*n+1) weights belong to
    # batch normalization.
    bn_start = 0
    w_start = 2 * (6*n+1) if has_bn else 0

    ### First layer ###
    cs = []
    cs.append(net.param_shapes[w_start])
    if has_bias:
        cs.append(net.param_shapes[w_start+1])
    if has_bn:
        cs.extend(net.param_shapes[:2])
    chunk_shapes.append(cs)

    bn_start += 2
    w_start += 2 if has_bias else 1

    ### Resnet blocks ###
    c_div_gcd = None
    if gcd_chunking:
        gcd = math.gcd(filter_sizes[0], filter_sizes[1])
        gcd = math.gcd(gcd, filter_sizes[2])
        gcd = math.gcd(gcd, filter_sizes[3])

        # The first block is made up of layers requiring `C1//gcd` chunks each,
        # and so on ...
        fsl = filter_sizes
        c_div_gcd = [fsl[1] // gcd, fsl[2] // gcd, fsl[3] // gcd]

        for i, fs in enumerate(filter_sizes):
            if i == 0:
                #n_layers = 1
                n_chunks = c_div_gcd[0]
            elif i == 1:
                #n_layers = 2 * n
                n_chunks = c_div_gcd[0] * (2*n-1) + c_div_gcd[1]
            elif i == 2:
                #n_layers = 2 * n
                n_chunks = c_div_gcd[1] * (2*n-1) + c_div_gcd[2]
            else:
                #n_layers = 2 * n - 1
                n_chunks = c_div_gcd[2] * (2*n-1)
            num_per_chunk[1+i] = n_chunks

            cs = []
            cs.append([gcd, fs, *net._kernel_size])
            if has_bias:
                cs.append([gcd])
            if has_bn:
                cs.extend([[gcd], [gcd]])
            chunk_shapes.append(cs)

        bn_start += 2 * (6*n)
        w_start += (2 if has_bias else 1) * (6*n)
    else:
        for i in range(3): # For each resnet block
            # FIXME If two consecutive filter sizes are identical, we could
            # add one chunk shape for this block rather than 2.

            # First layer of block.
            cs = []
            cs.append(net.param_shapes[w_start])
            if has_bias:
                cs.append(net.param_shapes[w_start+1])
            if has_bn:
                cs.extend(net.param_shapes[bn_start:bn_start+2])
            chunk_shapes.append(cs)

            bn_start += 2
            w_start += 2 if has_bias else 1

            # Remaining 2*n-1 layers of block.
            cs = []
            cs.append(net.param_shapes[w_start])
            if has_bias:
                cs.append(net.param_shapes[w_start+1])
            if has_bn:
                cs.extend(net.param_shapes[bn_start:bn_start+2])
            chunk_shapes.append(cs)

            bn_start += 2
            w_start += 2 if has_bias else 1
            for ii in range(2*n-2):
                assert len(cs[0]) == 4
                assert np.all(np.equal(net.param_shapes[w_start], cs[0]))
                if has_bias:
                    assert len(cs[1]) == 1
                    assert np.all(np.equal(net.param_shapes[w_start+1], cs[1]))
                if has_bn:
                    o = 2 if has_bias else 1
                    assert len(cs[o]) == 1 and len(cs[o+1]) == 1
                    assert np.all(np.equal(net.param_shapes[bn_start], cs[o]))
                    assert np.all(np.equal(net.param_shapes[bn_start+1],
                                           cs[o+1]))

                bn_start += 2
                w_start += 2 if has_bias else 1

    ### Final layer ###
    cs = []
    cs.append(net.param_shapes[w_start])
    if has_bias:
        cs.append(net.param_shapes[w_start+1])
    # No batchnorm for last layer!
    chunk_shapes.append(cs)

    assert len(chunk_shapes) == len(num_per_chunk)

    assembly_fct = lambda x : _resnet_chunking_afct(x, net, chunk_shapes,
        num_per_chunk, gcd_chunking, c_div_gcd)

    return chunk_shapes, num_per_chunk, assembly_fct

def _resnet_chunking_afct(list_of_chunks, net, chunk_shapes, num_per_chunk,
                          gcd_chunking, c_div_gcd):
    """The ``assembly_fct`` function required by function
    :func:`resnet_chunking`.
    """
    assert len(list_of_chunks) == np.sum(num_per_chunk)

    has_bn = net._use_batch_norm
    has_bias = net.has_bias
    n = net._n

    bn_weights = []
    layer_weights = []

    cind = 0

    ### First layer ###
    layer_weights.append(list_of_chunks[cind][0])
    if has_bias:
        layer_weights.append(list_of_chunks[cind][1])
    if has_bn:
        bn_weights.extend(list_of_chunks[cind][-2:])

    cind += 1

    ### Resnet blocks ###
    if gcd_chunking:
        # Number of layers per channel size.
        n_per_c = [1, 2*n, 2*n, 2*n-1]

        layer_ind = 0
        for i, n_layer in enumerate(n_per_c):
            for l in range(n_layer):
                # Out of how many chunks does this layer consist?
                n_c = c_div_gcd[layer_ind // (2*n)]
                layer_ind += 1

                chunks = list_of_chunks[cind:cind+n_c]
                cind += n_c

                layer_weights.append(torch.cat([c[0] for c in chunks], dim=0))
                if has_bias:
                    layer_weights.append(torch.cat([c[1] for c in chunks],
                                                   dim=0))
                if has_bn:
                    bn_weights.append(torch.cat([c[-2] for c in chunks], dim=0))
                    bn_weights.append(torch.cat([c[-1] for c in chunks], dim=0))
    else:
        for i in range(3): # For each block.
            # First layer in block.
            layer_weights.append(list_of_chunks[cind][0])
            if has_bias:
                layer_weights.append(list_of_chunks[cind][1])
            if has_bn:
                bn_weights.extend(list_of_chunks[cind][-2:])
            cind += 1

            # Remaining layers in block.
            for _ in range(2*n-1):
                layer_weights.append(list_of_chunks[cind][0])
                if has_bias:
                    layer_weights.append(list_of_chunks[cind][1])
                if has_bn:
                    bn_weights.extend(list_of_chunks[cind][-2:])
                cind += 1

    ### Last layer ###
    # No batchnorm for last layer!
    layer_weights.append(list_of_chunks[-1][0])
    if has_bias:
        layer_weights.append(list_of_chunks[-1][1])

    return bn_weights + layer_weights

def wrn_chunking(net, ignore_bn_weights=True, ignore_out_weights=True,
                 gcd_chunking=False):
    r"""Design a structured chunking for a Wide-ResNet (WRN).

    This function is in principle similar to function :func:`resnet_chunking`,
    but with the goal to provide a chunking scheme that is identical to the one
    proposed in (accessed August 18th, 2020):

        Sacramento et al., "Economical ensembles with hypernetworks", 2020
        https://arxiv.org/abs/2007.12927

    Therefore, a WRN as implemented in class :class:`mnets.wide_resnet.WRN`
    is required. For instance, a `WRN-28-10-B(3,3)` can be instantiated as
    follows, using batchnorm but no biases in all convolutional layers:

    .. code-block:: python

        wrn = WRN(in_shape=(32, 32, 3), num_classes=10, n=4, k=10,
                  num_feature_maps=(16, 16, 32, 64), use_bias=False,
                  use_fc_bias=True, no_weights=False, use_batch_norm=True)

    We denote channel sizes by ``[C_in, C_1, C_2, C_3, C_4]``, where ``C_in`` is
    the number of input channels and the remaining ``C_1, C_2, C_3, C_4`` denote
    the channel size per convolutional group. The widening factor is denoted by
    ``k``.

    In general, there will be up to 11 `layer groups`, which will be realized
    by separate hypernetworks (cmp table S1 in
    `Sacramento et al. <https://arxiv.org/pdf/2007.12927.pdf>`_):

    - ``0``: Input layer weights. If the network's convolutional layers have
      biases and batchnorm layers while ``ignore_bn_weights=False``, then this
      hypernet will produce weights of shape
      ``[[C_1, C_in, 3, 3], [C_1], [C_1], [C_1]]``. However, without
      convolutional bias terms and with ``ignore_bn_weights=True``, the hypernet
      will only produce weights of shape ``[[C_1, C_in, 3, 3]]``. This
      specification applies to all layer groups generating convolutional layers.
    - ``1``: This layer group will generate the weights of the first
      convolutional layer in the first convolutional group, e.g.,
      ``[[k*C_2, C_1, 3, 3]]``. Let's define
      ``r = max(k*C_2/C_1, C_1/k*C_2)``. If ``r=1`` or ``r=2`` or
      ``gcd_chunking=True``, then this group is merged with layer group ``2``.
    - ``2``: The remaining convolutional layer of the first convolutional group.
      If ``r=1``, ``r=2`` or ``gcd_chunking=True``, then all convolutional
      layers of the first group are generated. However, if biases or batch norm
      weights have to be generated, then this form of chunking leads to
      redundancy. Imagine bias terms are used and that the first layer in this
      convolutional group has weights ``[[160, 16, 3, 3], [160]]``, while the
      remaining layers have shape ``[[160, 160, 3, 3], [160]]``. If that's the
      case, the hypernetwork output will be of shape
      ``[[160, 16, 3, 3], [160]]``, meaning that 10 chunks have to be produced
      for each except the first layer. However, this means that per
      convolutional layer 10 bias vectors are generated, while only one is
      needed and therefore the other 9 will go to waste.
    - ``3``: Same as ``1`` for the first layer in the second convolutional
      group.
    - ``4`` (labelled as ``3`` in the paper): Same as ``2`` for all
      convolutional layers (potentially excluding the first) in the second
      convolutional group.
    - ``5``: Same as ``1`` for the first layer in the third convolutional
      group.
    - ``6`` (labelled as ``4`` in the paper): Same as ``2`` for all
      convolutional layers (potentially excluding the first) in the third
      convolutional group.
    - ``7`` (labelled as ``5`` in the paper): If existing, this hypernetwork
      produces the 1x1 convolutional layer realizing the residual connection
      connecting the first and second residual block in the first convolutional
      group.
    - ``8`` (labelled as ``6`` in the paper): Same as ``7`` but for the first
      residual connection in the second convolutional group.
    - ``9`` (labelled as ``7`` in the paper): Same as ``7`` but for the first
      residual connection in the third convolutional group.
    - ``10``: This hypernetwork will produce the weights of the fully connected
      output layer, if ``ignore_out_weights=False``.

    Thus, the WRN weights would maximally be produced by 11 different sub-
    hypernetworks.

    Note:
        There is currently an implementation mismatch, such that the
        implementation provided here does not 100% mimic the architecture
        described in
        `Sacramento et al. <https://arxiv.org/pdf/2007.12927.pdf>`_.

        To be specific, given the ``wrn`` generated above, the hypernetwork
        output for layer group ``2`` will be of shape ``[160, 160, 3, 3]``,
        while the paper expects a vertical chunking with a hypernet output of
        shape ``[160, 80, 3, 3]``.

    Args:
        net (mnets.wide_resnet.WRN): The network for which the structured
            chunking should be devised.
        ignore_bn_weights (bool): If ``True``, even if the given ``net`` has
            batchnorm weights, they will be ignored by this function.
        ignore_out_weights (bool): If ``True``, output weights (layer group
            ``10``) will be ignored by this function.
        gcd_chunking (bool): If ``True``, layer groups ``1``, ``3`` and ``5``
            are ignored. Instead, the greatest common divisor (gcd) of input and
            output feature size in a convolutional group is computed and weight
            tensors within a convolutional group (i.e., layer groups ``2``,
            ``4`` and ``6``) are chunked according to this value. However, note
            that this will cause the generation of unused bias and batchnorm
            weights if existing (cp. description of layer group ``2``).

    Returns:
        (tuple): Tuple containing the following arguments that can be passed
        to the constructor of class
        :class:`hnets.structured_mlp_hnet.StructuredHMLP`.

        - **chunk_shapes** (list)
        - **num_per_chunk** (list)
        - **assembly_fct** (func)
    """
    if not isinstance(net, WRN):
        raise ValueError('Function expects WRN as argument ""net".')

    if net._use_context_mod:
        raise NotImplementedError('This function doesn\'t handle context-mod ' +
                                  'layers yet!')

    assert net.param_shapes_meta is not None

    has_bn = net.batchnorm_layers is not None and len(net.batchnorm_layers) > 0
    has_conv_bias = net._use_bias
    has_fc_bias = net._use_fc_bias
    n = net._n
    filter_sizes = net._filter_sizes
    #n_conv_layers = 1 + 6 * n + np.sum(net._group_has_1x1)

    ### Group parameter shapes accoding to their meaning ###
    bn_shapes = None
    if has_bn:
        bn_shapes = net.param_shapes[:2*len(net.batchnorm_layers)]
        assert len(net.batchnorm_layers) == 6 * n + 1
        for i, meta in enumerate(net.param_shapes_meta[:len(bn_shapes)]):
            
            assert meta['name'].startswith('bn_')
            if i % 2 == 1:
                assert meta['layer'] == net.param_shapes_meta[i-1]['layer']
            elif i > 1:
                assert meta['layer'] > net.param_shapes_meta[i-2]['layer']

    conv_1x1_shapes = []
    pind = 0 if bn_shapes is None else len(bn_shapes)
    for g_has_1x1 in net._group_has_1x1:
        if g_has_1x1:
            conv_1x1_shapes.append(net.param_shapes[pind])
            pind += 1

            assert len(conv_1x1_shapes[-1]) == 4 and \
                conv_1x1_shapes[-1][-1] == 1
        else:
            conv_1x1_shapes.append(None)

    conv_layers = []
    conv_biases = [] if has_conv_bias else None
    for i in range(2*(1+6*n) if has_conv_bias else 1+6*n):
        shape = net.param_shapes[pind]
        meta = net.param_shapes_meta[pind]

        if has_conv_bias and i % 2 == 1:
            assert meta['name'] == 'bias'
            conv_biases.append(shape)
        else:
            assert meta['name'] == 'weight'
            conv_layers.append(shape)

        pind += 1

    assert pind == len(net.param_shapes) - (2 if has_fc_bias else 2)

    assert net.has_fc_out and net.mask_fc_out
    if has_fc_bias:
        fc_w_shape = net.param_shapes[-2]
        fc_b_shape = net.param_shapes[-1]
    else:
        fc_w_shape = net.param_shapes[-1]
        fc_b_shape = None

    ### Decide on chunking strategy ###
    use_lg_135 = [True, True, True] # Use layer group 1, 3 or 5?
    conv_group_gcd = [-1, -1, -1]
    for i in range(1, 4):
        fs_prev = filter_sizes[i-1]
        fs_curr = filter_sizes[i]

        # In this case, we always chunk.
        if max(fs_prev, fs_curr) / min(fs_prev, fs_curr) in [1, 2]:
            use_lg_135[i-1] = False
            conv_group_gcd[i-1] = min(fs_prev, fs_curr)
        elif gcd_chunking:
            use_lg_135[i-1] = False
            conv_group_gcd[i-1] = math.gcd(fs_prev, fs_curr)

    ### Prepare chunking for each layer group ###
    layer_groups = [True] * 11
    # Which layer group actually exist?
    if not use_lg_135[0]:
        layer_groups[1] = False
    if not use_lg_135[1]:
        layer_groups[3] = False
    if not use_lg_135[2]:
        layer_groups[5] = False
    # 7, 8, 9 are the 1x1 layer groups.
    for i, val in enumerate(net._group_has_1x1):
        if not val:
            layer_groups[7+i] = False
    if ignore_out_weights:
        layer_groups[-1] = False

    chunk_shapes = []
    num_per_chunk = []

    # Layer group 0.
    num_per_chunk.append(1)
    chunk_shapes.append([])
    chunk_shapes[-1].append(conv_layers[0])
    if has_conv_bias:
        chunk_shapes[-1].append(conv_biases[0])
    if not ignore_bn_weights and has_bn:
        chunk_shapes[-1].extend(bn_shapes[:2])

    # Layer groups 1 to 6.
    for g in range(3): # For each conv group.
        # Input layer to convolutional group.
        if layer_groups[1+2*g]:
            num_per_chunk.append(1)
            chunk_shapes.append([])
            chunk_shapes[-1].append(conv_layers[1+2*n*g])
            if has_conv_bias:
                chunk_shapes[-1].append(conv_biases[1+2*n*g])
            if not ignore_bn_weights and has_bn:
                chunk_shapes[-1].extend(bn_shapes[2*(1+2*n*g):2*(1+2*n*g)+2])

        # Remaining layers of convolutional group.
        fs_prev = filter_sizes[g]
        fs_curr = filter_sizes[g+1]

        assert not has_conv_bias or np.all(np.equal([a[0] for a in \
            conv_biases[1+2*n*g:1+2*n*(g+1)]], fs_curr))
        assert not has_bn or np.all(np.equal([a[0] for a in \
            bn_shapes[2*(1+2*n*g):2*(1+2*n*(g+1))]], fs_curr))

        if layer_groups[1+2*g]:
            num_per_chunk.append(2*n-1) # 1 chunk per conv layer.

            chunk_shapes.append([])
            chunk_shapes[-1].append(conv_layers[1+2*n*g+1])
        else:
            gcd = conv_group_gcd[g]
            num_per_chunk.append(fs_prev//gcd + (2*n-1) * fs_curr//gcd)

            chunk_shapes.append([[fs_curr, gcd, 3, 3]])

        if has_conv_bias:
            chunk_shapes[-1].append([fs_curr])
        if not ignore_bn_weights and has_bn:
            chunk_shapes[-1].extend([[fs_curr], [fs_curr]])

    # Layer group 7 - 9.
    for i in range(7, 10):
        if layer_groups[i]:
            num_per_chunk.append(1)
            chunk_shapes.append([conv_1x1_shapes[i-7]])

    # Layer group 10.
    if not ignore_out_weights:
        num_per_chunk.append(1)
        chunk_shapes.append([])
        chunk_shapes[-1].append(fc_w_shape)
        if has_fc_bias:
            chunk_shapes[-1].append(fc_b_shape)

    ### Get assembly function ###
    assembly_fct = lambda x : _wrn_chunking_afct(x, chunk_shapes, num_per_chunk,
        layer_groups, conv_group_gcd, has_conv_bias, has_fc_bias, has_bn,
        ignore_bn_weights, ignore_out_weights, n, filter_sizes)

    return chunk_shapes, num_per_chunk, assembly_fct

def _wrn_chunking_afct(list_of_chunks, chunk_shapes, num_per_chunk,
                       layer_groups, conv_group_gcd, has_conv_bias, has_fc_bias,
                       has_bn, ignore_bn_weights, ignore_out_weights, n,
                       filter_sizes):
    """The ``assembly_fct`` function required by function :func:`wrn_chunking`.
    """
    assert len(list_of_chunks) == np.sum(num_per_chunk)

    bn_weights = []
    conv_layer_weights = []
    res_1x1_layer_weights = []
    last_layer_weights = []

    cind = 0

    ### First layer ###
    conv_layer_weights.append(list_of_chunks[cind][0])
    if has_conv_bias:
        conv_layer_weights.append(list_of_chunks[cind][1])
    if not ignore_bn_weights and has_bn:
        bn_weights.extend(list_of_chunks[cind][-2:])

    cind += 1

    ### Resnet blocks ###
    for g in range(3): # For each block.
        # First layer in block.
        if layer_groups[1+2*g]:
            conv_layer_weights.append(list_of_chunks[cind][0])
            if has_conv_bias:
                conv_layer_weights.append(list_of_chunks[cind][1])
            if not ignore_bn_weights and has_bn:
                bn_weights.extend(list_of_chunks[cind][-2:])
            cind += 1

        # Remaining layers in block.
        fs_prev = filter_sizes[g]
        fs_curr = filter_sizes[g+1]

        if layer_groups[1+2*g]:
            for _ in range(2*n-1): # 1 chunk per layer
                conv_layer_weights.append(list_of_chunks[cind][0])
                if has_conv_bias:
                    conv_layer_weights.append(list_of_chunks[cind][1])
                if not ignore_bn_weights and has_bn:
                    bn_weights.extend(list_of_chunks[cind][-2:])
                cind += 1
        else:
            num_chunks_first = fs_prev // conv_group_gcd[g]
            num_chunks_rem = fs_curr // conv_group_gcd[g]

            # Important: Bias and batchnorm weights are always taken from the
            # first chunk of a layer (corresponding weights in remaining layers
            # are ignored). Weight tensors are concatenated across chunks.
            n_per_l = [num_chunks_first] + [num_chunks_rem] * (2*n-1)

            for n_c in n_per_l:
                chunks = list_of_chunks[cind:cind+n_c]
                cind += n_c

                conv_layer_weights.append(torch.cat([c[0] for c in chunks],
                                                    dim=1))
                if has_conv_bias:
                    conv_layer_weights.append(chunks[0][1])
                if not ignore_bn_weights and has_bn:
                    bn_weights.append(chunks[0][-2])
                    bn_weights.append(chunks[0][-1])

    ### 1x1 residual connections ###
    for i in range(3):
        if layer_groups[7+i]:
            res_1x1_layer_weights.append(list_of_chunks[cind][0])
            cind += 1

    ### Last layer ###
    # No batchnorm for last layer!
    if not ignore_out_weights:
        last_layer_weights.append(list_of_chunks[-1][0])
        if has_fc_bias:
            last_layer_weights.append(list_of_chunks[-1][1])

    return bn_weights + res_1x1_layer_weights + conv_layer_weights + \
        last_layer_weights

if __name__ == '__main__':
    pass
