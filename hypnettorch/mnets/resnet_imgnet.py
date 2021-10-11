#!/usr/bin/env python3
# Copyright 2021 Christian Henning
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
# @title          :mnets/resnet_imgnet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :05/04/2021
# @version        :1.0
# @python_version :3.8.8
"""
ResNet for ImageNet
-------------------

This module implements the class of Resnet networks described Table 1 of the
following paper:

    "Deep Residual Learning for Image Recognition", He et al., 2015
    https://arxiv.org/abs/1512.03385

Those networks are designed for inputs of size 224 x 224. In contrast, the
Resnet family implemented by class :class:`mnets.resnet.ResNet` is primarily
designed for CIFAR like inputs of size 32 x 32.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn

from hypnettorch.mnets.classifier_interface import Classifier
from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.mnets.wide_resnet import WRN
from hypnettorch.utils.torch_utils import init_params

class ResNetIN(Classifier):
    """Hypernet-compatible Resnets for ImageNet.

    The architecture of those Resnets is summarized in Table 4 of
    `He et al. <https://arxiv.org/abs/1512.03385>`__. They consist of 5 groups
    of convolutional layers, where the first group only has 1 convolutional
    layer followed by a max-pooling operation. The other 4 groups consist of
    blocks (see ``blocks_per_group``) of either 2 or 3 (see
    ``bottleneck_blocks``) convolutional layers per block. The network then
    computes its output via a final average pooling operation and a fully-
    connected layer.

    The number of layer per network is therewith
    ``1 + sum(blocks_per_group) * 2 + 1``, i.e., initial conv layer, num. conv
    layers in all blocks (assuming ``bottleneck_blocks=False``) and the final
    fully-connected layer. If ``projection_shortcut=True``, additional 1x1
    conv layers are added for shortcuts where the feature maps tensor shape
    changes.

    Here are a few implementation details worth noting:
      - If ``use_batch_norm=True``, it would be redundant to add convolutional
        layers to the conv layers, therefore one should set
        ``use_bias=False, use_fc_bias=True``. Skip connections never use biases.
      - Online implementations vary in their use of projection or identity
        shortcuts. We offer both possibilities (``projection_shortcut``). If
        ``projection_shortcut`` is used, then a batchnorm layer is added after
        each projection.

    Here are parameter configurations that can be used to obtain well-known
    Resnets (all configurations should use
    ``use_bias=False, use_fc_bias=True``):

      - `Resnet-18`: ``blocks_per_group=(2,2,2,2), bottleneck_blocks=False``
      - `Resnet-34`: ``blocks_per_group=(3,4,6,3), bottleneck_blocks=False``
      - `Resnet-50`: ``blocks_per_group=(3,4,6,3), bottleneck_blocks=True``
      - `Resnet-101`: ``blocks_per_group=(3,4,23,3), bottleneck_blocks=True``
      - `Resnet-152`: ``blocks_per_group=(3,4,36,3), bottleneck_blocks=True``

    Args:
        (....): See arguments of class:`mnets.wrn.WRN`.
        num_feature_maps (tuple):  A list of 5 integers, each denoting the
            number of feature maps in a group of convolutional layers.
            
            Note:
                If ``bottleneck_blocks=True``, then the last 1x1 conv layer in
                each block has 4 times as many feature maps as specified by this
                argument.
        blocks_per_group (tuple): A list of 4 integers, each denoting the
            number of convolutional blocks for the groups of convolutional
            layers.
        projection_shortcut (bool): If ``True``, skip connections that otherwise
            would require zero-padding or subsampling will be realized via 1x1
            conv layers followed by batchnorm. All other skip connections will
            be realized via identity mappings.
        bottleneck_blocks (bool): Whether normal blocks or bottleneck blocks
            should be used (cf. Fig. 5 in
            `He et al. <https://arxiv.org/abs/1512.03385>`__)
        cutout_mod (bool): Sometimes, networks from this family are used for
            smaller (CIFAR-like) images. In this case, one has to either
            upscale the images or adapt the architecture slightly (otherwise,
            small images are too agressively downscaled at the very beginning).

            When activating this option, the first conv layer is modified as
            described `here <https://github.com/uoguelph-mlrg/Cutout/blob/\
287f934ea5fa00d4345c2cccecf3552e2b1c33e3/model/resnet.py#L66>`__, i.e., it uses
            a kernel size of ``3`` with stride ``1`` and the max-pooling layer
            is omitted.

            Note, in order to recover the same architecture as in the link
            above one has to additionally set:
            ``use_bias=False, use_fc_bias=True, projection_shortcut=True``.
    """
    def __init__(self, in_shape=(224, 224, 3), num_classes=1000, use_bias=True,
                 use_fc_bias=None, num_feature_maps=(64, 64, 128, 256, 512),
                 blocks_per_group=(2,2,2,2), projection_shortcut=False,
                 bottleneck_blocks=False, cutout_mod=False, no_weights=False,
                 use_batch_norm=True, bn_track_stats=True,
                 distill_bn_stats=False, chw_input_format=False, verbose=True,
                 **kwargs):
        super(ResNetIN, self).__init__(num_classes, verbose)

        ### Parse or set context-mod arguments ###
        rem_kwargs = MainNetInterface._parse_context_mod_args(kwargs)
        if 'context_mod_apply_pixel_wise' in rem_kwargs:
            rem_kwargs.remove('context_mod_apply_pixel_wise')
        if len(rem_kwargs) > 0:
            raise ValueError('Keyword arguments %s unknown.' % str(rem_kwargs))
        # Since this is a conv-net, we may also want to add the following.
        if 'context_mod_apply_pixel_wise' not in kwargs.keys():
            kwargs['context_mod_apply_pixel_wise'] = False

        self._use_context_mod = kwargs['use_context_mod']
        self._context_mod_inputs = kwargs['context_mod_inputs']
        self._no_last_layer_context_mod = kwargs['no_last_layer_context_mod']
        self._context_mod_no_weights = kwargs['context_mod_no_weights']
        self._context_mod_post_activation = \
            kwargs['context_mod_post_activation']
        self._context_mod_gain_offset = kwargs['context_mod_gain_offset']
        self._context_mod_gain_softplus = kwargs['context_mod_gain_softplus']
        self._context_mod_apply_pixel_wise = \
            kwargs['context_mod_apply_pixel_wise']

        ### Check or parse remaining arguments ###
        self._in_shape = in_shape
        self._projection_shortcut = projection_shortcut
        self._bottleneck_blocks = bottleneck_blocks
        self._cutout_mod = cutout_mod
        if use_fc_bias is None:
            use_fc_bias = use_bias
        # Also, checkout attribute `_has_bias` below.
        self._use_bias = use_bias
        self._use_fc_bias = use_fc_bias
        self._no_weights = no_weights
        assert not use_batch_norm or (not distill_bn_stats or bn_track_stats)
        self._use_batch_norm = use_batch_norm
        self._bn_track_stats = bn_track_stats
        self._distill_bn_stats = distill_bn_stats and use_batch_norm
        self._chw_input_format = chw_input_format

        if len(blocks_per_group) != 4:
            raise ValueError('Option "blocks_per_group" must be a list of 4 ' +
                             'integers.')
        self._num_blocks = blocks_per_group
        if len(num_feature_maps) != 5:
            raise ValueError('Option "num_feature_maps" must be a list of 5 ' +
                             'integers.')
        self._filter_sizes = list(num_feature_maps)

        # The first layer of group 3, 4 and 5 uses a strided convolution, so
        # the shorcut connections need to perform a downsampling operation. In
        # addition, whenever traversing from one group to the next, the number
        # of feature maps might change. In all these cases, the network might
        # benefit from smart shortcut connections, which means using projection 
        # shortcuts, where a 1x1 conv is used for the mentioned skip connection.
        self._num_non_ident_skips = 3 # Strided convs: 2->3, 3->4 and 4->5
        fs1 = self._filter_sizes[1]
        if self._bottleneck_blocks:
            fs1 *= 4
        if self._filter_sizes[0] != fs1:
            self._num_non_ident_skips += 1 # Also handle 1->2.
        self._group_has_1x1 = [False] * 4
        if self._projection_shortcut:
            for i in range(3, 3-self._num_non_ident_skips, -1):
                self._group_has_1x1[i] = True
        # Number of conv layers (excluding skip connections)
        self._num_main_conv_layers = 1 + int(np.sum([self._num_blocks[i] * \
            (3 if self._bottleneck_blocks else 2) for i in range(4)]))

        # The original architecture uses a 7x7 kernel in the first conv layer
        # and 3x3 or 1x1 kernels in all remaining layers.
        self._init_kernel_size = (7, 7)
        # All 3x3 layers have padding 1 and 1x1 layers have padding 0.
        self._init_padding = 3
        self._init_stride = 2

        if self._cutout_mod:
            self._init_kernel_size = (3, 3)
            self._init_padding = 1
            self._init_stride = 1

        ### Set required class attributes ###
        # Note, we did overwrite the getter for attribute `has_bias`, as it is
        # not applicable if the values of `use_bias` and `use_fc_bias` differ.
        self._has_bias = use_bias if use_bias == use_fc_bias else False
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer!
        self._mask_fc_out = True
        self._has_linear_out = True

        self._param_shapes = []
        self._param_shapes_meta = []
        self._internal_params = None if no_weights and \
            self._context_mod_no_weights else nn.ParameterList()
        self._hyper_shapes_learned = None \
            if not no_weights and not self._context_mod_no_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        #################################
        ### Create context mod layers ###
        #################################
        self._context_mod_layers = nn.ModuleList() if self._use_context_mod \
            else None

        if self._use_context_mod:
            cm_layer_inds = []
            cm_shapes = [] # Output shape of all layers.
            if self._context_mod_inputs:
                cm_shapes.append([in_shape[2], *in_shape[:2]])
                # We reserve layer zero for input context-mod. Otherwise, there
                # is no layer zero.
                cm_layer_inds.append(0)

            layer_out_shapes = self._compute_layer_out_sizes()
            cm_shapes.extend(layer_out_shapes)
            # All layer indices `l` with `l mod 3 == 0` are context-mod layers.
            cm_layer_inds.extend(range(3, 3*len(layer_out_shapes)+1, 3))
            if self._no_last_layer_context_mod:
                cm_shapes = cm_shapes[:-1]
                cm_layer_inds = cm_layer_inds[:-1]
            if not self._context_mod_apply_pixel_wise:
                # Only scalar gain and shift per feature map!
                for i, s in enumerate(cm_shapes):
                    if len(s) == 3:
                        cm_shapes[i] = [s[0], 1, 1]

            self._add_context_mod_layers(cm_shapes, cm_layers=cm_layer_inds)

        ###############################
        ### Create batchnorm layers ###
        ###############################
        # We just use even numbers starting from 2 as layer indices for
        # batchnorm layers.
        if use_batch_norm:
            bn_sizes = []
            for i, s in enumerate(self._filter_sizes):
                if i == 0:
                    bn_sizes.append(s)
                else:
                    for _ in range( self._num_blocks[i-1]):
                        if self._bottleneck_blocks:
                            bn_sizes.extend([s, s, 4*s])
                        else:
                            bn_sizes.extend([s, s])

            # All layer indices `l` with `l mod 3 == 2` are batchnorm layers.
            bn_layers=list(range(2, 3*len(bn_sizes)+1, 3))

            # We also need a batchnorm layer per skip connection that uses 1x1
            # projections.
            if self._projection_shortcut:
                bn_layer_ind_skip = 3 * (self._num_main_conv_layers+1) + 2

                factor = 4 if self._bottleneck_blocks else 1
                for i in range(4): # For each transition between conv groups.
                    if self._group_has_1x1[i]:
                        bn_sizes.append(self._filter_sizes[i+1] * factor)
                        bn_layers.append(bn_layer_ind_skip)

                        bn_layer_ind_skip += 3

            self._add_batchnorm_layers(bn_sizes, no_weights,
                bn_layers=bn_layers, distill_bn_stats=distill_bn_stats,
                bn_track_stats=bn_track_stats)

        ######################################
        ### Create skip connection weights ###
        ######################################
        if self._projection_shortcut:
            layer_ind_skip = 3 * (self._num_main_conv_layers+1) + 1

            factor = 4 if self._bottleneck_blocks else 1

            n_in = self._filter_sizes[0]
            for i in range(4): # For each transition between conv groups.
                if not self._group_has_1x1[i]:
                    continue

                n_out = self._filter_sizes[i+1] * factor

                skip_1x1_shape = [n_out, n_in, 1, 1]

                if not no_weights:
                    self._internal_params.append(nn.Parameter( \
                        torch.Tensor(*skip_1x1_shape), requires_grad=True))
                    self._layer_weight_tensors.append(self._internal_params[-1])
                    self._layer_bias_vectors.append(None)
                    init_params(self._layer_weight_tensors[-1])
                else:
                    self._hyper_shapes_learned.append(skip_1x1_shape)
                    self._hyper_shapes_learned_ref.append( \
                        len(self.param_shapes))

                self._param_shapes.append(skip_1x1_shape)
                self._param_shapes_meta.append({
                    'name': 'weight',
                    'index': -1 if no_weights else \
                        len(self._internal_params)-1,
                    'layer': layer_ind_skip
                })

                layer_ind_skip += 3
                n_in = n_out

        ############################################################
        ### Create convolutional layers and final linear weights ###
        ############################################################
        # Convolutional layers will get IDs `l` such that `l mod 3 == 1`.
        layer_id = 1
        n_per_block = 3 if self._bottleneck_blocks else 2
        for i in range(6):
            if i == 0: ### Fist layer.
                num = 1
                prev_fs = self._in_shape[2]
                curr_fs = self._filter_sizes[0]

                kernel_size = self._init_kernel_size
                #stride = self._init_stride
            elif i == 5: ### Final fully-connected layer.
                num = 1
                curr_fs = num_classes

                kernel_size = None
            else: # Group of residual blocks.
                num = self._num_blocks[i-1] * n_per_block
                curr_fs = self._filter_sizes[i]

                kernel_size = (3, 3) # depends on block structure!

            for n in range(num):
                if i == 5:
                    layer_shapes = [[curr_fs, prev_fs]]
                    if use_fc_bias:
                        layer_shapes.append([curr_fs])

                    prev_fs = curr_fs
                else:
                    if i > 0 and self._bottleneck_blocks:
                        if n % 3 == 0:
                            fs = curr_fs
                            ks = (1, 1)
                        elif n % 3 == 1:
                            fs = curr_fs
                            ks = kernel_size
                        else:
                            fs = 4 * curr_fs
                            ks = (1, 1)
                    elif i > 0 and not self._bottleneck_blocks:
                        fs = curr_fs
                        ks = kernel_size
                    else:
                        fs = curr_fs
                        ks = kernel_size

                    layer_shapes = [[fs, prev_fs, *ks]]
                    if use_bias:
                        layer_shapes.append([fs])

                    prev_fs = fs

                for s in layer_shapes:
                    if not no_weights:
                        self._internal_params.append(nn.Parameter( \
                            torch.Tensor(*s), requires_grad=True))
                        if len(s) == 1:
                            self._layer_bias_vectors.append( \
                                self._internal_params[-1])
                        else:
                            self._layer_weight_tensors.append( \
                                self._internal_params[-1])
                    else:
                        self._hyper_shapes_learned.append(s)
                        self._hyper_shapes_learned_ref.append( \
                            len(self.param_shapes))

                    self._param_shapes.append(s)
                    self._param_shapes_meta.append({
                        'name': 'weight' if len(s) != 1 else 'bias',
                        'index': -1 if no_weights else \
                            len(self._internal_params)-1,
                        'layer': layer_id
                    })

                layer_id += 3

                # Initialize_weights
                if not no_weights:
                    init_params(self._layer_weight_tensors[-1],
                        self._layer_bias_vectors[-1] \
                        if len(layer_shapes) == 2 else None)

        ###########################
        ### Print infos to user ###
        ###########################
        if verbose:
            if self._use_context_mod:
                cm_param_shapes = []
                for cm_layer in self.context_mod_layers:
                    cm_param_shapes.extend(cm_layer.param_shapes)
                cm_num_params = \
                    MainNetInterface.shapes_to_num_weights(cm_param_shapes)

            print('Creating a "%s" with %d weights' \
                  % (str(self), self.num_params)
                  + (' (including %d weights associated with-' % cm_num_params
                     + 'context modulation)' if self._use_context_mod else '')
                  + '.'
                  + (' The network uses batchnorm.' if use_batch_norm  else ''))

        self._is_properly_setup(check_has_bias=False)

    def __str__(self):
        return 'ResNet-%d' % (self._num_main_conv_layers+1)

    @property
    def has_bias(self):
        """Getter for read-only attribute :attr:`has_bias`."""
        if self._use_bias != self._use_fc_bias:
            raise RuntimeError('Attribute "has_bias" does not apply to a ' +
                               'network where there is a mixture of layers ' +
                               'with and without biases.')
        if self._use_bias and self._projection_shortcut:
            warn('The network contains skip connections which are realized ' +
                 'via 1x1 convolutional layers without biases. The attribute ' +
                 '"has_bias" ignores these layers. It\'s best to avoid using ' +
                 'this attribute for this reason.')
        return self._has_bias

    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.resnet.ResNet.forward`. We provide some more
                specific information below.
            x (torch.Tensor): Based on the constructor argument
                ``chw_input_format``, either a flattened image batch with
                encoding ``HWC`` or an unflattened image batch with encoding
                ``CHW`` is expected.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
        # FIXME code mostly copied from MLP forward method.
        n_cm = self._num_context_mod_shapes()

        if weights is None:
            weights = self.weights

            if self._use_context_mod:
                cm_weights = weights[:n_cm]
                int_weights = weights[n_cm:]
            else:
                cm_weights = None
                int_weights = weights
        else:
            int_weights = None
            cm_weights = None

            if isinstance(weights, dict):
                assert('internal_weights' in weights.keys() or \
                       'mod_weights' in weights.keys())
                if 'internal_weights' in weights.keys():
                    int_weights = weights['internal_weights']
                if 'mod_weights' in weights.keys():
                    cm_weights = weights['mod_weights']
            else:
                if self._use_context_mod and \
                        len(weights) == n_cm:
                    cm_weights = weights
                else:
                    assert len(weights) == len(self.param_shapes)
                    if self._use_context_mod:
                        cm_weights = weights[:n_cm]
                        int_weights = weights[n_cm:]
                    else:
                        int_weights = weights

            if self._use_context_mod and cm_weights is None:
                if self._context_mod_no_weights:
                    raise Exception('Network was generated without weights ' +
                        'for context-mod layers. Hence, they must be passed ' +
                        'via the "weights" option.')
                cm_weights = self.weights[:n_cm]
            if int_weights is None:
                if self._no_weights:
                    raise Exception('Network was generated without internal ' +
                        'weights. Hence, they must be passed via the ' +
                        '"weights" option.')
                if self._context_mod_no_weights:
                    int_weights = self.weights
                else:
                    int_weights = self.weights[n_cm:]

            # Note, context-mod weights might have different shapes, as they
            # may be parametrized on a per-sample basis.
            if self._use_context_mod:
                assert len(cm_weights) == self._num_context_mod_shapes()
            int_shapes = self.param_shapes[n_cm:]
            assert len(int_weights) == len(int_shapes)
            for i, s in enumerate(int_shapes):
                assert np.all(np.equal(s, list(int_weights[i].shape)))

        ### Split context-mod weights per context-mod layer.
        if cm_weights is not None:
            cm_weights_layer = []
            cm_start = 0
            for cm_layer in self.context_mod_layers:
                cm_end = cm_start + len(cm_layer.param_shapes)
                cm_weights_layer.append(cm_weights[cm_start:cm_end])
                cm_start = cm_end

        int_meta = self.param_shapes_meta[n_cm:]
        int_weights = list(int_weights)
        ### Split batchnorm weights layer-wise.
        if self._use_batch_norm:
            lbw = 2 * len(self.batchnorm_layers)

            bn_weights = int_weights[:lbw]
            int_weights = int_weights[lbw:]
            bn_meta = int_meta[:lbw]
            int_meta = int_meta[lbw:]

            bn_scales = []
            bn_shifts = []

            for i in range(len(self.batchnorm_layers)):
                assert bn_meta[2*i]['name'] == 'bn_scale'
                bn_scales.append(bn_weights[2*i])
                assert bn_meta[2*i+1]['name'] == 'bn_shift'
                bn_shifts.append(bn_weights[2*i+1])

        ### Split internal weights layer-wise.
        # Weights of skip connections.
        n_skip_1x1 = np.sum(self._group_has_1x1)
        skip_1x1_weights = [None] * 4
        for i in range(4):
            if self._group_has_1x1[i]:
                skip_1x1_weights[i] = int_weights.pop(0)
        int_meta = int_meta[n_skip_1x1:]

        # Weights/biases per layer.
        layer_weights = [None] * (self._num_main_conv_layers + 1)
        layer_biases = [None] * (self._num_main_conv_layers + 1)

        for i, meta in enumerate(int_meta):
            ltype = meta['name']
            # Recals, layer IDs for this type of layer are `l mod 3 == 1`.
            lid = (meta['layer'] - 1) // 3
            if ltype == 'weight':
                layer_weights[lid] = int_weights[i]
            else:
                assert ltype == 'bias'
                layer_biases[lid] = int_weights[i]

        #######################
        ### Parse condition ###
        #######################
        bn_cond = None
        cmod_cond = None

        if condition is not None:
            if isinstance(condition, dict):
                assert 'bn_stats_id' in condition.keys() or \
                       'cmod_ckpt_id' in condition.keys()
                if 'bn_stats_id' in condition.keys():
                    bn_cond = condition['bn_stats_id']
                if 'cmod_ckpt_id' in condition.keys():
                    cmod_cond = condition['cmod_ckpt_id']
            else:
                bn_cond = condition

        if cmod_cond is not None:
            # FIXME We always require context-mod weight above, but
            # we can't pass both (a condition and weights) to the
            # context-mod layers.
            # An unelegant solution would be, to just set all
            # context-mod weights to None.
            raise NotImplementedError('CM-conditions not implemented!')
            cm_weights_layer = [None] * len(cm_weights_layer)

        ######################################
        ### Select batchnorm running stats ###
        ######################################
        if self._use_batch_norm:
            nn = len(self._batchnorm_layers)
            running_means = [None] * nn
            running_vars = [None] * nn

        if distilled_params is not None:
            if not self._distill_bn_stats:
                raise ValueError('Argument "distilled_params" can only be ' +
                                 'provided if the return value of ' +
                                 'method "distillation_targets()" is not None.')
            shapes = self.hyper_shapes_distilled
            assert len(distilled_params) == len(shapes)
            for i, s in enumerate(shapes):
                assert np.all(np.equal(s, list(distilled_params[i].shape)))

            # Extract batchnorm stats from distilled_params
            for i in range(0, len(distilled_params), 2):
                running_means[i//2] = distilled_params[i]
                running_vars[i//2] = distilled_params[i+1]

        elif self._use_batch_norm and self._bn_track_stats and \
                bn_cond is None:
            for i, bn_layer in enumerate(self._batchnorm_layers):
                running_means[i], running_vars[i] = bn_layer.get_stats()

        ###########################
        ### Forward Computation ###
        ###########################
        cm_ind = 0
        bn_ind = 0
        layer_ind = 0

        ### Helper function to process convolutional layers.
        def conv_layer(h, stride, padding=1, shortcut=None, no_conv=False):
            """Compute the output of a full conv layer within a residual block
            including batchnorm, context-mod, non-linearity and shortcut.

            The order if the following:

            conv-layer -> context-mod (if pre-activation) -> batch-norm ->
            shortcut -> non-linearity -> context-mod (if post-activation)

            This method increments the indices ``layer_ind``, ``cm_ind`` and
            ``bn_ind``.

            Args:
                h: Input activity.
                stride: Stride of conv. layer.
                padding (int): The padding of the conv-layer.
                shortcut: If set, this tensor will be added to the activation
                    before the non-linearity is applied.
                no_conv: If True, no convolutional layer is applied.

            Returns:
                Output of layer.
            """
            nonlocal layer_ind, cm_ind, bn_ind

            if not no_conv:
                h = F.conv2d(h, layer_weights[layer_ind],
                    bias=layer_biases[layer_ind], stride=stride,
                    padding=padding)
                layer_ind += 1

            # Context-dependent modulation (pre-activation).
            if self._use_context_mod and \
                    not self._context_mod_post_activation:
                h = self._context_mod_layers[cm_ind].forward(h,
                    weights=cm_weights_layer[cm_ind], ckpt_id=cmod_cond)
                cm_ind += 1

            # Batch-norm
            if self._use_batch_norm:
                h = self._batchnorm_layers[bn_ind].forward(h,
                    running_mean=running_means[bn_ind],
                    running_var=running_vars[bn_ind],
                    weight=bn_scales[bn_ind], bias=bn_shifts[bn_ind],
                    stats_id=bn_cond)
                bn_ind += 1

            # Note, as can be seen in figure 5 of the original paper, the
            # shortcut is performed before the ReLU is applied.
            if shortcut is not None:
                h += shortcut

            # Non-linearity
            h = F.relu(h)

            # Context-dependent modulation (post-activation).
            if self._use_context_mod and self._context_mod_post_activation:
                h = self._context_mod_layers[cm_ind].forward(h,
                    weights=cm_weights_layer[cm_ind], ckpt_id=cmod_cond)
                cm_ind += 1

            return h

        if not self._chw_input_format:
            x = x.view(-1, *self._in_shape)
            x = x.permute(0, 3, 1, 2)
        h = x

        # Context-dependent modulation of inputs directly.
        if self._use_context_mod and self._context_mod_inputs:
            h = self._context_mod_layers[cm_ind].forward(h,
                weights=cm_weights_layer[cm_ind], ckpt_id=cmod_cond)
            cm_ind += 1

        ### Initial convolutional layer.
        h = conv_layer(h, self._init_stride, padding=self._init_padding,
                       shortcut=None)

        ### The max-pooling layer at the beginning of group conv2_x.
        if not self._cutout_mod:
            h = F.max_pool2d(h, kernel_size=(3, 3), stride=2, padding=1)

        ### 4 groups, each containing `num_blocks` resnet blocks.
        fs_prev = self._filter_sizes[0]
        for i in range(4):
            fs_curr = self._filter_sizes[i+1]
            if self._bottleneck_blocks:
                fs_curr *= 4

            # Only the first layer in a group may be a strided convolution, but
            # not for group conv2_x.
            stride = 2 if i > 0 else 1
            # For each resnet block . A resnet block consists of 2 or 3
            # convolutional layers.
            for j in range(self._num_blocks[i]):
                shortcut_h = h
                if j == 0 and (stride != 1 or fs_prev != fs_curr):
                    if self._projection_shortcut:
                        assert self._group_has_1x1[i]
                        shortcut_h = F.conv2d(h, skip_1x1_weights[i], bias=None,
                                              stride=stride, padding=0)
                        if self._use_batch_norm:
                            bn_short = len(self._batchnorm_layers) - 4 + i
                            shortcut_h = \
                                self._batchnorm_layers[bn_short].forward( \
                                    shortcut_h,
                                    running_mean=running_means[bn_short],
                                    running_var=running_vars[bn_short],
                                    weight=bn_scales[bn_short],
                                    bias=bn_shifts[bn_short], stats_id=bn_cond)
                    else:
                        # Use padding and subsampling.
                        pad_left = (fs_curr - fs_prev) // 2
                        pad_right = int(np.ceil((fs_curr - fs_prev) / 2))
                        if stride == 2:
                            shortcut_h = h[:, :, ::2, ::2]
                        shortcut_h = F.pad(shortcut_h,
                            (0, 0, 0, 0, pad_left, pad_right), "constant", 0)


                if self._bottleneck_blocks:
                    h = conv_layer(h, stride, padding=0, shortcut=None)
                    stride = 1
                    h = conv_layer(h, stride, padding=1, shortcut=None)
                    h = conv_layer(h, stride, padding=0, shortcut=shortcut_h)
                else:
                    h = conv_layer(h, stride, padding=1, shortcut=None)
                    stride = 1
                    h = conv_layer(h, stride, padding=1, shortcut=shortcut_h)

            fs_prev = fs_curr

        ### Average pool all activities within a feature map.
        h = F.avg_pool2d(h, [h.size()[2], h.size()[3]])
        h = h.view(h.size(0), -1)

        ### Apply final fully-connected layer and compute outputs.
        h = F.linear(h, layer_weights[layer_ind], bias=layer_biases[layer_ind])

        # Context-dependent modulation in output layer.
        if self._use_context_mod and not self._no_last_layer_context_mod:
            h = self._context_mod_layers[cm_ind].forward(h,
                weights=cm_weights[2*cm_ind:2*cm_ind+2], ckpt_id=cmod_cond)

        return h

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This method will return the current batch statistics of all batch
        normalization layers if ``distill_bn_stats`` and ``use_batch_norm``
        were set to ``True`` in the constructor.

        Returns:
            The target tensors corresponding to the shapes specified in
            attribute :attr:`hyper_shapes_distilled`.
        """
        if self.hyper_shapes_distilled is None:
            return None

        ret = []
        for bn_layer in self._batchnorm_layers:
            ret.extend(bn_layer.get_stats())

        return ret

    def _compute_layer_out_sizes(self):
        """Compute the output shapes of all layers in this network excluding
        skip connection layers.

        This method will compute the output shape of each layer in this network,
        including the output layer, which just corresponds to the number of
        classes.

        Returns:
            (list): A list of shapes (lists of integers). The first entry will
            correspond to the shape of the output of the first convolutional
            layer. The last entry will correspond to the output shape.

            .. note:
                Output shapes of convolutional layers will adhere PyTorch
                convention, i.e., ``[C, H, W]``, where ``C`` denotes the channel
                dimension.
        """
        in_shape = self._in_shape
        fs = self._filter_sizes
        init_ks = self._init_kernel_size
        stride_init = self._init_stride
        pd_init = self._init_padding

        # Note, `in_shape` is in Tensorflow layout.
        assert(len(in_shape) == 3)
        in_shape = [in_shape[2], *in_shape[:2]]

        ret = []

        C, H, W = in_shape

        # Recall the formular for convolutional layers:
        # W_new = (W - K + 2P) // S + 1

        # First conv layer.
        C = fs[0]
        H = (H - init_ks[0] + 2*pd_init) // stride_init + 1
        W = (W - init_ks[1] + 2*pd_init) // stride_init + 1
        ret.append([C, H, W])

        def add_block(H, W, C, stride):
            if self._bottleneck_blocks:
                H = (H - 1 + 2*0) // stride + 1
                W = (W - 1 + 2*0) // stride + 1
                ret.append([C, H, W])

                H = (H - 3 + 2*1) // 1 + 1
                W = (W - 3 + 2*1) // 1 + 1
                ret.append([C, H, W])

                C = 4 * C
                H = (H - 1 + 2*0) // 1 + 1
                W = (W - 1 + 2*0) // 1 + 1
                ret.append([C, H, W])

            else:
                H = (H - 3 + 2*1) // stride + 1
                W = (W - 3 + 2*1) // stride + 1
                ret.append([C, H, W])

                H = (H - 3 + 2*1) // 1 + 1
                W = (W - 3 + 2*1) // 1 + 1
                ret.append([C, H, W])

            return H, W, C

        # Group conv2_x
        if not self._cutout_mod: # Max-pooling layer.
            H = (H - 3 + 2*1) // 2 + 1
            W = (W - 3 + 2*1) // 2 + 1

        for b in range(self._num_blocks[0]):
            H, W, C = add_block(H, W, fs[1], 1)

        # Group conv3_x
        for b in range(self._num_blocks[1]):
            H, W, C = add_block(H, W, fs[2], 2 if b==0 else 1)
            
        # Group conv4_x
        for b in range(self._num_blocks[2]):
            H, W, C = add_block(H, W, fs[3], 2 if b==0 else 1)
            
        # Group conv5_x
        for b in range(self._num_blocks[3]):
            H, W, C = add_block(H, W, fs[4], 2 if b==0 else 1)

        # Final fully-connected layer (after avg pooling), i.e., output size.
        ret.append([self._num_classes])

        return ret

    def get_output_weight_mask(self, out_inds=None, device=None):
        """Create a mask for selecting weights connected solely to certain
        output units.

        See docstring of overwritten super method
        :meth:`mnets.mnet_interface.MainNetInterface.get_output_weight_mask`.
        """
        return WRN.get_output_weight_mask(self, out_inds=out_inds,
                                          device=device)

if __name__ == '__main__':
    pass


