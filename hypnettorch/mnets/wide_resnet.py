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
# @title          :mnets/wide_resnet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :05/05/2020
# @version        :1.0
# @python_version :3.6.10
"""
Wide-ResNet
-----------

The module :mod:`mnets.wide_resnet` implements the class of Wide Residual
Networks as described in:

    Zagoruyko et al.,
    `"Wide Residual Networks" <https://arxiv.org/abs/1605.07146>`__, 2017.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn

from hypnettorch.mnets.classifier_interface import Classifier
from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils.torch_utils import init_params

class WRN(Classifier):
    """Hypernet-compatible Wide Residual Network (WRN).

    In the documentation of this class, we follow the notation of the original
    `paper  <https://arxiv.org/abs/1605.07146>`__:

    - :math:`l` - deepening factor (number of convolutional layers per residual
      block). In our case, :math:`l` is always going to be 2, as this was the
      configuration found to work best by the authors.
    - :math:`k` - widening factor (multiplicative factor for the number of
      features in a convolutional layer, see argument ``k``).
    - :math:`B(3,3)` - the block structure. The numbers denote the size of the
      quadratic kernels used in each convolutional layer from a block. Note, the
      authors found that :math:`B(3,3)` works best, which is why we use this
      configuration.
    - :math:`d` - total number of convolutional layers. Note, here we deviate
      from the original notation (where this quantity is called :math:`n`).
      Though, we want our notation to stay consistent with the one used in class
      :class:`mnets.resnet.ResNet`.
    - :math:`n` - number of residual blocks in a group. Note, a resnet consists
      of 3 groups of residual blocks. See also argument ``n`` of class
      :class:`mnets.resnet.ResNet`.

    Given this notation, the original paper denotes a WRN architecture via the
    following notation: *WRN-d-k-B(3,3)*. Note, :math:`d` contains the total
    number of convolutional layers (including the input layer and all residual
    connections that are realized via 1x1 convolutions), but it does not contain
    the final fully-connected layer. The total depth of the network (assuming
    residual connection do not add to this depth) remains :math:`6n+2` as for
    :class:`mnets.resnet.ResNet`.

    Notable implementation differences to :class:`mnets.resnet.ResNet`
    (some differences might vanish in the future, this list was updated on
    05/06/2020):

    - Within a block, convolutional layers are preceeded by a batchnorm layer
      and the application of the nonlinearity. This changes the structure within
      a block and therefore, residual connections interface with the network at
      different locations than in class :class:`mnets.resnet.ResNet`.
    - Dropout can be used. It will act right after the first convolutional layer
      of each block.
    - If the number of feature maps differs along a skip connection or a
      downsampling has been applied, 1x1 convolutions rather than padding and
      manual downsampling is used.

    Args:
        in_shape (tuple or list): The shape of an input sample in format
            ``HWC``.

            Note
                We assume the Tensorflow format, where the last entry
                denotes the number of channels. Also, see argument
                ``chw_input_format``.
        num_classes (int): The number of output neurons.

            Note:
                The network outputs logits.
        n (int): The number of residual blocks per group.
        k (int): The widening factor. Feature maps in the 3 convolutional groups
            will be multiplied by this number. See argument
            ``num_feature_maps``.
        num_feature_maps (tuple): A list of 4 integers, each denoting the number
            of feature maps of convolutional layers in a certain group of the
            network architecture. The first entry is the number of feature
            maps of the first convolutional layer, the remaining 3 numbers
            determine the number of feature maps in the consecutive groups
            comprising :math:`2n` convolutional layers each.

            Note:
                The last 3 entries of this list are multiplied by the factor
                ``k``.
                use_bias (bool): Whether layers may have bias terms.
        use_bias (bool): Whether layers may have bias terms.

            Note:
                Bias terms are unnecessary in convolutional layers if batch
                normalization is used. However, this option disables bias terms
                altogether (including in the final fully-connected layer). See
                option ``use_fc_bias``.
        use_fc_bias (optional, bool): If ``None``, the value will be linked to
            ``use_bias``. Otherwise, this option can alter the usage of bias
            terms in the final layer compared to the remaining (convolutional)
            layers in the network.
        no_weights (bool): If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.

            Note, this also affects the affine parameters of the
            batchnorm layer. I.e., if set to ``True``, then the argument
            ``affine`` of :class:`utils.batchnorm_layer.BatchNormLayer`
            will be set to ``False`` and we expect the batchnorm parameters
            to be passed to the :meth:`forward`.
        use_batch_norm (bool): Whether batch normalization should used.
            There will be a batchnorm layer after each convolutional layyer
            (excluding possible 1x1 conv layers in the skip connections).
            However, the logical order is as follows: batchnorm layer -> ReLU ->
            convolutional layer. Hence, a residual block (containing multiple of
            these logical units) starts before a batchnorm layer and ends after
            a convolutional layer.
        bn_track_stats (bool): See argument ``bn_track_stats`` of class
            :class:`mnets.resnet.ResNet`.
        distill_bn_stats (bool): See argument ``bn_track_stats`` of class
            :class:`mnets.resnet.ResNet`.
        dropout_rate (float): If ``-1``, no dropout will be applied. Otherwise a
            number between 0 and 1 is expected, denoting the dropout rate.

            Dropout will be applied after the first convolutional layers
            (and before the second batchnorm layer) in each residual block.
        chw_input_format (bool): Due to legacy reasons, the network expects
            by default flattened images as input that were encoded in the
            ``HWC`` format. When enabling this option, the network expects
            unflattened images in the ``CHW`` format (as typical for PyTorch).
        verbose (bool): Allow printing of general information about the
            generated network (such as number of weights).
        **kwargs: Keyword arguments regarding context modulation. This class
            can process the same context-modulation related arguments as class
            :class:`mnets.mlp.MLP`. One may additionally specify the argument
            ``context_mod_apply_pixel_wise`` (see class
            :class:`mnets.resnet.ResNet`).
    """
    def __init__(self, in_shape=(32, 32, 3), num_classes=10, n=4, k=10,
                 num_feature_maps=(16, 16, 32, 64), use_bias=True,
                 use_fc_bias=None, no_weights=False, use_batch_norm=True,
                 bn_track_stats=True, distill_bn_stats=False, dropout_rate=-1,
                 chw_input_format=False, verbose=True, **kwargs):
        super(WRN, self).__init__(num_classes, verbose)

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
        self._n = n
        self._k = k
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
        self._dropout_rate = dropout_rate
        self._chw_input_format = chw_input_format

        # The original authors found that the best configuration uses this
        # kernel in all convolutional layers.
        self._kernel_size = (3, 3)
        if len(num_feature_maps) != 4:
            raise ValueError('Option "num_feature_maps" must be a list of 4 ' +
                             'integers.')
        self._filter_sizes = list(num_feature_maps)
        if k != 1:
            for i in range(1, 4):
                self._filter_sizes[i] = k * num_feature_maps[i]
        # Strides used in the first layer of each convolutional group.
        self._strides = (1, 1, 2, 2)

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

        if dropout_rate != -1:
            assert dropout_rate >= 0. and dropout_rate <= 1.
            self._dropout = nn.Dropout(p=dropout_rate)

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
                    bn_sizes.extend([s] * (2*n))

            # All layer indices `l` with `l mod 3 == 2` are batchnorm layers.
            self._add_batchnorm_layers(bn_sizes, no_weights,
                bn_layers=list(range(2, 3*len(bn_sizes)+1, 3)),
                distill_bn_stats=distill_bn_stats,
                bn_track_stats=bn_track_stats)

        ######################################
        ### Create skip connection weights ###
        ######################################
        # We use 1x1 convolutional layers for residual blocks in case the
        # number of input and output feature maps disagrees. We also use 1x1
        # convolutions whenever a stride greater than 1 is applied. This is not
        # necessary in my opinion (as it adds extra weights that do not affect
        # the downsampling itself), but commonly done; for instance, in the
        # original PyTorch implementation.
        # Note, there may be maximally 3 1x1 layers added to the network.
        # Note, we use 1x1 conv layers without biases.
        skip_1x1_shapes = []
        self._group_has_1x1 = [False] * 3
        for i in range(1, 4):
            if self._filter_sizes[i-1] != self._filter_sizes[i] or \
                    self._strides[i] != 1:
                skip_1x1_shapes.append([self._filter_sizes[i],
                                        self._filter_sizes[i-1], 1, 1])
                self._group_has_1x1[i-1] = True

        for s in skip_1x1_shapes:
            if not no_weights:
                self._internal_params.append(nn.Parameter( \
                    torch.Tensor(*s), requires_grad=True))
                self._layer_weight_tensors.append(self._internal_params[-1])
                self._layer_bias_vectors.append(None)
                init_params(self._layer_weight_tensors[-1])
            else:
                self._hyper_shapes_learned.append(s)
                self._hyper_shapes_learned_ref.append(len(self.param_shapes))

            self._param_shapes.append(s)
            self._param_shapes_meta.append({
                'name': 'weight',
                'index': -1 if no_weights else \
                    len(self._internal_params)-1,
                'layer': -1
            })

        ############################################################
        ### Create convolutional layers and final linear weights ###
        ############################################################
        # Convolutional layers will get IDs `l` such that `l mod 3 == 1`.
        layer_id = 1
        for i in range(5):
            if i == 0: ### Fist layer.
                num = 1
                prev_fs = self._in_shape[2]
                curr_fs = self._filter_sizes[0]
            elif i == 4: ### Final fully-connected layer.
                num = 1
                curr_fs = num_classes
            else: # Group of residual blocks.
                num = 2 * n
                curr_fs = self._filter_sizes[i]

            for _ in range(num):
                if i == 4:
                    layer_shapes = [[curr_fs, prev_fs]]
                    if use_fc_bias:
                        layer_shapes.append([curr_fs])
                else:
                    layer_shapes = [[curr_fs, prev_fs, *self._kernel_size]]
                    if use_bias:
                        layer_shapes.append([curr_fs])

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

                prev_fs = curr_fs
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

            print('Creating a WideResnet "%s" with %d weights' \
                  % (str(self), self.num_params)
                  + (' (including %d weights associated with-' % cm_num_params
                     + 'context modulation)' if self._use_context_mod else '')
                  + '.'
                  + (' The network uses batchnorm.' if use_batch_norm  else '')
                  + (' The network uses dropout.' if dropout_rate != -1 \
                     else ''))

        self._is_properly_setup(check_has_bias=False)

    @property
    def has_bias(self):
        """Getter for read-only attribute :attr:`has_bias`."""
        if self._use_bias != self._use_fc_bias:
            raise RuntimeError('Attribute "has_bias" does not apply to a ' +
                               'network where there is a mixture of layers ' +
                               'with and without biases.')
        if self._use_bias and np.any(self._group_has_1x1):
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
        skip_1x1_weights = [None] * 3
        for i in range(3):
            if self._group_has_1x1[i]:
                skip_1x1_weights[i] = int_weights.pop(0)
        int_meta = int_meta[n_skip_1x1:]

        # Weights/biases per layer.
        layer_weights = [None] * (6 * self._n + 2)
        layer_biases = [None] * (6 * self._n + 2)

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
        def conv_layer(h, stride, shortcut=None, no_conv=False):
            """Compute the output of a full conv layer within a residual block
            including batchnorm, context-mod, non-linearity and shortcut.

            The order if the following:

            context-mod (if pre-activation) -> batch-norm -> non-linearity ->
            context-mod (if post-activation) -> conv-layer -> shortcut

            This method increments the indices ``layer_ind``, ``cm_ind`` and
            ``bn_ind``.

            Args:
                h: Input activity.
                stride: Stride of conv. layer (padding is set to 1).
                shortcut: If set, this tensor will be added to the activation
                    before the non-linearity is applied.
                no_conv: If True, no convolutional layer is applied.

            Returns:
                Output of layer.
            """
            nonlocal layer_ind, cm_ind, bn_ind

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

            # Non-linearity
            h = F.relu(h)

            # Context-dependent modulation (post-activation).
            if self._use_context_mod and self._context_mod_post_activation:
                h = self._context_mod_layers[cm_ind].forward(h,
                    weights=cm_weights_layer[cm_ind], ckpt_id=cmod_cond)
                cm_ind += 1

            if not no_conv:
                h = F.conv2d(h, layer_weights[layer_ind],
                    bias=layer_biases[layer_ind], stride=stride, padding=1)
                layer_ind += 1

            if shortcut is not None:
                h += shortcut

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
        h = F.conv2d(h, layer_weights[layer_ind], bias=layer_biases[layer_ind],
                     stride=self._strides[0], padding=1)
        layer_ind += 1

        ### Three groups, each containing n resnet blocks.
        for i in range(3):
            # Only the first layer in a group may be a strided convolution.
            stride = self._strides[i+1]
            # For each resnet block . A resnet block consists of 2 convolutional
            # layers.
            for j in range(self._n):
                shortcut_h = h
                if j == 0 and self._group_has_1x1[i]:
                    shortcut_h = F.conv2d(h, skip_1x1_weights[i], bias=None,
                                          stride=stride, padding=0)

                h = conv_layer(h, stride, shortcut=None)

                if self._dropout_rate != -1:
                    h = self._dropout(h)

                stride = 1

                h = conv_layer(h, stride, shortcut=shortcut_h)

        ### Final batchnorm + non-linearity.
        # Note, that the logical structure of a resnet block in a wide resnet
        # ends with the convolution. So we still need to apply the batchnorm
        # and non-linearity for the very last resnet block.
        h = conv_layer(h, None, shortcut=None, no_conv=True)

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
        # FIXME Method has been copied and only slightly modified from class
        # `ResNet`.
        in_shape = self._in_shape
        fs = self._filter_sizes
        ks = self._kernel_size
        strides = self._strides
        pd = 1 # all paddings are 1.
        assert len(ks) == 2
        assert len(fs) == 4
        n = self._n

        # Note, `in_shape` is in Tensorflow layout.
        assert(len(in_shape) == 3)
        in_shape = [in_shape[2], *in_shape[:2]]

        ret = []

        C, H, W = in_shape

        # Recall the formular for convolutional layers:
        # W_new = (W - K + 2P) // S + 1

        # First conv layer.
        C = fs[0]
        H = (H - ks[0] + 2*pd) // strides[0] + 1
        W = (W - ks[1] + 2*pd) // strides[0] + 1
        ret.append([C, H, W])

        # First block (only first layer may have stride).
        C = fs[1]
        H = (H - ks[0] + 2*pd) // strides[1] + 1
        W = (W - ks[1] + 2*pd) // strides[1] + 1
        ret.extend([[C, H, W]] * (2*n))

        # Second block (only first layer may have stride).
        C = fs[2]
        H = (H - ks[0] + 2*pd) // strides[2] + 1
        W = (W - ks[1] + 2*pd) // strides[2] + 1
        ret.extend([[C, H, W]] * (2*n))

        # Third block (only first layer may have stride).
        C = fs[3]
        H = (H - ks[0] + 2*pd) // strides[3] + 1
        W = (W - ks[1] + 2*pd) // strides[3] + 1
        ret.extend([[C, H, W]] * (2*n))

        # Final fully-connected layer (after avg pooling), i.e., output size.
        ret.append([self._num_classes])

        assert len(ret) == 6*n + 2

        return ret

    def __str__(self):
        n_conv_layers = 1 +  6 * self._n + np.sum(self._group_has_1x1)
        # WRN-d-k-B(3,3)
        return 'WRN-%d-%d-B(3,3)' % (n_conv_layers, self._k)

    def get_output_weight_mask(self, out_inds=None, device=None):
        """Create a mask for selecting weights connected solely to certain
        output units.

        See docstring of overwritten super method
        :meth:`mnets.mnet_interface.MainNetInterface.get_output_weight_mask`.
        """
        # Note: Super method `get_output_weight_mask` fails if `has_bias` fails.
        # Therefore, we simply replace `has_bias` with `_use_fc_bias`.

        if not (self.has_fc_out and self.mask_fc_out):
            raise NotImplementedError('Method not applicable for this ' +
                                      'network type.')

        ret = [None] * len(self.param_shapes)

        obias_ind = len(self.param_shapes)-1 if self._use_fc_bias else None
        oweights_ind = len(self.param_shapes)-2 if self._use_fc_bias \
            else len(self.param_shapes)-1

        # Bias weights for outputs.
        if obias_ind is not None:
            if out_inds is None:
                mask = torch.ones(*self.param_shapes[obias_ind],
                                  dtype=torch.bool)
            else:
                mask = torch.zeros(*self.param_shapes[obias_ind],
                                   dtype=torch.bool)
                mask[out_inds] = 1
            if device is not None:
                mask = mask.to(device)
            ret[obias_ind] = mask

        # Weights from weight matrix of output layer.
        if out_inds is None:
            mask = torch.ones(*self.param_shapes[oweights_ind],
                              dtype=torch.bool)
        else:
            mask = torch.zeros(*self.param_shapes[oweights_ind],
                               dtype=torch.bool)
            mask[out_inds, :] = 1
        if device is not None:
            mask = mask.to(device)
        ret[oweights_ind] = mask

        return ret


if __name__ == '__main__':
    pass
