#!/usr/bin/env python3
# Copyright 2019 Maria Cervera
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
# title          :mnets/simple_rnn.py
# author         :mc, be
# contact        :mariacer, behret@ethz.ch
# created        :10/28/2019
# version        :1.0
# python_version :3.6.8
"""
SimpleRNN
---------

Implementation of a simple recurrent neural network that has stacked vanilla RNN
or LSTM layers that are optionally enclosed by fully-connected layers.

An example usage is as a main model, where the main weights are initialized
and protected by a method such as EWC, and the context-modulation patterns of
the neurons are produced by an external hypernetwork.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils.torch_utils import init_params

class SimpleRNN(nn.Module, MainNetInterface):
    """Implementation of a simple RNN.

    This is a simple recurrent network, that receives input vector
    :math:`\mathbf{x}` and outputs a vector :math:`\mathbf{y}` of real values.

    Note:
        The output is non-linear if the last layer is recurrent! Otherwise,
        logits are returned (cmp. attribute
        :attr:`mnets.mnet_interface.MainNetInterface.has_fc_out`).

    Args:
        n_in (int): Number of inputs.
        rnn_layers (list or tuple): List of integers. Each entry denotes the
            size of a recurrent layer. Recurrent layers will simply be stacked
            as layers of this network.

            If ``fc_layers_pre`` is empty, then the recurrent layers are the
            initial layers.
            If ``fc_layers`` is empty, then the last entry of this list will
            denote the output size.

            Note:
                This list may never be empty.
        fc_layers_pre (list or tuple): List of integers. Before the recurrent
            layers a set of fully-connected layers may be added. This might be
            specially useful when constructing recurrent autoencoders. The
            entries of this list will denote the sizes of those layers.

            If ``fc_layers_pre`` is not empty, its first entry will denote the
            input size of this network.
        fc_layers (list or tuple): List of integers. After the recurrent layers,
            a set of fully-connected layers is added. The entries of this list
            will denote the sizes of those layers.

            If ``fc_layers`` is not empty, its last entry will denote the output
            size of this network.
        activation: The nonlinearity used in hidden layers.
        use_lstm (bool): If set to `True``, the recurrent layers will be LSTM
            layers.
        use_bias (bool): Whether layers may have bias terms.
        no_weights (bool): If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.
        init_weights (list, optional): This option is for convinience reasons.
            The option expects a list of parameter values that are used to
            initialize the network weights. As such, it provides a
            convinient way of initializing a network with a weight draw
            produced by the hypernetwork.

            Note, internal weights (see 
            :attr:`mnets.mnet_interface.MainNetInterface.weights`) will be
            affected by this argument only.
        kaiming_rnn_init (bool): By default, PyTorch initializes its recurrent
            layers uniformly with an interval defined by the square-root of the
            inverse of the layer size.

            If this option is enabled, then the recurrent layers will be
            initialized using the kaiming init as implemented by the function
            :func:`utils.torch_utils.init_params`.
        context_mod_last_step (bool): Whether context modulation is applied
            at the last time step os a recurrent layer only. If ``False``,
            context modulation is applied at every time step.

            Note:
                This option only applies if ``use_context_mod`` is ``True``.
        context_mod_num_ts (int, optional): The maximum number of timesteps.
            If specified, context-modulation with a different set of weights is
            applied at every timestep. If ``context_mod_separate_layers_per_ts``
            is ``True``, then a separate context-mod layer per timestep will be
            created. Otherwise, a single context-mod layer is created, but the
            expected parameter shapes for this layer are
            ``[context_mod_num_ts, *context_mod_shape]``.

            Note:
                This option only applies if ``use_context_mod`` is ``True``.
        context_mod_separate_layers_per_ts (bool): If specified, a separate
            context-mod layer per timestep is created (required if
            ``context_mod_no_weights`` is ``False``).
            
            Note:
                Only applies if ``context_mod_num_ts`` is specified.
        verbose (bool): Whether to print information (e.g., the number of
            weights) during the construction of the network.
        **kwargs: Keyword arguments regarding context modulation. This class
            can process the same context-modulation related arguments as class
            :class:`mnets.mlp.MLP` (plus the additional ones noted above).
    """
    def __init__(self, n_in=1, rnn_layers=(10,), fc_layers_pre=(),
                 fc_layers=(1,), activation=torch.nn.Tanh(), use_lstm=False,
                 use_bias=True, no_weights=False,
                 init_weights=None, kaiming_rnn_init=False,
                 context_mod_last_step=False,
                 context_mod_num_ts=-1,
                 context_mod_separate_layers_per_ts=False,
                 verbose=True,
                 **kwargs):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        self._bptt_depth = -1

        # FIXME Arbitrary restriction.
        if activation is None or isinstance(activation, (torch.nn.ReLU, \
                torch.nn.Tanh)):
            self._a_fun = activation
        else:
            raise ValueError('Only linear, relu and tanh activations are ' + \
                             'allowed for recurrent networks.')

        if len(rnn_layers) == 0:
            raise ValueError('The network always needs to have at least one ' +
                             'recurrent layer.')
        if len(fc_layers) == 0:
            has_rec_out_layer = True
            #n_out = rnn_layers[-1]
        else:
            has_rec_out_layer = False
            #n_out = fc_layers[-1]

        self._n_in = n_in
        self._rnn_layers = rnn_layers
        self._fc_layers_pre = fc_layers_pre
        self._fc_layers = fc_layers

        self._no_weights = no_weights

        ### Parse or set context-mod arguments ###
        rem_kwargs = MainNetInterface._parse_context_mod_args(kwargs)
        if len(rem_kwargs) > 0:
            raise ValueError('Keyword arguments %s unknown.' % str(rem_kwargs))

        self._use_context_mod = kwargs['use_context_mod']
        self._context_mod_inputs = kwargs['context_mod_inputs']
        self._no_last_layer_context_mod = kwargs['no_last_layer_context_mod']
        self._context_mod_no_weights = kwargs['context_mod_no_weights']
        self._context_mod_post_activation = \
            kwargs['context_mod_post_activation']
        self._context_mod_gain_offset = kwargs['context_mod_gain_offset']
        self._context_mod_gain_softplus = kwargs['context_mod_gain_softplus']

        # Context-mod options specific to RNNs
        self._context_mod_last_step = context_mod_last_step
        # FIXME We have to specify this option even if
        # `context_mod_separate_layers_per_ts` is False (in order to set
        # sensible parameter shapes). However, the forward method can deal with
        # an arbitrary timestep length.
        self._context_mod_num_ts = context_mod_num_ts
        self._context_mod_separate_layers_per_ts = \
            context_mod_separate_layers_per_ts

        # More appropriate naming of option.
        self._context_mod_outputs = not self._no_last_layer_context_mod

        if context_mod_num_ts != -1:
            if context_mod_last_step:
                raise ValueError('Options "context_mod_last_step" and ' +
                                 '"context_mod_num_ts" are not compatible.')
            if not self._context_mod_no_weights and \
                    not context_mod_separate_layers_per_ts:
                raise ValueError('When applying context-mod per timestep ' +
                    'while maintaining weights internally, option' +
                    '"context_mod_separate_layers_per_ts" must be set.')
        ### Parse or set context-mod arguments - DONE ###

        self._has_bias = use_bias
        self._has_fc_out = True if not has_rec_out_layer else False
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True if not has_rec_out_layer else False
        # Note, recurrent layers always use non-linearities and their activities
        # are squashed by a non-linearity (otherwise, internal states could
        # vanish/explode with increasing sequence length).
        self._has_linear_out = True if not has_rec_out_layer else False

        self._param_shapes = []
        self._param_shapes_meta = []

        self._weights = None if no_weights and self._context_mod_no_weights \
            else nn.ParameterList()
        self._hyper_shapes_learned = None \
            if not no_weights and not self._context_mod_no_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []

        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        self._use_lstm = use_lstm
        if use_lstm:
            self._rnn_fct = self.lstm_rnn_step
        else:
            self._rnn_fct = self.basic_rnn_step

        #################################################
        ### Define and initialize context mod weights ###
        #################################################

        # The context-mod layers consist of sequential layers ordered as:
        # - initial fully-connected layers if len(fc_layers_pre)>0
        # - recurrent layers
        # - final fully-connected layers if len(fc_layers)>0

        self._context_mod_layers = nn.ModuleList() if self._use_context_mod \
            else None

        self._cm_rnn_start_ind = 0
        self._num_fc_cm_layers = None
        if self._use_context_mod:
            cm_layer_inds = []
            cm_shapes = []

            # Gather sizes of all activation vectors within the network that
            # will be subject to context-modulation.
            if self._context_mod_inputs:
                self._cm_rnn_start_ind += 1
                cm_shapes.append([n_in])

                # We reserve layer zero for input context-mod. Otherwise, there
                # is no layer zero.
                cm_layer_inds.append(0)

            if len(fc_layers_pre) > 0:
                self._cm_rnn_start_ind += len(fc_layers_pre)

            # We use odd numbers for actual layers and even number for all
            # context-mod layers.
            rem_cm_inds = range(2, 2*(len(fc_layers_pre)+len(rnn_layers)+\
                len(fc_layers))+1, 2)

            num_rec_cm_layers = len(rnn_layers)
            if has_rec_out_layer and not self._context_mod_outputs:
                num_rec_cm_layers -= 1
            self._num_rec_cm_layers = num_rec_cm_layers

            jj = 0
            # Add initial fully-connected context-mod layers.
            num_fc_pre_cm_layers = len(fc_layers_pre)
            self._num_fc_pre_cm_layers = num_fc_pre_cm_layers
            for i in range(num_fc_pre_cm_layers):
                cm_shapes.append([fc_layers_pre[i]])
                cm_layer_inds.append(rem_cm_inds[jj])
                jj += 1

            # Add recurrent context-mod layers.
            for i in range(num_rec_cm_layers):
                if context_mod_num_ts != -1:
                    if context_mod_separate_layers_per_ts:
                        cm_rnn_shapes = [[rnn_layers[i]]] * context_mod_num_ts
                    else:
                        # Only a single context-mod layer will be added, but we
                        # directly edit the correponding `param_shape` later.
                        assert self._context_mod_no_weights
                        cm_rnn_shapes = [[rnn_layers[i]]]
                else:
                    cm_rnn_shapes = [[rnn_layers[i]]]

                cm_shapes.extend(cm_rnn_shapes)
                cm_layer_inds.extend([rem_cm_inds[jj]] * len(cm_rnn_shapes))
                jj += 1

            # Add final fully-connected context-mod layers.
            num_fc_cm_layers = len(fc_layers)
            if num_fc_cm_layers > 0 and not self._context_mod_outputs:
                num_fc_cm_layers -= 1
            self._num_fc_cm_layers = num_fc_cm_layers
            for i in range(num_fc_cm_layers):
                cm_shapes.append([fc_layers[i]])
                cm_layer_inds.append(rem_cm_inds[jj])
                jj += 1

            self._add_context_mod_layers(cm_shapes, cm_layers=cm_layer_inds)

            if context_mod_num_ts != -1 and not \
                    context_mod_separate_layers_per_ts:
                # In this case, there is only one context-mod layer for each
                # recurrent layer, but we want to have separate weights per
                # timestep.
                # Hence, we adapt the expected parameter shape, such that we
                # get a different set of weights per timestep. This will be
                # split into multiple weights that are succesively fed into the
                # same layer inside the forward method.

                for i in range(num_rec_cm_layers):
                    cmod_layer = \
                        self.context_mod_layers[self._cm_rnn_start_ind+i]
                    cm_shapes_rnn = [[context_mod_num_ts, *s] for s in \
                                      cmod_layer.param_shapes]

                    ps_ind = int(np.sum([ \
                        len(self.context_mod_layers[ii].param_shapes) \
                        for ii in range(self._cm_rnn_start_ind+i)]))
                    self._param_shapes[ps_ind:ps_ind+len(cm_shapes_rnn)] = \
                        cm_shapes_rnn
                    assert self._hyper_shapes_learned is not None
                    self._hyper_shapes_learned[ \
                        ps_ind:ps_ind+len(cm_shapes_rnn)] = cm_shapes_rnn

        ########################
        ### Internal weights ###
        ########################
        prev_dim = self._n_in

        def define_fc_layer_weights(fc_layers, prev_dim, num_prev_layers):
            """Define the weights and shapes of the fully-connected layers.

            Args:
                fc_layers (list): The list of fully-connected layer dimensions.
                prev_dim (int): The output size of the previous layer.
                num_prev_layers (int): The number of upstream layers to the 
                    current one (a layer with its corresponding
                    context-mod layer(s) count as one layer). Count should
                    start at ``1``.

            Returns:
                (int): The output size of the last fully-connected layer
                considered here.
            """
            # FIXME We should instead build an MLP instance. But then we still
            # have to adapt all attributes accordingly.
            for i, n_fc in enumerate(fc_layers):
                s_w = [n_fc, prev_dim]
                s_b = [n_fc] if self._has_bias else None

                for j, s in enumerate([s_w, s_b]):
                    if s is None:
                        continue

                    is_bias = True
                    if j % 2 == 0:
                        is_bias = False

                    if not self._no_weights:
                        self._weights.append(nn.Parameter(torch.Tensor(*s),
                                                          requires_grad=True))
                        if is_bias:
                            self._layer_bias_vectors.append(self._weights[-1])
                        else:
                            self._layer_weight_tensors.append(self._weights[-1])
                    else:
                        self._hyper_shapes_learned.append(s)
                        self._hyper_shapes_learned_ref.append( \
                            len(self.param_shapes))

                    self._param_shapes.append(s)
                    self._param_shapes_meta.append({
                        'name': 'bias' if is_bias else 'weight',
                        'index': -1 if self._no_weights else \
                            len(self._weights)-1,
                        'layer': i * 2 + num_prev_layers, # Odd numbers
                    })

                prev_dim = n_fc

            return prev_dim

        ### Initial fully-connected layers.
        prev_dim = define_fc_layer_weights(self._fc_layers_pre, prev_dim, 1)

        ### Recurrent layers.
        coeff = 4 if self._use_lstm else 1
        for i, n_rec in enumerate(self._rnn_layers):
            # Input-to-hidden
            s_w_ih = [n_rec*coeff, prev_dim]
            s_b_ih = [n_rec*coeff] if use_bias else None

            # Hidden-to-hidden
            s_w_hh = [n_rec*coeff, n_rec]
            s_b_hh = [n_rec*coeff] if use_bias else None

            # Hidden-to-output.
            # Note, for an LSTM cell, the hidden state vector is also the
            # output vector.
            if not self._use_lstm:
                s_w_ho = [n_rec, n_rec]
                s_b_ho = [n_rec] if use_bias else None
            else:
                s_w_ho = None
                s_b_ho = None

            for j, s in enumerate([s_w_ih, s_b_ih, s_w_hh, s_b_hh, s_w_ho,
                                   s_b_ho]):
                if s is None:
                    continue

                is_bias = True
                if j % 2 == 0:
                    is_bias = False

                wtype = 'ih'
                if 2 <= j < 4:
                    wtype = 'hh'
                elif j >=4:
                    wtype = 'ho'

                if not no_weights:
                    self._weights.append(nn.Parameter(torch.Tensor(*s),
                                                      requires_grad=True))
                    if is_bias:
                        self._layer_bias_vectors.append(self._weights[-1])
                    else:
                        self._layer_weight_tensors.append(self._weights[-1])
                else:
                    self._hyper_shapes_learned.append(s)
                    self._hyper_shapes_learned_ref.append( \
                        len(self.param_shapes))

                self._param_shapes.append(s)
                self._param_shapes_meta.append({
                    'name': 'bias' if is_bias else 'weight',
                    'index': -1 if no_weights else len(self._weights)-1,
                    'layer': i * 2 + 1 + 2 * len(fc_layers_pre), # Odd numbers
                    'info': wtype
                })

            prev_dim = n_rec

        ### Fully-connected layers.
        prev_dim = define_fc_layer_weights(self._fc_layers, prev_dim, \
            1 + 2 * len(fc_layers_pre) + 2 * len(rnn_layers))

        ### Initialize weights.
        if init_weights is not None:
            assert self._weights is not None
            assert len(init_weights) == len(self.weights)
            for i in range(len(init_weights)):
                assert np.all(np.equal(list(init_weights[i].shape),
                                       self.weights[i].shape))
                self.weights[i].data = init_weights[i]
        else:
            rec_start = len(fc_layers_pre)
            rec_end = rec_start + len(rnn_layers) * (2 if use_lstm else 3)
            # Note, Pytorch applies a uniform init to its recurrent layers, as
            # defined here:
            # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py#L155
            for i in range(len(self._layer_weight_tensors)):
                if i >=rec_start and i < rec_end:
                    # Recurrent layer weights.
                    if kaiming_rnn_init:
                        init_params(self._layer_weight_tensors[i],
                            self._layer_bias_vectors[i] if use_bias else None)
                    else:
                        a = 1.0 / math.sqrt(rnn_layers[(i-rec_start) // \
                            (2 if use_lstm else 3)])
                        nn.init.uniform_(self._layer_weight_tensors[i], -a, a)
                        if use_bias:
                            nn.init.uniform_(self._layer_bias_vectors[i], -a, a)
                else:
                    # FC layer weights.
                    init_params(self._layer_weight_tensors[i],
                        self._layer_bias_vectors[i] if use_bias else None)

        num_weights = MainNetInterface.shapes_to_num_weights(self._param_shapes)

        if verbose:
            if self._use_context_mod:
                cm_num_weights =  \
                    MainNetInterface.shapes_to_num_weights(cm_shapes)

            print('Creating a simple RNN with %d weights' % num_weights
                  + (' (including %d weights associated with-' % cm_num_weights
                     + 'context modulation)' if self._use_context_mod else '')
                  + '.')

        self._is_properly_setup()

    @property
    def bptt_depth(self):
        """The truncation depth for backprop through time.

        If ``-1``, backprop through time (BPTT) will unroll all timesteps
        present in the input. Otherwise, the forward pass will detach the
        RNN hidden states smaller or equal than ``num_timesteps - bptt_depth``
        timesteps, resulting in truncated BPTT (T-BPTT).

        :type: int
        """
        return self._bptt_depth

    @bptt_depth.setter
    def bptt_depth(self, value):
        self._bptt_depth = value

    @property
    def num_rec_layers(self):
        """Number of recurrent layers in this network (i.e., length of
        constructor argument ``rnn_layers``).

        :type: int
        """
        return len(self._rnn_layers)

    @property
    def use_lstm(self):
        """See constructor argument ``use_lstm``.

        :type: bool
        """
        return self._use_lstm

    def split_cm_weights(self, cm_weights, condition, num_ts=0):
        """Split context-mod weights per context-mod layer.

        Args:
            cm_weights (torch.Tensor): All context modulation weights.
            condition (optional, int): If provided, then this argument will be
                passed as argument ``ckpt_id`` to the method
                :meth:`utils.context_mod_layer.ContextModLayer.forward`.
            num_ts (int): The length of the sequences.

        Returns:
            (Tuple): Where the tuple contains:

            - **cm_inputs_weights**: The cm input weights.
            - **cm_fc_pre_layer_weights**: The cm pre-recurrent weights.
            - **cm_rec_layer_weights**: The cm recurrent weights.
            - **cm_fc_layer_weights**: The cm post-recurrent weights.
            - **n_cm_rec**: The number of recurrent cm layers.
            - **cmod_cond**: The context-mod condition.
        """

        n_cm_rec = -1
        cm_fc_pre_layer_weights = None
        cm_fc_layer_weights = None
        cm_inputs_weights = None
        cm_rec_layer_weights = None
        if cm_weights is not None:
            if self._context_mod_num_ts != -1 and \
                    self._context_mod_separate_layers_per_ts:
                assert num_ts <= self._context_mod_num_ts

            # Note, an mnet layer might contain multiple context-mod layers
            # (a recurrent layer can have a separate context-mod layer per
            # timestep).
            cm_fc_pre_layer_weights = []
            cm_rec_layer_weights = [[] for _ in range(self._num_rec_cm_layers)]
            cm_fc_layer_weights = []

            # Number of cm-layers per recurrent layer.
            n_cm_per_rec = self._context_mod_num_ts if \
                self._context_mod_num_ts != -1 and \
                    self._context_mod_separate_layers_per_ts else 1
            n_cm_rec = n_cm_per_rec * self._num_rec_cm_layers

            cm_start = 0
            for i, cm_layer in enumerate(self.context_mod_layers):
                cm_end = cm_start + len(cm_layer.param_shapes)

                if i == 0 and self._context_mod_inputs:
                    cm_inputs_weights = cm_weights[cm_start:cm_end]
                elif i < self._cm_rnn_start_ind:
                    cm_fc_pre_layer_weights.append(cm_weights[cm_start:cm_end])
                elif i >= self._cm_rnn_start_ind and \
                        i < self._cm_rnn_start_ind + n_cm_rec:
                    # Index of recurrent layer.
                    i_r = (i-self._cm_rnn_start_ind) // n_cm_per_rec
                    cm_rec_layer_weights[i_r].append( \
                        cm_weights[cm_start:cm_end])
                else:
                    cm_fc_layer_weights.append(cm_weights[cm_start:cm_end])
                cm_start = cm_end

            # We need to split the context-mod weights in the following case,
            # as they are currently just stacked on top of each other.
            if self._context_mod_num_ts != -1 and \
                    not self._context_mod_separate_layers_per_ts:
                for i, cm_w_list in enumerate(cm_rec_layer_weights):
                    assert len(cm_w_list) == 1

                    cm_rnn_weights = cm_w_list[0]
                    cm_rnn_layer = self.context_mod_layers[ \
                        self._cm_rnn_start_ind+i]

                    assert len(cm_rnn_weights) == len(cm_rnn_layer.param_shapes)
                    # The first dimension are the weights of this layer per
                    # timestep.
                    num_ts_cm = -1
                    for j, s in enumerate(cm_rnn_layer.param_shapes):
                        assert len(cm_rnn_weights[j].shape) == len(s) + 1
                        if j == 0:
                            num_ts_cm = cm_rnn_weights[j].shape[0]
                        else:
                            assert num_ts_cm == cm_rnn_weights[j].shape[0]
                    assert num_ts <= num_ts_cm

                    cm_w_chunked = [None] * len(cm_rnn_weights)
                    for j, cm_w in enumerate(cm_rnn_weights):
                        cm_w_chunked[j] = torch.chunk(cm_w, num_ts_cm, dim=0)

                    # Now we gather all these chunks to assemble the weights
                    # needed per timestep (as if
                    # `_context_mod_separate_layers_per_t` were True).
                    cm_w_list = []
                    for j in range(num_ts_cm):
                        tmp_list = []
                        for chunk in cm_w_chunked:
                            tmp_list.append(chunk[j].squeeze(dim=0))
                        cm_w_list.append(tmp_list)
                    cm_rec_layer_weights[i] = cm_w_list

            # Note, the last layer does not necessarily have context-mod
            # (depending on `self._context_mod_outputs`).
            if len(cm_rec_layer_weights) < len(self._rnn_layers):
                cm_rec_layer_weights.append(None)
            if len(cm_fc_layer_weights) < len(self._fc_layers):
                cm_fc_layer_weights.append(None)


        #######################
        ### Parse condition ###
        #######################
        cmod_cond = None
        if condition is not None:
            assert isinstance(condition, int)
            cmod_cond = condition

            # Note, the cm layer will ignore the cmod condition if weights
            # are passed.
            # FIXME Find a more elegant solution.
            cm_inputs_weights = None
            cm_fc_pre_layer_weights = [None] * len(cm_fc_pre_layer_weights)
            cm_rec_layer_weights = [[None] * len(cm_ws) for cm_ws in \
                                    cm_rec_layer_weights]
            cm_fc_layer_weights = [None] * len(cm_fc_layer_weights)

        return cm_inputs_weights, cm_fc_pre_layer_weights, cm_fc_layer_weights,\
            cm_rec_layer_weights, n_cm_rec, cmod_cond

    def split_internal_weights(self, int_weights):
        """Split internal weights per layer.

        Args:
            int_weights (torch.Tensor): All internal weights.

        Returns:
            (Tuple): Where the tuple contains:

            - **fc_pre_w_weights**: The pre-recurrent w weights.
            - **fc_pre_b_weights**: The pre-recurrent b weights.
            - **rec_weights**: The recurrent weights.
            - **fc_w_weights**:The post-recurrent w weights.
            - **fc_b_weights**: The post-recurrent b weights.
        """
        n_cm = self._num_context_mod_shapes()

        int_meta = self.param_shapes_meta[n_cm:]
        assert len(int_meta) == len(int_weights)
        fc_pre_w_weights = []
        fc_pre_b_weights = []
        rec_weights =[[] for _ in range(len(self._rnn_layers))]
        fc_w_weights = []
        fc_b_weights = []

        # Number of pre-fc weights in total.
        n_fc_pre = len(self._fc_layers_pre)
        if self.has_bias:
            n_fc_pre *= 2

        # Number of weights per recurrent layer.
        if self._use_lstm:
            n_rw = 4 if self.has_bias else 2
        else:
            n_rw = 6 if self.has_bias else 3

        for i, w in enumerate(int_weights):
            if i < n_fc_pre: # fc pre weights
                if int_meta[i]['name'] == 'weight':
                    fc_pre_w_weights.append(w)
                else:
                    assert int_meta[i]['name'] == 'bias'
                    fc_pre_b_weights.append(w)
            elif i >= n_fc_pre and \
                    i < n_rw * len(self._rnn_layers) + n_fc_pre: # recurrent w
                r_ind = (i - n_fc_pre) // n_rw
                rec_weights[r_ind].append(w)
            else: # fc weights
                if int_meta[i]['name'] == 'weight':
                    fc_w_weights.append(w)
                else:
                    assert int_meta[i]['name'] == 'bias'
                    fc_b_weights.append(w)

        if not self.has_bias:
            assert len(fc_pre_b_weights) == 0
            fc_pre_b_weights = [None] * len(fc_pre_w_weights)

            assert len(fc_b_weights) == 0
            fc_b_weights = [None] * len(fc_w_weights)

        return fc_pre_w_weights, fc_pre_b_weights, rec_weights, fc_w_weights, \
            fc_b_weights

    def split_weights(self, weights):
        """Split weights into internal and context-mod weights.

        Extract which weights should be used,  I.e., are we using internally
        maintained weights or externally given ones or are we even mixing
        between these groups.

        Args:
            weights (torch.Tensor): All weights.

        Returns:
            (Tuple): Where the tuple contains:

            - **int_weights**: The internal weights.
            - **cm_weights**: The context-mod weights.
        """
        n_cm = self._num_context_mod_shapes()

        ### FIXME Code copied from MLP its `forward` method ###

        # Make sure cm_weights are either `None` or have the correct dimensions.
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
                assert 'internal_weights' in weights.keys() or \
                       'mod_weights' in weights.keys()
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
                assert len(cm_weights) == n_cm
            int_shapes = self.param_shapes[n_cm:]
            assert len(int_weights) == len(int_shapes)
            for i, s in enumerate(int_shapes):
                assert np.all(np.equal(s, list(int_weights[i].shape)))

        ### FIXME Code copied until here ###
        return int_weights, cm_weights

    def forward(self, x, weights=None, distilled_params=None, condition=None,
                return_hidden=False, return_hidden_int=False):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            weights (list or dict): See argument ``weights`` of method
                :meth:`mnets.mlp.MLP.forward`.
            condition (optional, int): If provided, then this argument will be
                passed as argument ``ckpt_id`` to the method
                :meth:`utils.context_mod_layer.ContextModLayer.forward`.
            return_hidden (bool, optional): If ``True``, all hidden activations
                of fully-connected and recurrent layers (where we defined
                :math:`y_t` as hidden state of vannila RNN layers as these are
                the layer output passed to the next layer) are returned.

                Specifically, hidden activations are the outputs of each hidden
                layer that are passed to the next layer.
            return_hidden_int (bool, optional): If ``True``, in addition to
                ``hidden``, an additional variable ``hidden_int`` is returned
                containing the internal hidden states of recurrent layers (i.e.,
                the cell states :math:`c_t` for LSTMs and the actual hidden
                state :math:`h_t` for Elman layers) are returned. Since fully-
                connected layers have no such internal hidden activations, the
                corresponding entry in ``hidden_int`` will be ``None``.

        Returns:
            (torch.Tensor or tuple): Where the tuple is containing:

            - **output** (torch.Tensor): The output of the network.
            - **hidden** (list): If ``return_hidden`` is ``True``, then the
              hidden activities of the layers are returned, which have
              the shape ``(seq_length, batch_size, n_hidden)``.
            - **hidden_int** (list): If ``return_hidden_int`` is ``True``, then
              in addition to ``hidden`` a tensor ``hidden_int`` per recurrent
              layer is returned containing internal hidden states. The list will
              contain a ``None`` entry for each fully-connected layer to ensure
              same length as ``hidden``.
        """
        assert distilled_params is None

        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        if return_hidden_int and not return_hidden:
            raise ValueError('"return_hidden_int" requires "return_hidden" ' + \
                             'to be set.')

        #######################
        ### Extract weights ###
        #######################
        # Extract which weights should be used.
        int_weights, cm_weights = self.split_weights(weights)

        ### Split context-mod weights per context-mod layer.
        cm_inputs_weights, cm_fc_pre_layer_weights, cm_fc_layer_weights, \
            cm_rec_layer_weights, n_cm_rec, cmod_cond = self.split_cm_weights(
                cm_weights, condition, num_ts=x.shape[0])

        ### Extract internal weights.
        fc_pre_w_weights, fc_pre_b_weights, rec_weights, fc_w_weights, \
            fc_b_weights = self.split_internal_weights(int_weights)

        ###########################
        ### Forward Computation ###
        ###########################
        ret_hidden = None
        if return_hidden:
            ret_hidden = []
        ret_hidden_int = [] # the internal hidden activations

        h = x

        cm_offset = 0
        if self._use_context_mod and self._context_mod_inputs:
            cm_offset += 1
            # Apply context modulation in the inputs.
            h = self._context_mod_layers[0].forward(h,
                weights=cm_inputs_weights, ckpt_id=cmod_cond, bs_dim=1)

        ### Initial fully-connected layer activities.
        ret_hidden, h = self.compute_fc_outputs(h, fc_pre_w_weights, \
            fc_pre_b_weights, len(self._fc_layers_pre), \
            cm_fc_pre_layer_weights, cm_offset, cmod_cond, False, ret_hidden)
        if return_hidden:
            ret_hidden_int = [None] * len(ret_hidden)

        ### Recurrent layer activities.
        for d in range(len(self._rnn_layers)):
            if self._use_context_mod:
                h, h_int = self.compute_hidden_states(h, d, rec_weights[d],
                    cm_rec_layer_weights[d], cmod_cond)
            else:
                h, h_int = self.compute_hidden_states(h, d, rec_weights[d],
                                                      None, None)
            if return_hidden:
                ret_hidden.append(h)
                ret_hidden_int.append(h_int)

        ### Fully-connected layer activities.
        cm_offset = self._cm_rnn_start_ind + n_cm_rec
        ret_hidden, h = self.compute_fc_outputs(h, fc_w_weights, fc_b_weights, \
            self._num_fc_cm_layers, cm_fc_layer_weights, cm_offset, cmod_cond,
            True, ret_hidden)
        if return_hidden:
            ret_hidden_int.extend( \
                [None] * (len(ret_hidden)-len(ret_hidden_int)))

        # FIXME quite ugly
        if return_hidden:
            # The last element is the output activity.
            ret_hidden.pop()
            if return_hidden_int:
                ret_hidden_int.pop()
                return h, ret_hidden, ret_hidden_int
            else:
                return h, ret_hidden
        else:
            return h

    def compute_fc_outputs(self, h, fc_w_weights, fc_b_weights, num_fc_cm_layers,
                cm_fc_layer_weights, cm_offset, cmod_cond, is_post_fc,
                ret_hidden):
        """Compute the forward pass through the fully-connected layers.

        This method also appends activations to ``ret_hidden``.

        Args:
            h (torch.Tensor): The input from the previous layer.
            fc_w_weights (list): The weights for the fc layers.
            fc_b_weights (list): The biases for the fc layers.
            num_fc_cm_layers (int): The number of context-modulation
                layers associated with this set of fully-connected layers.
            cm_fc_layer_weights (list): The context-modulation weights
                associated with the current layers.
            cm_offset (int): The index to access the correct context-mod
                layers.
            cmod_cond (bool): Some condition to perform context modulation.
            is_post_fc (bool); Whether those layers are applied as last
                layers of the network. In this case, there will be no
                activation applied to the last layer outputs.
            ret_hidden (list or None): The list where to append the hidden
                recurrent activations.

        Return:
            (Tuple): Tuple containing:

            - **ret_hidden**: The hidden recurrent activations.
            - **h**: Transformed activation ``h``.
        """
        for d in range(len(fc_w_weights)):
            use_cm = self._use_context_mod and d < num_fc_cm_layers
            # Compute output.
            h = F.linear(h, fc_w_weights[d], bias=fc_b_weights[d])

            # Context-dependent modulation (pre-activation).
            if use_cm and not self._context_mod_post_activation:
                h = self._context_mod_layers[cm_offset+d].forward(h,
                    weights=cm_fc_layer_weights[d], ckpt_id=cmod_cond, 
                    bs_dim=1)

            # Non-linearity
            # Note, non-linearity is not applied to outputs of the network.
            if self._a_fun is not None and \
                    (not is_post_fc or d < len(fc_w_weights)-1):
                h = self._a_fun(h)

            # Context-dependent modulation (post-activation).
            if use_cm and self._context_mod_post_activation:
                h = self._context_mod_layers[cm_offset+d].forward(h,
                    weights=cm_fc_layer_weights[d], ckpt_id=cmod_cond, 
                    bs_dim=1)

            if ret_hidden is not None:
                ret_hidden.append(h)

        return ret_hidden, h

    def compute_hidden_states(self, x, layer_ind, int_weights, cm_weights,
                              ckpt_id, h_0=None, c_0=None):
        """Compute the hidden states for the recurrent layer ``layer_ind`` from
        a sequence of inputs :math:`x`.

        If so specified, context modulation is applied before or after the
        nonlinearities.

        Args:
            x: The inputs :math:`x` to the layer. :math:`x` has shape
                ``[sequence_len, batch_size, n_hidden_prev]``.
            layer_ind (int): Index of the layer.
            int_weights: Internal weights associated with this recurrent layer.
            cm_weights: Context modulation weights.
            ckpt_id: Will be passed as option ``ckpt_id`` to method
                :meth:`utils.context_mod_layer.ContextModLayer.forward` if
                context-mod layers are used.
            h_0 (torch.Tensor, optional): The initial state for :math:`h`.
            c_0 (torch.Tensor, optional): The initial state for :math:`c`. Note
                that for LSTMs, if the initial state is to be defined, this
                variable is necessary also, not only :math:`h_0`, whereas for
                vanilla RNNs it is enough to provide :math:`h_0` as :math:`c_0`
                represents the output of the layer and it can be easily computed
                from `h_0`.

        Returns:
            (tuple): Tuple containing:

            - **outputs** (torch.Tensor): The sequence of visible hidden states
              given the input. It has shape 
              ``[sequence_len, batch_size, n_hidden]``.
            - **hiddens** (torch.Tensor): The sequence of hidden states given
              the input. For LSTMs, this corresponds to :math:`c`.
              It has shape ``[sequence_len, batch_size, n_hidden]``.
        """
        seq_length, batch_size, n_hidden_prev = x.shape
        n_hidden = self._rnn_layers[layer_ind]

        # Generate initial hidden states.
        # Note that for LSTMs h_0 is the hidden state and output vector whereas
        # c_0 is the internal cell state vector.
        # For a vanilla RNN h_0 is the hidden state whereas c_0 is the output
        # vector.
        if h_0 is None:
            h_0 = (torch.zeros(batch_size, n_hidden, device=x.device))
        if c_0 is None:
            c_0 = (torch.zeros(batch_size, n_hidden, device=x.device))
        assert h_0.shape[0] == c_0.shape[0] == batch_size
        assert h_0.shape[1] == c_0.shape[1] == n_hidden

        # If we want to apply context modulation in each time step, we need
        # to split the input sequence and call pytorch function at every
        # time step.
        outputs = []
        hiddens = []
        h_t = h_0
        c_t = c_0
        for t in range(seq_length):
            x_t = x[t,:,:]

            if cm_weights is not None and self._context_mod_num_ts != -1:
                curr_cm_weights = cm_weights[t]
            elif cm_weights is not None:
                assert len(cm_weights) == 1
                curr_cm_weights = cm_weights[0]
            else:
                curr_cm_weights = cm_weights

            # Compute the actual rnn step (either vanilla or LSTM, depending on
            # the flag self._use_lstm).
            is_last_step = t==(seq_length-1)
            h_t, c_t = self._rnn_fct(layer_ind, t, x_t, (h_t, c_t), int_weights,
                                     curr_cm_weights, ckpt_id, is_last_step)

            if self.bptt_depth != -1:
                if t < (seq_length - self.bptt_depth):
                    # Detach hidden/output states, such that we don't backprop
                    # through these timesteps.
                    h_t = h_t.detach()
                    c_t = c_t.detach()

            # FIXME Solution is a bit ugly. For an LSTM, the hidden state is
            # also the output whereas a normal RNN has a separate output.
            if self._use_lstm:
                outputs.append(h_t)
                hiddens.append(c_t)
            else:
                outputs.append(c_t)
                hiddens.append(h_t)

        return torch.stack(outputs), torch.stack(hiddens)


    def basic_rnn_step(self, d, t, x_t, h_t, int_weights, cm_weights, ckpt_id,
                       is_last_step):
        """Perform vanilla rnn pass from inputs to hidden units.

        Apply context modulation if necessary (i.e. if ``cm_weights`` is
        not ``None``).

        This function implements a step of an
        `Elman RNN <https://en.wikipedia.org/wiki/\
Recurrent_neural_network#Elman_networks_and_Jordan_networks>`__.

        Note:
            We made the following design choice regarding context-modulation.
            In contrast to the LSTM, the Elman network layer consists of "two
            steps", updating the hidden state and computing an output based
            on this hidden state. To be fair, context-mod should influence both
            these "layers". Therefore, we apply context-mod twice, but using the
            same weights. This of course assumes that the hidden state and
            output vector have the same dimensionality.

        Args:
            d (int): Index of the layer.
            t (int): Current timestep.
            x_t: Tensor of size ``[batch_size, n_hidden_prev]`` with inputs.
            h_t (tuple): Tuple of length 2, containing two tensors of size
                ``[batch_size, n_hidden]`` with previous hidden states ``h`` and
                and previous outputs ``y``.

                Note:
                    The previous outputs ``y`` are ignored by this method, since
                    they are not required in an Elman RNN step.
            int_weights: See docstring of method :meth:`compute_hidden_states`.
            cm_weights (list): The weights of the context-mod layer, if context-
                mod should be applied.
            ckpt_id: See docstring of method :meth:`compute_hidden_states`.
            is_last_step (bool): Whether the current time step is the last one.

        Returns:
            (tuple): Tuple containing:

            - **h_t** (torch.Tensor): The tensor ``h_t`` of size
              ``[batch_size, n_hidden]`` with the new hidden state.
            - **y_t** (torch.Tensor): The tensor ``y_t`` of size
              ``[batch_size, n_hidden]`` with the new cell state.
        """
        h_t = h_t[0]

        use_cm = self._use_context_mod and d < self._num_rec_cm_layers
        # Determine the index of the hidden context mod layer.
        # Number of cm-layers per recurrent layer.
        n_cm_per_rec = self._context_mod_num_ts if \
            self._context_mod_num_ts != -1 and \
                self._context_mod_separate_layers_per_ts else 1
        cm_idx = self._cm_rnn_start_ind + d * n_cm_per_rec
        if self._context_mod_num_ts != -1 and \
                self._context_mod_separate_layers_per_ts:
            cm_idx += t
 
        if self.has_bias:
            assert len(int_weights) == 6
            weight_ih = int_weights[0]
            bias_ih = int_weights[1]
            weight_hh = int_weights[2]
            bias_hh = int_weights[3]
        else:
            assert len(int_weights) == 3
            weight_ih = int_weights[0]
            bias_ih = None
            weight_hh = int_weights[1]
            bias_hh = None

        ###########################
        ### Update hidden state ###
        ###########################
        h_t = x_t @ weight_ih.t() + h_t @ weight_hh.t()
        if self.has_bias:
            h_t += bias_ih + bias_hh

        # Context-dependent modulation (pre-activation).
        if use_cm and not self._context_mod_post_activation:
            # Only apply context mod if you are in the last time step, or if
            # you want to apply it in every single time step (i.e. if
            # self._context_mod_last_step is False).
            if not self._context_mod_last_step or is_last_step:
                h_t = self._context_mod_layers[cm_idx].forward(h_t,
                    weights=cm_weights, ckpt_id=ckpt_id)

        # Compute activation.
        if self._a_fun is not None:
            h_t = self._a_fun(h_t)

        # Context-dependent modulation (post-activation).
        if use_cm and self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                h_t = self._context_mod_layers[cm_idx].forward(h_t,
                    weights=cm_weights, ckpt_id=ckpt_id)

        ######################
        ### Compute output ###
        ######################
        y_t = self.compute_basic_rnn_output(h_t, int_weights, use_cm, 
            cm_weights, cm_idx, ckpt_id, is_last_step)

        return h_t, y_t

    def compute_basic_rnn_output(self, h_t, int_weights, use_cm, cm_weights,
                                 cm_idx, ckpt_id, is_last_step):
        """Compute the output of a vanilla RNN given the hidden state.

        Args:
            (...): See docstring of method :meth:`basic_rnn_step`.
            use_cm (boolean): Whether context modulation is being used.
            cm_idx (int): Index of the context-mod layer.

        Returns:
            (torch.tensor): The output.
        """
        if self.has_bias:
            weight_ho = int_weights[4]
            bias_ho = int_weights[5]
        else:
            weight_ho = int_weights[2]
            bias_ho = None

        y_t = h_t @ weight_ho.t()
        if self.has_bias:
            y_t += bias_ho

        # Context-dependent modulation (pre-activation).
        if use_cm and not self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                y_t = self._context_mod_layers[cm_idx].forward(y_t,
                    weights=cm_weights, ckpt_id=ckpt_id)

        # Compute activation.
        if self._a_fun is not None:
            y_t = self._a_fun(y_t)

        # Context-dependent modulation (post-activation).
        if use_cm and self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                y_t = self._context_mod_layers[cm_idx].forward(y_t,
                    weights=cm_weights, ckpt_id=ckpt_id)
        return y_t

    def lstm_rnn_step(self, d, t, x_t, h_t, int_weights, cm_weights, ckpt_id,
                      is_last_step):
        """ Perform an LSTM pass from inputs to hidden units.

        Apply masks to the temporal sequence for computing the loss.
        Obtained from:

            https://mlexplained.com/2019/02/15/building-an-lstm-from-scratch-\
in-pytorch-lstms-in-depth-part-1/

        and:

            https://d2l.ai/chapter_recurrent-neural-networks/lstm.html

        Args:
            d (int): Index of the layer.
            t (int): Current timestep.
            x_t: Tensor of size ``[batch_size, n_inputs]`` with inputs.
            h_t (tuple): Tuple of length 2, containing two tensors of size
                ``[batch_size, n_hidden]`` with previous hidden states ``h`` and
                ``c``.
            int_weights: See docstring of method :meth:`basic_rnn_step`.
            cm_weights: See docstring of method :meth:`basic_rnn_step`.
            ckpt_id: See docstring of method :meth:`basic_rnn_step`.
            is_last_step (bool): See docstring of method :meth:`basic_rnn_step`.

        Returns:
            (tuple): Tuple containing:

            - **h_t** (torch.Tensor): The tensor ``h_t`` of size
              ``[batch_size, n_hidden]`` with the new hidden state.
            - **c_t** (torch.Tensor): The tensor ``c_t`` of size
              ``[batch_size, n_hidden]`` with the new cell state.
        """
        use_cm = self._use_context_mod and d < self._num_rec_cm_layers
        # Determine the index of the hidden context mod layer.
        # Number of cm-layers per recurrent layer.
        n_cm_per_rec = self._context_mod_num_ts if \
            self._context_mod_num_ts != -1 and \
                self._context_mod_separate_layers_per_ts else 1
        cm_idx = self._cm_rnn_start_ind + d * n_cm_per_rec
        if self._context_mod_num_ts != -1 and \
                self._context_mod_separate_layers_per_ts:
            cm_idx += t

        c_t = h_t[1]
        h_t = h_t[0]
        HS = self._rnn_layers[d]

        if self._has_bias:
            assert len(int_weights) == 4
            weight_ih = int_weights[0]
            bias_ih = int_weights[1]
            weight_hh = int_weights[2]
            bias_hh = int_weights[3]
        else:
            assert len(int_weights) == 2
            weight_ih = int_weights[0]
            bias_ih = None
            weight_hh = int_weights[1]
            bias_hh = None

        # Compute total pre-activation input.
        gates = x_t @ weight_ih.t() + h_t @ weight_hh.t()
        if self.has_bias:
            gates += bias_ih + bias_hh

        i_t = gates[:, :HS]
        f_t = gates[:, HS:HS*2]
        g_t = gates[:, HS*2:HS*3]
        o_t = gates[:, HS*3:]

        # Compute activation.
        i_t = torch.sigmoid(i_t) # input
        f_t = torch.sigmoid(f_t) # forget
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t) # output

        # Compute c states.
        c_t = f_t * c_t + i_t * g_t

        # Note, we don't want to context-modulate the internal state c_t.
        # Otherwise, it might explode over timesteps since it wouldn't be
        # limited to [-1, 1] anymore. Instead, we only modulate the current
        # state (which is used to compute the current output h_t.
        c_t_mod = c_t

        # Context-dependent modulation (pre-activation).
        if use_cm and not self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                c_t_mod = self._context_mod_layers[cm_idx].forward(c_t_mod,
                    weights=cm_weights, ckpt_id=ckpt_id)

        # Compute h states.
        if self._a_fun is not None:
            h_t = o_t * self._a_fun(c_t_mod)
        else:
            h_t = o_t * c_t_mod

        # Context-dependent modulation (post-activation).
        # FIXME Christian: Shouldn't we apply the output gate `o_t` after
        # applying post-activation context-mod?
        if use_cm and self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                h_t = self._context_mod_layers[cm_idx].forward(h_t,
                    weights=cm_weights, ckpt_id=ckpt_id)

        return h_t, c_t

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This network does not have any distillation targets.

        Returns:
            ``None``
        """
        return None

    def get_output_weight_mask(self, out_inds=None, device=None):
        """Get masks to select output weights.

        See docstring of overwritten super method
        :meth:`mnets.mnet_interface.MainNetInterface.get_output_weight_mask`.
        """
        if len(self._fc_layers) > 0:
            return MainNetInterface.get_output_weight_mask(self,
                out_inds=out_inds, device=device)

        # TODO Output layer is recurrent. Hence, we have to properly handle
        # which weights contribute solely to certain output activations.
        raise NotImplementedError()

    def get_cm_weights(self):
        """Get internal maintained weights that are associated with context-
        modulation.

        Returns:
            (list): List of weights from
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params` that
            are belonging to context-mod layers.
        """
        n_cm = self._num_context_mod_shapes()

        if n_cm == 0 or self._context_mod_no_weights:
            raise ValueError('Network maintains no context-modulation weights.')

        return self.internal_params[:n_cm]

    def get_non_cm_weights(self):
        """Get internal weights that are not associated with context-modulation.

        Returns:
            (list): List of weights from
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params` that
            are not belonging to context-mod layers.
        """
        n_cm = 0 if self._context_mod_no_weights else \
            self._num_context_mod_shapes()

        return self.internal_params[n_cm:]

    def get_cm_inds(self):
        """Get the indices of
        :attr:`mnets.mnet_interface.MainNetInterface.param_shapes` that are
        associated with context-modulation.

        Returns:
            (list): List of integers representing indices of
            :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.
        """
        n_cm = self._num_context_mod_shapes()

        ret = []

        for i, meta in enumerate(self.param_shapes_meta):
            if meta['name'] == 'cm_shift' or meta['name'] == 'cm_scale':
                ret.append(i)
        assert n_cm == len(ret)

        return ret

    def init_hh_weights_orthogonal(self):
        """Initialize hidden-to-hidden weights orthogonally.

        This method will overwrite the hidden-to-hidden weights of recurrent
        layers.
        """
        for meta in self.param_shapes_meta:
            if meta['name'] == 'weight' and 'info' in meta.keys() and \
                    meta['info'] == 'hh' and meta['index'] != -1:
                print('Initializing hidden-to-hidden weights of recurrent ' +
                      'layer %d orthogonally.' % meta['layer'])
                W = self.internal_params[meta['index']]
                # LSTM weight matrices are stored such that the hidden-to-hidden 
                # matrices for the 4 gates are concatenated.
                if self.use_lstm:
                    out_dim, _ = W.shape
                    assert out_dim % 4 == 0
                    fs = out_dim // 4

                    W1 = W[:fs, :]
                    W2 = W[fs:2*fs, :]
                    W3 = W[2*fs:3*fs, :]
                    W4 = W[3*fs:, :]

                    torch.nn.init.orthogonal_(W1.data)
                    torch.nn.init.orthogonal_(W2.data)
                    torch.nn.init.orthogonal_(W3.data)
                    torch.nn.init.orthogonal_(W4.data)

                    # Sanity check to show that the init on partial matrices
                    # propagates back to the original tensor.
                    assert W[0,0] == W1[0,0]
                else:
                    torch.nn.init.orthogonal_(W.data)

    def _internal_weight_shapes(self):
        """Compute the tensor shapes of all internal weights (i.e., those not
        associated with context-modulation).

        Returns:
            (list): A list of list of integers, denoting the shapes of the
            individual parameter tensors.
        """
        coeff = 4 if self._use_lstm else 1
        shapes = []

        # Initial fully-connected layers.
        prev_dim = self._n_in
        for n_fc in self._fc_layers_pre:
            shapes.append([n_fc, prev_dim])
            if self._use_bias:
                shapes.append([n_fc])

            prev_dim = n_fc

        # Recurrent layers.
        for n_rec in self._rnn_layers:
            # Input-to-hidden
            shapes.append([n_rec*coeff, prev_dim])
            if self._use_bias:
                shapes.append([n_rec*coeff])

            # Hidden-to-hidden
            shapes.append([n_rec*coeff, n_rec])
            if self._use_bias:
                shapes.append([n_rec*coeff])

            if not self._use_lstm:
                # Hidden-to-output
                shapes.append([n_rec, n_rec])
                if self._use_bias:
                    shapes.append([n_rec])

            prev_dim = n_rec

        # Fully-connected layers.
        for n_fc in self._fc_layers:
            shapes.append([n_fc, prev_dim])
            if self._use_bias:
                shapes.append([n_fc])

            prev_dim = n_fc

        return shapes

if __name__ == '__main__':
    pass

