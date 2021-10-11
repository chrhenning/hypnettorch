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
# @title          :mnets/bi_rnn.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :08/09/2020
# @version        :1.0
# @python_version :3.6.10
r"""
Bidirectional Recurrent Neural Network
--------------------------------------

This module implements a bidirectional recurrent neural networt (BiRNN).
To realize recurrent layers, it utilizes class
:class:`mnets.simple_rnn.SimpleRNN`. Hence different kinds of BiRNNs can be
realized, such as Elman-type BiRNNs and BiLSTMs. In particular, this class
implements the BiRNN in the following manner. Given an input :math:`x_{1:T}`,
the forward RNN is run to produce hidden states :math:`\hat{h}_{1:T}^{(f)}`
and the backward RNN is run to produce states :math:`\hat{h}_{1:T}^{(b)}`.

These hidden states are concatenated to produce the final hidden state which
is the output of the recurrent layer(s):
:math:`h_t = \text{concat}(\hat{h}_t^{(f)}, \hat{h}_t^{(b)})`.

Those inputs are subsequently processed by an instance of class
:class:`mnets.mlp.MLP` to produce the final network outputs.
"""

import torch
import torch.nn as nn
from warnings import warn

from hypnettorch.mnets.mlp import MLP
from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.mnets.simple_rnn import SimpleRNN

class BiRNN(nn.Module, MainNetInterface):
    r"""Implementation of a bidirectional RNN.

    Note:
        The output is non-linear if the last layer is recurrent! Otherwise,
        logits are returned (cmp. attribute
        :attr:`mnets.mnet_interface.MainNetInterface.has_fc_out`).

    Example:
        Here is an example instantiation of a BiLSTM with a single bidirectional
        layer of dimensionality 256, assuming 100 dimensional inputs and
        10 dimensional outputs.

        .. code-block:: python

            net = BiRNN(rnn_args={'n_in': 100, 'rnn_layers': [256],
                                  'use_lstm': True, 'fc_layers_pre': [],
                                  'fc_layers': []},
                        mlp_args={'n_in': 512, 'n_out': 10,
                                  'hidden_layers': []},
                        no_weights=False)

    Args:
        rnn_args (dict or list): A dictionary of arguments for an instance of
            class :class:`mnets.simple_rnn.SimpleRNN`. These arguments will be
            used to create two instances of this class, one representing the
            forward RNN and one the backward RNN.

            Note, each of these instances may contain multiple layers, even
            non-recurrent layers. The outputs of such an instance are considered
            the hidden activations :math:`\hat{h}_{1:T}^{(f)}` or
            :math:`\hat{h}_{1:T}^{(b)}`, respectively.

            To realize multiple bidirectional layers (which in itself can be
            multi-layer RNNs), one may provide a list of dictionaries. Each
            entry in such list will be used to generate a single bidirectional
            layer (i.e., consisting of two instances of class
            :class:`mnets.simple_rnn.SimpleRNN`). Note, the input size of
            each new layer has to be twice the size of :math:`\hat{h}_t^{(f)}`
            from the previous layer.
        mlp_args (dict, optional): A dictionary of arguments for class
            :class:`mnets.mlp.MLP`. The input size of such an MLP should be
            twice the size of :math:`\hat{h}_t^{(f)}`. If ``None``, then the
            output of the last bidirectional layer is considered the output of
            the network.
        preprocess_fct (func, optional): A function handle can be provided,
            that will process inputs ``x`` passed to the method :meth:`forward`.
            An example usecase could be the translation or selection of word
            embeddings.

            The function handle must have the signature:
            ``preprocess_fct(x, seq_lengths=None)``. See the corresponding
            argument descriptions of method :meth:`forward`.The function is
            expected to return the preprocessed ``x``.
        no_weights (bool): See parameter ``no_weights`` of class
            :class:`mnets.mlp.MLP`.
        verbose (bool): See parameter ``verbose`` of class
            :class:`mnets.mlp.MLP`.
    """
    def __init__(self, rnn_args={}, mlp_args=None, preprocess_fct=None,
                 no_weights=False, verbose=True):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        assert isinstance(rnn_args, (dict, list, tuple))
        assert mlp_args is None or isinstance(mlp_args, dict)

        if isinstance(rnn_args, dict):
            rnn_args = [rnn_args]

        self._forward_rnns = []
        self._backward_rnns = []
        self._out_mlp = None
        self._preprocess_fct = preprocess_fct
        self._forward_called = False

        # FIXME At the moment we do not control input and output size of
        # individual networks and need to assume that the user sets them
        # correctly.

        ### Create all forward and backward nets for each bidirectional layer.
        for rargs in rnn_args:
            assert isinstance(rargs, dict)
            if 'verbose' not in rargs.keys():
                rargs['verbose'] = False
            if 'no_weights' in rargs.keys() and \
                    rargs['no_weights'] != no_weights:
                raise ValueError('Keyword argument "no_weights" of ' +
                                 'bidirectional layer is in conflict with ' +
                                 'constructor argument "no_weights".')
            elif 'no_weights' not in rargs.keys():
                rargs['no_weights'] = no_weights

            self._forward_rnns.append(SimpleRNN(**rargs))
            self._backward_rnns.append(SimpleRNN(**rargs))

        ### Create output network.
        if mlp_args is not None:
            if 'verbose' not in mlp_args.keys():
                mlp_args['verbose'] = False
            if 'no_weights' in mlp_args.keys() and \
                    mlp_args['no_weights'] != no_weights:
                raise ValueError('Keyword argument "no_weights" of ' +
                                 'output MLP is in conflict with ' +
                                 'constructor argument "no_weights".')
            elif 'no_weights' not in mlp_args.keys():
                mlp_args['no_weights'] = no_weights

            self._out_mlp = MLP(**mlp_args)

        ### Set all interface attributes correctly.
        if self._out_mlp is None:
            self._has_fc_out = self._forward_rnns[-1].has_fc_out
            # We can't set the following attribute to true, as the output is
            # a concatenation of the outputs from two networks. Therefore, the
            # weights used two compute the outputs are at different locations
            # in the `param_shapes` list.
            self._mask_fc_out = False
            self._has_linear_out = self._forward_rnns[-1].has_linear_out
        else:
            self._has_fc_out = self._out_mlp.has_fc_out
            self._mask_fc_out = self._out_mlp.mask_fc_out
            self._has_linear_out = self._out_mlp.has_linear_out

        # Collect all internal net objects from which we need to collect
        # attributes.
        nets = []
        for i, fnet in enumerate(self._forward_rnns):
            bnet = self._backward_rnns[i]

            nets.append((fnet, 'forward_rnn', i))
            nets.append((bnet, 'backward_rnn', i))
        if self._out_mlp is not None:
            nets.append((self._out_mlp, 'out_mlp', -1))

        # Iterate over all nets to collect their attribute values.
        self._param_shapes = []
        self._param_shapes_meta = []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        for i, net_tup in enumerate(nets):
            net, net_type, net_id = net_tup
            # Note, it is important to convert lists into new object and not
            # just copy references!
            # Note, we have to adapt all references if `i > 0`.

            # Sanity check:
            if i == 0:
                cm_nw = net._context_mod_no_weights
            elif cm_nw != net._context_mod_no_weights:
                raise ValueError('Network expect that either all internal ' +
                                 'networks maintain their context-mod ' +
                                 'weights or non of them does!')

            ps_len_old = len(self._param_shapes)

            if net._internal_params is not None:
                if self._internal_params is None:
                    self._internal_params = nn.ParameterList()
                ip_len_old = len(self._internal_params)
                self._internal_params.extend( \
                    nn.ParameterList(net._internal_params))
            self._param_shapes.extend(list(net._param_shapes))
            for meta in net.param_shapes_meta:
                assert 'birnn_layer_type' not in meta.keys()
                assert 'birnn_layer_id' not in meta.keys()

                new_meta = dict(meta)
                new_meta['birnn_layer_type'] = net_type
                new_meta['birnn_layer_id'] = net_id
                if i > 0:
                    # FIXME We should properly adjust colliding `layer` IDs.
                    new_meta['layer'] = -1
                new_meta['index'] = meta['index'] + ip_len_old
                self._param_shapes_meta.append(new_meta)

            if net._hyper_shapes_learned is not None:
                if self._hyper_shapes_learned is None:
                    self._hyper_shapes_learned = []
                    self._hyper_shapes_learned_ref = []
                self._hyper_shapes_learned.extend( \
                    list(net._hyper_shapes_learned))
                for ref in net._hyper_shapes_learned_ref:
                    self._hyper_shapes_learned_ref.append(ref + ps_len_old)
            if net._hyper_shapes_distilled is not None:
                if self._hyper_shapes_distilled is None:
                    self._hyper_shapes_distilled = []
                self._hyper_shapes_distilled.extend( \
                    list(net._hyper_shapes_distilled))

            if self._has_bias is None:
                self._has_bias = net._has_bias
            elif self._has_bias != net._has_bias:
                self._has_bias = False
                # FIXME We should overwrite the getter and throw an error!
                warn('Some internally maintained networks use biases, ' +
                     'while others don\'t. Setting attribute "has_bias" to ' +
                     'False.')

            self._layer_weight_tensors.extend( \
                nn.ParameterList(net._layer_weight_tensors))
            self._layer_bias_vectors.extend( \
                nn.ParameterList(net._layer_bias_vectors))
            if net._batchnorm_layers is not None:
                if self._batchnorm_layers is None:
                    self._batchnorm_layers = nn.ModuleList()
                self._batchnorm_layers.extend( \
                    nn.ModuleList(net._batchnorm_layers))
            if net._context_mod_layers is not None:
                if self._context_mod_layers is None:
                    self._context_mod_layers = nn.ModuleList()
                self._context_mod_layers.extend( \
                    nn.ModuleList(net._context_mod_layers))

        self._is_properly_setup()

        ### Print user information.
        if verbose:
            print('Constructed Bidirectional RNN with %d weights.' \
                  % self.num_params)

    @property
    def preprocess_fct(self):
        """See constructor argument ``preprocess_fct``.

        :setter: The setter may only be called before the first call of the
            :meth:`forward` method.

        :type: func
        """
        return self._preprocess_fct

    @preprocess_fct.setter
    def preprocess_fct(self, value):
        if self._forward_called:
            raise RuntimeError('Attribute "preprocess_fct" cannot be ' +
                               'modified after method "forward" has been ' +
                               'called.')
        self._preprocess_fct = value

    @property
    def num_rec_layers(self):
        """See attribute :attr:`mnets.simple_rnn.SimpleRNN.num_rec_layers`.
        Total number of recurrent layer, where each bidirectional layer consists
        of at least two recurrent layers (forward and backward layer).

        :type: int
        """
        num_rec_layers = 0
        for net in self._forward_rnns + self._backward_rnns:
            num_rec_layers += net.num_rec_layers
        return num_rec_layers

    @property
    def use_lstm(self):
        """See attribute :attr:`mnets.simple_rnn.SimpleRNN.use_lstm`.

        :type: bool
        """
        use_lstm = self._forward_rnns[0].use_lstm
        for i in range(1, len(self._forward_rnns)):
            if self._forward_rnns[i].use_lstm != use_lstm:
                raise RuntimeError('Attribute "use_lstm" not applicable to ' +
                                   'this network as layers use mixed types ' +
                                   'of RNNs.')
        return use_lstm

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.
        """
        # SimpleRNNs should not have any distillation targets.
        for net in self._forward_rnns + self._backward_rnns:
            if net.distillation_targets is not None:
                raise RuntimeError()

        if self._out_mlp is not None:
            return self._out_mlp.distillation_targets()
        return None

    def forward(self, x, weights=None, distilled_params=None, condition=None,
                seq_lengths=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Note:
            If constructor argument ``preprocess_fct`` was set, then all
            inputs ``x`` are first processed by this function.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            weights (list or dict): See argument ``weights`` of method
                :meth:`mnets.mlp.MLP.forward`.
            distilled_params: Will only be passed to the underlying instance
                of class :class:`mnets.mlp.MLP`
            condition (int or dict, optional): If provided, then this argument
                will be passed as argument ``ckpt_id`` to the method
                :meth:`utils.context_mod_layer.ContextModLayer.forward`.

                When providing as dict, see argument ``condition`` of method
                :meth:`mnets.mlp.MLP.forward` for more details.
            seq_lengths (numpy.ndarray, optional): List of sequence
                lengths. The length of the list has to match the batch size of
                inputs ``x``. The entries will correspond to the unpadded
                sequence lengths. If this option is provided, then the
                bidirectional layers will reverse its input sequences according
                to the unpadded sequence lengths.

                Example:
                    ``x = [[a,b,0,0], [a,b,c,0]].T``. If
                    ``seq_lengths = [2, 3]`` if provided, then the reverse
                    sequences ``[[b,a,0,0], [c,b,a,0]].T`` are fed into the
                    first bidirectional layer (and similarly for all subsequent
                    bidirectional layers). Otherwise reverse sequences
                    ``[[0,0,b,a], [0,c,b,a]].T`` are used.

                Caution:
                    If this option is not provided but padded input sequences
                    are used, the output of a bidirectional layer will depent on
                    the padding. I.e., different padding lengths will lead to
                    different results.

        Returns:
            (torch.Tensor or tuple): Where the tuple is containing:

            - **output** (torch.Tensor): The output of the network.
            - **hidden** (list): ``None`` - not implemented yet.
        """
        # FIXME Delete warning below.
        if seq_lengths is None:
            warn('"seq_lengths" has not been provided to BiRNN.')

        if self._out_mlp is None:
            assert distilled_params is None

        ########################
        ### Parse condition ###
        #######################
        rnn_cmod_cond = None
        mlp_cond = None

        if condition is not None:
            if isinstance(condition, dict):
                if 'cmod_ckpt_id' in condition.keys():
                    rnn_cmod_cond = condition['cmod_ckpt_id']
                    mlp_cond = condition
            else:
                rnn_cmod_cond = condition
                mlp_cond = {'cmod_ckpt_id': condition}

        ########################################
        ### Extract-weights for each network ###
        ########################################
        forward_weights = [None] * len(self._forward_rnns)
        backward_weights = [None] * len(self._backward_rnns)
        mlp_weights = None

        n_cm = self._num_context_mod_shapes()
        int_weights = None
        cm_weights = None
        all_weights = None
        if weights is not None and isinstance(weights, dict):
            if 'internal_weights' in weights.keys():
                int_weights = weights['internal_weights']
            if 'mod_weights' in weights.keys():
                cm_weights = weights['mod_weights']

        elif weights is not None:
            if len(weights) == n_cm:
                cm_weights = weights
            else:
                assert len(weights) == len(self.param_shapes)
                all_weights = weights

        if weights is not None:
            # Collect all context-mod and internal weights if not explicitly
            # passed. Note, those will either be taken from `all_weights` or
            # have to exist internally.
            if n_cm > 0 and cm_weights is None:
                cm_weights = []
                for ii, meta in enumerate(self.param_shapes_meta):
                    if meta['name'].startswith('cm_'):
                        if all_weights is not None:
                            cm_weights.append(all_weights[ii])
                        else:
                            assert meta['index'] != -1
                            cm_weights.append( \
                                self.internal_params[meta['index']])
            if int_weights is None:
                int_weights = []
                for ii, meta in enumerate(self.param_shapes_meta):
                    if not meta['name'].startswith('cm_'):
                        if all_weights is not None:
                            int_weights.append(all_weights[ii])
                        else:
                            assert meta['index'] != -1
                            int_weights.append( \
                                self.internal_params[meta['index']])

            # Now that we have all context-mod and internal weights, we need to
            # distribute them across networks. Therefore, note that the order
            # in which they appear in `param_shapes` matches the order of
            # `cm_weights` and `int_weights`.
            cm_ind = 0
            int_ind = 0
            for ii, meta in enumerate(self.param_shapes_meta):
                net_type = meta['birnn_layer_type']
                net_id = meta['birnn_layer_id']

                if net_type == 'forward_rnn':
                    if forward_weights[net_id] is None:
                        forward_weights[net_id] = dict()
                    curr_weights = forward_weights[net_id]
                elif net_type == 'backward_rnn':
                    if backward_weights[net_id] is None:
                        backward_weights[net_id] = dict()
                    curr_weights = backward_weights[net_id]
                else:
                    assert net_type == 'out_mlp'
                    if mlp_weights is None:
                        mlp_weights = dict()
                    curr_weights = mlp_weights

                if meta['name'].startswith('cm_'):
                    if 'mod_weights' not in curr_weights.keys():
                        curr_weights['mod_weights'] = []
                    curr_weights['mod_weights'].append(cm_weights[cm_ind])
                    cm_ind += 1
                else:
                    if 'internal_weights' not in curr_weights.keys():
                        curr_weights['internal_weights'] = []
                    curr_weights['internal_weights'].append( \
                        int_weights[int_ind])
                    int_ind += 1

        #####################################
        ### Apply potential preprocessing ###
        #####################################
        self._forward_called = True
        if self._preprocess_fct is not None:
            x = self._preprocess_fct(x, seq_lengths=seq_lengths)

        ####################################
        ### Process bidirectional layers ###
        ####################################
        # Create reverse input sequence for backward network.
        if seq_lengths is not None:
            assert seq_lengths.size == x.shape[1]
        def revert_order(inp):
            if seq_lengths is None:
                return torch.flip(inp, [0])
            else:
                inp_back = torch.zeros_like(inp)
                for ii in range(seq_lengths.size):
                    inp_back[:int(seq_lengths[ii]),ii, :] = \
                        torch.flip(inp[:int(seq_lengths[ii]),ii, :], [0])
                return inp_back

        h = x

        for ll, fnet in enumerate(self._forward_rnns):
            bnet = self._backward_rnns[ll]

            # Revert inputs in time before processing them by the backward RNN.
            h_rev = revert_order(h)

            h_f = fnet.forward(h, weights=forward_weights[ll],
                condition=rnn_cmod_cond, return_hidden=False,
                return_hidden_int=False)
            h_b = bnet.forward(h_rev, weights=backward_weights[ll],
                condition=rnn_cmod_cond, return_hidden=False,
                return_hidden_int=False)

            # Revert outputs in time from the backward RNN before concatenation.
            # NOTE If `seq_lengths` are given, then this function will also set
            # the hidden timesteps corresponding to "padded timesteps" to zero.
            h_b = revert_order(h_b)

            # Set hidden states of `h_f` corresponding to padded timesteps to
            # zero to ensure consistency. Note, will only ever affect those
            # "padded timesteps".
            if seq_lengths is not None:
                for ii in range(seq_lengths.size):
                    h_f[:int(seq_lengths[ii]),ii, :] = 0

            h = torch.cat([h_f, h_b], dim=2)

        ##############################
        ### Compute network output ###
        ##############################
        if self._out_mlp is not None:
            #n_time, n_batch, n_feat = h.shape
            #h = h.view(n_time*n_batch, n_feat)
            h = self._out_mlp.forward(h, weights=mlp_weights,
                distilled_params=distilled_params, condition=mlp_cond)
            #h = h.view(n_time, n_batch, -1)

        return h

    def init_hh_weights_orthogonal(self):
        """Initialize hidden-to-hidden weights orthogonally.

        This method will call method
        :meth:`mnets.simple_rnn.SimpleRNN.init_hh_weights_orthogonal` of all
        internally maintained instances of class
        :class:`mnets.simple_rnn.SimpleRNN`.
        """
        for net in self._forward_rnns + self._backward_rnns:
            net.init_hh_weights_orthogonal()

    def get_cm_weights(self):
        """Get internal maintained weights that are associated with context-
        modulation.

        Returns:
            (list): List of weights from
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params` that
            are belonging to context-mod layers.
        """
        ret = []
        for i, meta in enumerate(self.param_shapes_meta):
            if not (meta['name'] == 'cm_shift' or meta['name'] == 'cm_scale'):
                continue
            if meta['index'] != -1:
                ret.append(self.internal_params[meta['index']])
        return ret

    def get_non_cm_weights(self):
        """Get internal weights that are not associated with context-modulation.

        Returns:
            (list): List of weights from
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params` that
            are not belonging to context-mod layers.
        """
        n_cm = self._num_context_mod_shapes()
        if n_cm == 0:
            return self.internal_params
        else:
            ret = []
            for i, meta in enumerate(self.param_shapes_meta):
                if meta['name'] == 'cm_shift' or meta['name'] == 'cm_scale':
                    continue
                if meta['index'] != -1:
                    ret.append(self.internal_params[meta['index']])
            return ret

if __name__ == '__main__':
    pass


