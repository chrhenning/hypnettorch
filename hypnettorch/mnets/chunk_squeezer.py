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
# @title          :mnets/chunk_squeezer.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/26/2020
# @version        :1.0
# @python_version :3.6.9
"""
MLP with input dimensionality reduction via chunking
----------------------------------------------------

The module :mod:`mnets.chunk_squeezer` contains a network implementation that
expects high-dimensional input (e.g., the output of a hypernetwork).

Since the processing of very high-dimensional inputs via an MLP might lead
to an extremely large network size, this network splits the input into chunks
and processes each chunk individually (conditioned on a learned chunk embedding)
to readuce its dimensionality. The squeezed chunks are subsequently concatenated
and send through a second MLP to compute the output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.mnets.mlp import MLP
from hypnettorch.utils.torch_utils import init_params

class ChunkSqueezer(nn.Module, MainNetInterface):
    """An MLP that first reduces the dimensionality of its inputs.

    The input dimensionality ``n_in`` is first reduced by a `reducer` network
    (which is an instance of class :class:`mnets.mlp.MLP`) using a chunking
    strategy. The reduced input will be then passed to the actual `network`
    (which is another instance of :class:`mnets.mlp.MLP`) to compute an output.
    
    Args:
        n_in (int): Input dimensionality.
        n_out (int): Number of output neurons.
        inp_chunk_dim (int): The input (dimensionality ``n_in``) will be split
            into chunks of size ``inp_chunk_dim``. Thus, there will be
            ``np.ceil(n_in/inp_chunk_dim)`` input chunks that are individually
            squeezed through the `reducer` network.

            Note:
                If the last chunk chunk might be zero-padded.
        out_chunk_dim (int): The output size of the `reducer` network. The
            input size of the actual `network` is then
            ``np.ceil(n_in/inp_chunk_dim) * out_chunk_dim``.
        cemb_size (int): The `reducer` network processes every chunk
            individually. In order to do so, it needs to know which chunk it is
            processing. Therefore, it is conditioned on a learned chunk
            embedding (there will be ``np.ceil(n_in/inp_chunk_dim)`` chunk
            embeddings). The dimensionality of these chunk embeddings is
            dertermined by this argument.
        cemb_init_std (float): Standard deviation used for the normal
            initialization of the chunk embeddings.
        red_layers (list or tuple): The architecture of the `reducer` network.
            See argument ``hidden_layers`` of class :class:`mnets.mlp.MLP`.
        net_layers (list or tuple): The architecture of the actual `network`.
            See argument ``hidden_layers`` of class :class:`mnets.mlp.MLP`. 
        activation_fn: The nonlinearity used in hidden layers. If ``None``, no
            nonlinearity will be applied.
        use_bias: Will be passed as option ``use_bias`` to the underlying MLPs
            (see :class:`mnets.mlp.MLP`).
        dynamic_biases (list, optional): This option determines the hidden
            layers of the `reducer` networks that receive the chunk embedding as
            dynamic biases. It is a list of indexes with the first hidden layer
            having index 0 and the output of the `reducer` would have index
            ``len(red_layers)``. The chunk embeddings will be transformed
            through a fully connected layer (no bias) and then added as
            "dynamic" bias to the output of the corresponding hidden layer.

            Note:
                If left unspecified, the chunk embeddings will just be another
                input to the `reducer` network.
        no_weights (bool): If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.
        init_weights (optional): This option is for convinience reasons.
            The option expects a list of parameter values that are used to
            initialize the network weights. As such, it provides a
            convinient way of initializing a network with a weight draw
            produced by the hypernetwork.

            Note, internal weights (see 
            :attr:`mnets.mnet_interface.MainNetInterface.weights`) will be
            affected by this argument only.
        dropout_rate (float): Will be passed as option ``dropout_rate`` to the
            underlying MLPs (see :class:`mnets.mlp.MLP`).
        use_spectral_norm (bool): Will be passed as option ``use_spectral_norm``
            to the underlying MLPs (see :class:`mnets.mlp.MLP`).
        use_batch_norm (bool): Will be passed as option ``use_batch_norm``
            to the underlying MLPs (see :class:`mnets.mlp.MLP`).
        bn_track_stats (bool): Will be passed as option ``bn_track_stats``
            to the underlying MLPs (see :class:`mnets.mlp.MLP`).
        distill_bn_stats (bool): Will be passed as option ``distill_bn_stats``
            to the underlying MLPs (see :class:`mnets.mlp.MLP`).
    """
    def __init__(self, n_in, n_out=1, inp_chunk_dim=100, out_chunk_dim=10,
                 cemb_size=8, cemb_init_std=1., red_layers=(10, 10),
                 net_layers=(10, 10), activation_fn=torch.nn.ReLU(),
                 use_bias=True, dynamic_biases=None, no_weights=False,
                 init_weights=None, dropout_rate=-1, use_spectral_norm=False,
                 use_batch_norm=False, bn_track_stats=True,
                 distill_bn_stats=False, verbose=True):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        self._n_in = n_in
        self._n_out = n_out
        self._inp_chunk_dim = inp_chunk_dim
        self._out_chunk_dim = out_chunk_dim
        self._cemb_size = cemb_size
        self._a_fun = activation_fn
        self._no_weights = no_weights

        self._has_bias = use_bias
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True
        self._has_linear_out = True # Ensure that `out_fn` is `None`!

        self._param_shapes = []
        #self._param_shapes_meta = [] # TODO implement!
        self._weights = None if no_weights else nn.ParameterList()
        self._hyper_shapes_learned = None if not no_weights else []
        #self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
        #    is None else [] # TODO implement.

        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        self._context_mod_layers = None
        self._batchnorm_layers = nn.ModuleList() if use_batch_norm else None

        #################################
        ### Generate Chunk Embeddings ###
        #################################
        self._num_cembs = int(np.ceil(n_in / inp_chunk_dim))
        last_chunk_size = n_in % inp_chunk_dim
        if last_chunk_size != 0:
            self._pad = inp_chunk_dim - last_chunk_size
        else:
            self._pad = -1

        cemb_shape = [self._num_cembs, cemb_size]
        self._param_shapes.append(cemb_shape)
        if no_weights:
            self._cembs = None
            self._hyper_shapes_learned.append(cemb_shape)
        else:
            self._cembs = nn.Parameter(data=torch.Tensor(*cemb_shape),
                                       requires_grad=True)
            nn.init.normal_(self._cembs, mean=0., std=cemb_init_std)

            self._weights.append(self._cembs)

        ############################
        ### Setup Dynamic Biases ###
        ############################
        self._has_dyn_bias = None
        if dynamic_biases is not None:
            assert np.all(np.array(dynamic_biases) >= 0) and \
                   np.all(np.array(dynamic_biases) < len(red_layers) + 1)
            dynamic_biases = np.sort(np.unique(dynamic_biases))

            # For each layer in the `reducer`, where we want to apply a dynamic
            # bias, we have to create a weight matrix for a corresponding
            # linear layer (we just ignore)
            self._dyn_bias_weights = nn.ModuleList()
            self._has_dyn_bias = []

            for i in range(len(red_layers) + 1):
                if i in dynamic_biases:
                    self._has_dyn_bias.append(True)

                    trgt_dim = out_chunk_dim
                    if i < len(red_layers):
                        trgt_dim = red_layers[i]
                    trgt_shape = [trgt_dim, cemb_size]

                    self._param_shapes.append(trgt_shape)
                    if not no_weights:
                        self._dyn_bias_weights.append(None)
                        self._hyper_shapes_learned.append(trgt_shape)
                    else:
                        self._dyn_bias_weights.append(nn.Parameter( \
                            torch.Tensor(*trgt_shape), requires_grad=True))
                        self._weights.append(self._dyn_bias_weights[-1])

                        init_params(self._dyn_bias_weights[-1])

                        self._layer_weight_tensors.append( \
                            self._dyn_bias_weights[-1])
                        self._layer_bias_vectors.append(None)
                else:
                    self._has_dyn_bias.append(False)
                    self._dyn_bias_weights.append(None)

        ################################
        ### Create `Reducer` Network ###
        ################################
        red_inp_dim = inp_chunk_dim + \
            (cemb_size if dynamic_biases is None else 0)
        self._reducer = MLP(n_in=red_inp_dim, n_out=out_chunk_dim,
            hidden_layers=red_layers, activation_fn=activation_fn,
            use_bias=use_bias, no_weights=no_weights, init_weights=None,
            dropout_rate=dropout_rate, use_spectral_norm=use_spectral_norm,
            use_batch_norm=use_batch_norm, bn_track_stats=bn_track_stats,
            distill_bn_stats=distill_bn_stats,
            # We use context modulation to realize dynamic biases, since they
            # allow a different modulation per sample in the input mini-batch.
            # Hence, we can process several chunks in parallel with the reducer
            # network.
            use_context_mod=not dynamic_biases is None,
            context_mod_inputs=False, no_last_layer_context_mod=False,
            context_mod_no_weights=True, context_mod_post_activation=False,
            context_mod_gain_offset=False, context_mod_gain_softplus=False,
            out_fn=None, verbose=True)

        if dynamic_biases is not None:
            # FIXME We have to extract the param shapes from
            # `self._reducer.param_shapes`, as well as from
            # `self._reducer._hyper_shapes_learned` that belong to context-mod
            # layers. We may not add them to our own `param_shapes` attribute,
            # as these are not parameters (due to our misuse of the context-mod
            # layers).
            # Note, in the `forward` method, we need to supply context-mod
            # weights for all reducer networks, independent on whether they have
            # a dynamic bias or not. We can do so, by providing constant ones
            # for all gains and constance zero-shift for all layers without
            # dynamic biases (note, we need to ensure the correct batch dim!).
            raise NotImplementedError('Dynamic biases are not yet implemented!')

        assert self._reducer._context_mod_layers is None

        ### Overtake all attributes from the underlying MLP.
        for s in self._reducer.param_shapes:
            self._param_shapes.append(s)
        if no_weights:
            for s in self._reducer._hyper_shapes_learned:
                self._hyper_shapes_learned.append(s)
        else:
            for p in self._reducer._weights:
                self._weights.append(p)

        for p in self._reducer._layer_weight_tensors:
            self._layer_weight_tensors.append(p)
        for p in self._reducer._layer_bias_vectors:
            self._layer_bias_vectors.append(p)

        if use_batch_norm:
            for p in self._reducer._batchnorm_layers:
                self._batchnorm_layers.append(p)

        if self._reducer._hyper_shapes_distilled is not None:
            self._hyper_shapes_distilled = []
            for s in self._reducer._hyper_shapes_distilled:
                self._hyper_shapes_distilled.append(s)

        ###############################
        ### Create Actual `Network` ###
        ###############################
        net_inp_dim = out_chunk_dim * self._num_cembs
        self._network = MLP(n_in=net_inp_dim, n_out=n_out,
            hidden_layers=net_layers, activation_fn=activation_fn,
            use_bias=use_bias, no_weights=no_weights, init_weights=None,
            dropout_rate=dropout_rate, use_spectral_norm=use_spectral_norm,
            use_batch_norm=use_batch_norm, bn_track_stats=bn_track_stats,
            distill_bn_stats=distill_bn_stats, use_context_mod=False,
            out_fn=None, verbose=True)

        ### Overtake all attributes from the underlying MLP.
        for s in self._network.param_shapes:
            self._param_shapes.append(s)
        if no_weights:
            for s in self._network._hyper_shapes_learned:
                self._hyper_shapes_learned.append(s)
        else:
            for p in self._network._weights:
                self._weights.append(p)

        for p in self._network._layer_weight_tensors:
            self._layer_weight_tensors.append(p)
        for p in self._network._layer_bias_vectors:
            self._layer_bias_vectors.append(p)

        if use_batch_norm:
            for p in self._network._batchnorm_layers:
                self._batchnorm_layers.append(p)

        if self._hyper_shapes_distilled is not None:
            assert self._network._hyper_shapes_distilled is not None
            for s in self._network._hyper_shapes_distilled:
                self._hyper_shapes_distilled.append(s)

        #####################################
        ### Takeover given Initialization ###
        #####################################
        if init_weights is not None:
            assert len(init_weights) == len(self._weights)
            for i in range(len(init_weights)):
                assert np.all(np.equal(list(init_weights[i].shape),
                                       self._param_shapes[i]))
                self._weights[i].data = init_weights[i]

        ######################
        ### Finalize Setup ###
        ######################
        num_weights = MainNetInterface.shapes_to_num_weights(self.param_shapes)
        print('Constructed MLP that processes dimensionality reduced inputs ' +
              'through chunking. The network has a total of %d weights.' %
              num_weights)

        self._is_properly_setup()

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This method will return the distillation targets from the 2 underlying
        networks, see method :meth:`mnets.mlp.MLP.distillation_targets`.

        Returns:
            The target tensors corresponding to the shapes specified in
            attribute :attr:`hyper_shapes_distilled`.
        """
        if self.hyper_shapes_distilled is None:
            return None

        ret = self._reducer.distillation_targets + \
            self._network.distillation_targets

        return ret

    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            distilled_params: Will be split and passed as distillation targets
                to the underying instances of class :class:`mnets.mlp.MLP` if
                specified.
            condition (optional, int or dict): Will be passed to the underlying
                instances of class :class:`mnets.mlp.MLP`.

        Returns:
            The output :math:`y` of the network.
        """
        if self._no_weights and weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        if weights is None:
            weights = self._weights
        else:
            assert len(weights) == len(self.param_shapes)
            for i, s in enumerate(self.param_shapes):
                assert np.all(np.equal(s, list(weights[i].shape)))

        #########################################
        ### Extract parameters from `weights` ###
        #########################################
        cembs = weights[0]
        w_ind = 1

        if self._has_dyn_bias is not None:
            w_ind_new = w_ind+len(self._dyn_bias_weights)
            dyn_bias_weights = weights[w_ind:w_ind_new]
            w_ind = w_ind_new

            # TODO use `dyn_bias_weights` to construct weights for context-mod
            # layers.
            raise NotImplementedError

        w_ind_new = w_ind+len(self._reducer.param_shapes)
        red_weights = weights[w_ind:w_ind_new]
        w_ind = w_ind_new

        w_ind_new = w_ind+len(self._network.param_shapes)
        net_weights = weights[w_ind:w_ind_new]
        w_ind = w_ind_new

        red_distilled_params = None
        net_distilled_params = None
        if distilled_params is not None:
            if self.hyper_shapes_distilled is None:
                raise ValueError('Argument "distilled_params" can only be ' +
                                 'provided if the return value of ' +
                                 'method "distillation_targets()" is not None.')

            assert len(distilled_params) == len(self.hyper_shapes_distilled)
            red_distilled_params = \
                distilled_params[:len(self._reducer.hyper_shapes_distilled)]
            net_distilled_params = \
                distilled_params[len(self._reducer.hyper_shapes_distilled):]

        ###########################
        ### Chunk network input ###
        ###########################
        assert x.shape[1] == self._n_in

        if self._pad != -1:
            x = F.pad(x, (0, self._pad))
            assert x.shape[1] % self._out_chunk_dim == 0

        batch_size = x.shape[0]
        # We now split the input `x` into chunks and convert them into
        # separate samples, i.e., the `batch_size` will be multiplied by the
        # number of chunks.
        # So, we parallel process a huge batch with a small network rather than
        # processing a huge input with a huge network.

        chunks = torch.split(x, self._inp_chunk_dim, dim=1)
        # Concatenate the chunks along the batch dimension.
        chunks = torch.cat(chunks, dim=0)
        if self._has_dyn_bias is not None:
            raise NotImplementedError()
        else:
            # Within a chunk the same chunk embedding is used.
            cembs = torch.split(cembs, 1, dim=0)
            cembs = [emb.expand(batch_size, -1) for emb in cembs]
            cembs = torch.cat(cembs, dim=0)

            chunks = torch.cat([chunks, cembs], dim=1)

        ###################################
        ### Reduce input dimensionality ###
        ###################################
        if self._has_dyn_bias is not None:
            # TODO pass context-mod weights to `reducer`.
            raise NotImplementedError()
        chunks = self._reducer.forward(chunks, weights=red_weights,
            distilled_params=red_distilled_params, condition=condition)

        ### Reformat `reducer` output into the input of the actual `network`.
        chunks = torch.split(chunks, batch_size, dim=0)
        net_input = torch.cat(chunks, dim=1)
        assert net_input.shape[0] == batch_size

        ###############################
        ### Compute network output ###
        ##############################
        return self._network.forward(net_input, weights=net_weights,
            distilled_params=net_distilled_params, condition=condition)

if __name__ == '__main__':
    pass


