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
# @title          :hnets/mlp_hnet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :04/14/2020
# @version        :1.0
# @python_version :3.6.10
"""
MLP - Hypernetwork
------------------

The module :mod:`hnets.mlp_hnet` contains a fully-connected hypernetwork
(also termed `full hypernet`).

This type of hypernetwork represents one of the most simplistic architectural
choices to realize a weight generator. An embedding input, which may consists of
conditional and unconditional parts (for instance, in the case of
`task-conditioned hypernetwork <https://arxiv.org/abs/1906.00695>`__ the
conditional input will be a task embedding) is mapped via a series of fully-
connected layers onto a final hidden representation. Then a linear
fully-connected output layer per is used to produce the target weights, output
tensors with shapes specified via the target shapes (see
:attr:`hnets.hnet_interface.HyperNetInterface.target_shapes`).

If no hidden layers are used, then this resembles a simplistic linear
hypernetwork, where the input embeddings are linearly mapped onto target
weights.
"""
from collections import defaultdict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn

from hypnettorch.hnets.hnet_interface import HyperNetInterface
from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils import init_utils as iutils

class HMLP(nn.Module, HyperNetInterface):
    """Implementation of a `full hypernet`.

    The network will consist of several hidden layers and a final linear output
    layer that produces all weight matrices/bias-vectors the network has to
    produce.

    The network allows to maintain a set of embeddings internally that can be
    used as conditional input.

    Args:
        target_shapes (list): List of lists of intergers, i.e., a list of tensor
            shapes. Those will be the shapes of the output weights produced by
            the hypernetwork. For each entry in this list, a separate output
            layer will be instantiated.
        uncond_in_size (int): The size of unconditional inputs (for instance,
            noise).
        cond_in_size (int): The size of conditional input embeddings.

            Note, if ``no_cond_weights`` is ``False``, those embeddings will be
            maintained internally.
        layers (list or tuple): List of integers denoteing the sizes of each
            hidden layer. If empty, no hidden layers will be produced.
        verbose (bool): Whether network information should be printed during
            network creation.
        activation_fn (func): The activation function to be used for hidden
            activations. For instance, an instance of class
            :class:`torch.nn.ReLU`.
        use_bias (bool): Whether the fully-connected layers that make up this
            network should have bias vectors.
        no_uncond_weights (bool): If ``True``, unconditional weights are not
            maintained internally and instead expected to be produced
            externally and passed to the :meth:`forward`.
        no_cond_weights (bool): If ``True``, conditional embeddings are assumed
            to be maintained externally. Otherwise, option ``num_cond_embs``
            has to be properly set, which will determine the number of
            embeddings that are internally maintained.
        num_cond_embs (int): Number of conditional embeddings to be internally
            maintained. Only used if option ``no_cond_weights`` is ``False``.

            Note:
                Embeddings will be initialized with a normal distribution using
                zero mean and unit variance.
        dropout_rate (float): If ``-1``, no dropout will be applied. Otherwise a
            number between 0 and 1 is expected, denoting the dropout rate of
            hidden layers.
        use_spectral_norm (bool): Use spectral normalization for training.
        use_batch_norm (bool): Whether batch normalization should be used. Will
            be applied before the activation function in all hidden layers.

            Note:
                Batch norm only makes sense if the hypernetwork is envoked with
                batch sizes greater than 1 during training.
    """
    def __init__(self, target_shapes, uncond_in_size=0, cond_in_size=8,
                 layers=(100, 100), verbose=True, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_uncond_weights=False, no_cond_weights=False,
                 num_cond_embs=1, dropout_rate=-1, use_spectral_norm=False,
                 use_batch_norm=False):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        HyperNetInterface.__init__(self)

        if use_spectral_norm:
            raise NotImplementedError('Spectral normalization not yet ' +
                                      'implemented for this hypernetwork type.')

        assert len(target_shapes) > 0
        if cond_in_size == 0 and num_cond_embs > 0:
            warn('Requested that conditional weights are managed, but ' +
                 'conditional input size is zero! Setting "num_cond_embs" to ' +
                 'zero.')
            num_cond_embs = 0
        elif not no_cond_weights and num_cond_embs == 0 and cond_in_size > 0:
            warn('Requested that conditional weights are internally ' +
                 'maintained, but "num_cond_embs" is zero.')
        # Do we maintain conditional weights internally?
        has_int_cond_weights = not no_cond_weights and num_cond_embs > 0
        # Do we expect external conditional weights?
        has_ext_cond_weights = no_cond_weights and num_cond_embs > 0

        ### Make constructor arguments internally available ###
        self._uncond_in_size = uncond_in_size
        self._cond_in_size = cond_in_size
        self._layers = layers
        self._act_fn = activation_fn
        self._no_uncond_weights = no_uncond_weights
        self._no_cond_weights = no_cond_weights
        self._num_cond_embs = num_cond_embs
        self._dropout_rate = dropout_rate
        self._use_spectral_norm = use_spectral_norm
        self._use_batch_norm = use_batch_norm

        ### Setup attributes required by interface ###
        self._target_shapes = target_shapes
        self._num_known_conds = self._num_cond_embs
        self._unconditional_param_shapes_ref = []

        self._has_bias = use_bias
        self._has_fc_out = True
        self._mask_fc_out = True
        self._has_linear_out = True

        self._param_shapes = []
        self._param_shapes_meta = []
        self._internal_params = None if no_uncond_weights and \
            has_int_cond_weights else nn.ParameterList()
        self._hyper_shapes_learned = None \
            if not no_uncond_weights and has_ext_cond_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        self._dropout = None
        if dropout_rate != -1:
            assert dropout_rate > 0 and dropout_rate < 1
            self._dropout = nn.Dropout(dropout_rate)

        ### Create conditional weights ###
        for _ in range(num_cond_embs):
            assert cond_in_size > 0
            if not no_cond_weights:
                self._internal_params.append(nn.Parameter( \
                    data=torch.Tensor(cond_in_size), requires_grad=True))
                torch.nn.init.normal_(self._internal_params[-1], mean=0.,
                                      std=1.)
            else:
                self._hyper_shapes_learned.append([cond_in_size])
                self._hyper_shapes_learned_ref.append(len(self.param_shapes))

            self._param_shapes.append([cond_in_size])
            # Embeddings belong to the input, so we just assign them all to
            # "layer" 0.
            self._param_shapes_meta.append({
                'name': 'embedding',
                'index': -1 if no_cond_weights else \
                    len(self._internal_params)-1,
                'layer': 0
            })

        ### Create batch-norm layers ###
        # We just use even numbers starting from 2 as layer indices for
        # batchnorm layers.
        if use_batch_norm:
            self._add_batchnorm_layers(layers, no_uncond_weights,
                bn_layers=list(range(2, 2*len(layers)+1, 2)),
                distill_bn_stats=False, bn_track_stats=True)

        ### Create fully-connected hidden-layers ###
        in_size = uncond_in_size + cond_in_size
        if len(layers) > 0:
            # We use odd numbers starting at 1 as layer indices for hidden
            # layers.
            self._add_fc_layers([in_size, *layers[:-1]], layers,
                no_uncond_weights, fc_layers=list(range(1, 2*len(layers), 2)))
            hidden_size = layers[-1]
        else:
            hidden_size = in_size

        ### Create fully-connected output-layers ###
        # Note, technically there is no difference between having a separate
        # fully-connected layer per target shape or a single fully-connected
        # layer producing all weights at once (in any case, each output is
        # connceted to all hidden units).
        # I guess it is more computationally efficient to have one output layer
        # and then split the output according to the target shapes.
        self._add_fc_layers([hidden_size], [self.num_outputs],
                            no_uncond_weights, fc_layers=[2*len(layers)+1])

        ### Finalize construction ###
        # All parameters are unconditional except the embeddings created at the
        # very beginning.
        self._unconditional_param_shapes_ref = \
            list(range(num_cond_embs, len(self.param_shapes)))

        self._is_properly_setup()

        if verbose:
            print('Created MLP Hypernet.')
            print(self)

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed'):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
            condition (int, optional): This argument will be passed as argument
                ``stats_id`` to the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if batch
                normalization is used.

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """
        uncond_input, cond_input, uncond_weights, _ = \
            self._preprocess_forward_args(uncond_input=uncond_input,
                cond_input=cond_input, cond_id=cond_id, weights=weights,
                distilled_params=distilled_params, condition=condition,
                ret_format=ret_format)

        ### Prepare hypernet input ###
        assert self._uncond_in_size == 0 or uncond_input is not None
        assert self._cond_in_size == 0 or cond_input is not None
        if uncond_input is not None:
            assert len(uncond_input.shape) == 2 and \
                   uncond_input.shape[1] == self._uncond_in_size
            h = uncond_input
        if cond_input is not None:
            assert len(cond_input.shape) == 2 and \
                   cond_input.shape[1] == self._cond_in_size
            h = cond_input
        if uncond_input is not None and cond_input is not None:
            h = torch.cat([uncond_input, cond_input], dim=1)

        ### Extract layer weights ###
        bn_scales = []
        bn_shifts = []
        fc_weights = []
        fc_biases = []

        assert len(uncond_weights) == len(self.unconditional_param_shapes_ref)
        for i, idx in enumerate(self.unconditional_param_shapes_ref):
            meta = self.param_shapes_meta[idx]

            if meta['name'] == 'bn_scale':
                bn_scales.append(uncond_weights[i])
            elif meta['name'] == 'bn_shift':
                bn_shifts.append(uncond_weights[i])
            elif meta['name'] == 'weight':
                fc_weights.append(uncond_weights[i])
            else:
                assert meta['name'] == 'bias'
                fc_biases.append(uncond_weights[i])

        if not self.has_bias:
            assert len(fc_biases) == 0
            fc_biases = [None] * len(fc_weights)

        if self._use_batch_norm:
            assert len(bn_scales) == len(fc_weights) - 1

        ### Process inputs through network ###
        for i in range(len(fc_weights)):
            last_layer = i == (len(fc_weights) - 1)

            h = F.linear(h, fc_weights[i], bias=fc_biases[i])

            if not last_layer:
                # Batch-norm
                if self._use_batch_norm:
                    h = self.batchnorm_layers[i].forward(h, running_mean=None,
                        running_var=None, weight=bn_scales[i],
                        bias=bn_shifts[i], stats_id=condition)

                # Dropout
                if self._dropout_rate != -1:
                    h = self._dropout(h)

                # Non-linearity
                if self._act_fn is not None:
                    h = self._act_fn(h)

        ### Split output into target shapes ###
        ret = self._flat_to_ret_format(h, ret_format)

        return ret

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This network does not have any distillation targets.

        Returns:
            ``None``
        """
        return None

    def apply_hyperfan_init(self, method='in', use_xavier=False,
                            uncond_var=1., cond_var=1., mnet=None,
                            w_val=None, w_var=None, b_val=None, b_var=None):
        r"""Initialize the network using `hyperfan init`.

        Hyperfan initialization was developed in the following paper for this
        kind of hypernetwork

            "Principled Weight Initialization for Hypernetworks"
            https://openreview.net/forum?id=H1lma24tPB

        The initialization is based on the following idea: When the main network
        would be initialized using Xavier or Kaiming init, then variance of
        activations (fan-in) or gradients (fan-out) would be preserved by using
        a proper variance for the initial weight distribution (assuming certain
        assumptions hold at initialization, which are different for Xavier and
        Kaiming).

        When using this kind of initializations in the hypernetwork, then the
        variance of the initial main net weight distribution would simply equal
        the variance of the input embeddings (which can lead to exploding
        activations, e.g., for fan-in inits).

        The above mentioned paper proposes a quick fix for the type of hypernet
        that resembles the simple MLP hnet implemented in this class, i.e.,
        which have a separate output head per weight tensor in the main network.

        Assuming that input embeddings are initialized with a certain variance
        (e.g., 1) and we use Xavier or Kaiming init for the hypernet, then the
        variance of the last hidden activation will also be 1.

        Then, we can modify the variance of the weights of each output head in
        the hypernet to obtain the same variance per main net weight tensor that
        we would typically obtain when applying Xavier or Kaiming to the main
        network directly.

        Note:
            If ``mnet`` is not provided or the corresponding attribute
            :attr:`mnets.mnet_interface.MainNetInterface.param_shapes_meta` is
            not implemented, then this method assumes that 1D target tensors
            (cf. constructor argument ``target_shapes``) represent bias vectors
            in the main network.

        Note:
            To compute the hyperfan-out initialization of bias vectors, we need
            access to the fan-in of the layer, which we can only compute based
            on the corresponding weight tensor in the same layer. This is only
            possible if ``mnet`` is provided. Otherwise, the following
            heuristic is applied. We assume that the shape directly preceding
            a bias shape in the constructor argument ``target_shapes`` is the
            corresponding weight tensor.

        Note:
            All hypernet inputs are assumed to be zero-mean random variables.

        **Variance of the hypernet input**

        In general, the input to the hypernetwork can be a concatenation of
        multiple embeddings (see description of arguments ``uncond_var`` and
        ``cond_var``).

        Let's denote the complete hypernetwork input by
        :math:`\mathbf{x} \in \mathbb{R}^n`, which consists of a conditional
        embedding :math:`\mathbf{e} \in \mathbb{R}^{n_e}` and an unconditional
        input :math:`\mathbf{c} \in \mathbb{R}^{n_c}`, i.e.,

        .. math::

            \mathbf{x} = \begin{bmatrix} \
            \mathbf{e} \\ \
            \mathbf{c} \
            \end{bmatrix}

        We simply define the variance of an input :math:`\text{Var}(x_j)` as
        the weighted average of the individual variances, i.e.,

        .. math::

            \text{Var}(x_j) \equiv \frac{n_e}{n_e+n_c} \text{Var}(e) + \
                \frac{n_c}{n_e+n_c} \text{Var}(c)

        To see that this is correct, consider a linear layer
        :math:`\mathbf{y} = W \mathbf{x}` or

        .. math::

            y_i &= \sum_j w_{ij} x_j \\ \
                &= \sum_{j=1}^{n_e} w_{ij} e_j + \
                   \sum_{j=n_e+1}^{n_e+n_c} w_{ij} c_{j-n_e}

        Hence, we can compute the variance of :math:`y_i` as follows (assuming
        the typical Xavier assumptions):

        .. math::

            \text{Var}(y) &= n_e \text{Var}(w) \text{Var}(e) + \
                             n_c \text{Var}(w) \text{Var}(c) \\ \
                          &= \frac{n_e}{n_e+n_c} \text{Var}(e) + \
                             \frac{n_c}{n_e+n_c} \text{Var}(c)

        Note, that Xavier would have initialized :math:`W` using
        :math:`\text{Var}(w) = \frac{1}{n} = \frac{1}{n_e+n_c}`.

        Args:
            method (str): The type of initialization that should be applied.
                Possible options are:

                - ``'in'``: Use `Hyperfan-in`.
                - ``'out'``: Use `Hyperfan-out`.
                - ``'harmonic'``: Use the harmonic mean of the `Hyperfan-in` and
                  `Hyperfan-out` init.
            use_xavier (bool): Whether Kaiming (``False``) or Xavier (``True``)
                init should be used.
            uncond_var (float): The variance of unconditional embeddings. This
                value is only taken into consideration if ``uncond_in_size > 0``
                (cf. constructor arguments).
            cond_var (float): The initial variance of conditional embeddings.
                This value is only taken into consideration if
                ``cond_in_size > 0`` (cf. constructor arguments).
            mnet (mnets.mnet_interface.MainNetInterface, optional): If
                applicable, the user should provide the main (or target)
                network, whose weights are generated by this hypernetwork. The
                ``mnet`` instance is used to extract valuable information that
                improve the initialization result. If provided, it is assumed
                that ``target_shapes`` (cf. constructor arguments) corresponds
                either to
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes` or
                :attr:`mnets.mnet_interface.MainNetInterface.hyper_shapes_learned`.
            w_val (list or dict, optional): The mean of the distribution with
                which output head weight matrices are initialized. Note, each
                weight tensor prescribed by
                :attr:`hnets.hnet_interface.HyperNetInterface.target_shapes` is
                produced via an independent linear output head.

                One may either specify a list of numbers having the same length
                as :attr:`hnets.hnet_interface.HyperNetInterface.target_shapes`
                or specify a dictionary which may have as keys the tensor names
                occurring in
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes_meta`
                and the corresponding mean value for the weight matrices of all
                output heads producing this type of tensor.
                If a list is provided, entries may be ``None`` and if a
                dictionary is provided, not all types of parameter tensors need
                to be specified. For tensors, for which no value is specified,
                the default value will be used. The default values for tensor
                types ``'weight'`` and ``'bias'`` are calculated based on the
                proposed hyperfan-initialization. For other tensor types the
                actual hypernet outputs should be drawn from the following
                distributions

                  - ``'bn_scale'``: :math:`w \sim \delta(w - 1)`
                  - ``'bn_shift'``: :math:`w \sim \delta(w)`
                  - ``'cm_scale'``: :math:`w \sim \delta(w - 1)`
                  - ``'cm_shift'``: :math:`w \sim \delta(w)`
                  - ``'embedding'``: :math:`w \sim \mathcal{N}(0, 1)`

                Which would correspond to the following passed arguments

                .. code-block:: python

                    w_val = {
                        'bn_scale': 0,
                        'bn_shift': 0,
                        'cm_scale': 0,
                        'cm_shift': 0,
                        'embedding': 0
                    }
                    w_var = {
                        'bn_scale': 0,
                        'bn_shift': 0,
                        'cm_scale': 0,
                        'cm_shift': 0,
                        'embedding': 0
                    }
                    b_val = {
                        'bn_scale': 1,
                        'bn_shift': 0,
                        'cm_scale': 1,
                        'cm_shift': 0,
                        'embedding': 0
                    }
                    b_var = {
                        'bn_scale': 0,
                        'bn_shift': 0,
                        'cm_scale': 0,
                        'cm_shift': 0,
                        'embedding': 1
                    }
            w_var (list or dict, optional): The variance of the distribution
                with which output head weight matrices are initialized. Variance
                values of zero means that weights are set to a constant defined
                by ``w_val``. See description of argument ``w_val`` for more
                details.
            b_val (list or dict, optional): The mean of the distribution
                with which output head bias vectors are initialized.
                See description of argument ``w_val`` for more details.
            b_var (list or dict, optional): The variance of the distribution
                with which output head bias vectors are initialized.
                See description of argument ``w_val`` for more details.
        """
        if method not in ['in', 'out', 'harmonic']:
            raise ValueError('Invalid value "%s" for argument "method".' %
                             method)
        if self.unconditional_params is None:
            assert self._no_uncond_weights
            raise ValueError('Hypernet without internal weights can\'t be ' +
                             'initialized.')

        ### Extract meta-information about target shapes ###
        meta = None
        if mnet is not None:
            assert isinstance(mnet, MainNetInterface)

            try:
                meta = mnet.param_shapes_meta
            except:
                meta = None

            if meta is not None:
                if len(self.target_shapes) == len(mnet.param_shapes):
                    pass
                    # meta = mnet.param_shapes_meta
                elif len(self.target_shapes) == len(mnet.hyper_shapes_learned):
                    meta = []
                    for ii in mnet.hyper_shapes_learned_ref:
                        meta.append(mnet.param_shapes_meta[ii])
                else:
                    warn('Target shapes of this hypernetwork could not be ' +
                         'matched to the meta information provided to the ' +
                         'initialization.')
                    meta = None

        # TODO If the user doesn't (or can't) provide an `mnet` instance, we
        # should alternatively allow him to pass meta information directly.
        if meta is None:
            meta = []

            # Heuristical approach to derive meta information from given shapes.
            layer_ind = 0
            for i, s in enumerate(self.target_shapes):
                curr_meta = dict()

                if len(s) > 1:
                    curr_meta['name'] = 'weight'
                    curr_meta['layer'] = layer_ind
                    layer_ind += 1
                else: # just a heuristic, we can't know
                    curr_meta['name'] = 'bias'
                    if i > 0 and meta[-1]['name'] == 'weight':
                        curr_meta['layer'] = meta[-1]['layer']
                    else:
                        curr_meta['layer'] = -1

                meta.append(curr_meta)

        assert len(meta) == len(self.target_shapes)

        # Mapping from layer index to the corresponding shape.
        layer_shapes = dict()
        # Mapping from layer index to whether the layer has a bias vector.
        layer_has_bias = defaultdict(lambda: False)
        for i, m in enumerate(meta):
            if m['name'] == 'weight' and m['layer'] != -1:
                assert len(self.target_shapes[i]) > 1
                layer_shapes[m['layer']] = self.target_shapes[i]
            if m['name'] == 'bias' and m['layer'] != -1:
                layer_has_bias[m['layer']] = True

        ### Compute input variance ###
        cond_dim = self._cond_in_size
        uncond_dim = self._uncond_in_size
        inp_dim = cond_dim + uncond_dim

        input_variance = 0
        if cond_dim > 0:
            input_variance += (cond_dim / inp_dim) * cond_var
        if uncond_dim > 0:
            input_variance += (uncond_dim / inp_dim) * uncond_var

        ### Initialize hidden layers to preserve variance ###
        # Note, if batchnorm layers are used, they will simply be initialized to
        # have no effect after initialization. This does not effect the
        # performed whitening operation.
        if self.batchnorm_layers is not None:
            for bn_layer in self.batchnorm_layers:
                if hasattr(bn_layer, 'scale'):
                    nn.init.ones_(bn_layer.scale)
                if hasattr(bn_layer, 'bias'):
                    nn.init.zeros_(bn_layer.bias)

            # Since batchnorm layers whiten the statistics of hidden
            # acitivities, the variance of the input will not be preserved by
            # Xavier/Kaiming.
            if len(self.batchnorm_layers) > 0:
                input_variance = 1.

        # We initialize biases with 0 (see Xavier assumption 4 in the Hyperfan
        # paper). Otherwise, we couldn't ignore the biases when computing the
        # output variance of a layer.
        # Note, we have to use fan-in init for the hidden layer to ensure the
        # property, that we preserve the input variance.
        assert len(self._layers) + 1 == len(self.layer_weight_tensors)
        for i, w_tensor in enumerate(self.layer_weight_tensors[:-1]):
            if use_xavier:
                iutils.xavier_fan_in_(w_tensor)
            else:
                torch.nn.init.kaiming_uniform_(w_tensor, mode='fan_in',
                                               nonlinearity='relu')

            if self.has_bias:
                nn.init.zeros_(self.layer_bias_vectors[i])

        ### Define default parameters of weight init distributions ###
        w_val_list = []
        w_var_list = []
        b_val_list = []
        b_var_list = []

        for i, m in enumerate(meta):
            def extract_val(user_arg):
                curr = None
                if isinstance(user_arg, (list, tuple)) and \
                        user_arg[i] is not None:
                    curr = user_arg[i]
                elif isinstance(user_arg, (dict)) and \
                        m['name'] in user_arg.keys():
                    curr = user_arg[m['name']]
                return curr
            curr_w_val = extract_val(w_val)
            curr_w_var = extract_val(w_var)
            curr_b_val = extract_val(b_val)
            curr_b_var = extract_val(b_var)

            if m['name'] == 'weight' or m['name'] == 'bias':
                if None in [curr_w_val, curr_w_var, curr_b_val, curr_b_var]:
                    # If distribution not fully specified, then we just fall
                    # back to hyper-fan init.
                    curr_w_val = None
                    curr_w_var = None
                    curr_b_val = None
                    curr_b_var = None
            else:
                assert m['name'] in ['bn_scale', 'bn_shift', 'cm_scale',
                                     'cm_shift', 'embedding']
                if curr_w_val is None:
                    curr_w_val = 0
                if curr_w_var is None:
                    curr_w_var = 0
                if curr_b_val is None:
                    curr_b_val = 1 if m['name'] in ['bn_scale', 'cm_scale'] \
                                   else 0
                if curr_b_var is None:
                    curr_b_var = 1 if m['name'] in ['embedding'] else 0

            w_val_list.append(curr_w_val)
            w_var_list.append(curr_w_var)
            b_val_list.append(curr_b_val)
            b_var_list.append(curr_b_var)

        ### Initialize output heads ###
        # Note, that all output heads are realized internally via one large
        # fully-connected layer.
        # All output heads are linear layers. The biases of these linear
        # layers (called gamma and beta in the paper) are simply initialized
        # to zero. Note, that we allow deviations from this below.
        if self.has_bias:
            nn.init.zeros_(self.layer_bias_vectors[-1])

        c_relu = 1 if use_xavier else 2


        # We are not interested in the fan-out, since the fan-out is just
        # the number of elements in the main network.
        # `fan-in` is called `d_k` in the paper and is just the size of the
        # last hidden layer.
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(\
            self.layer_weight_tensors[-1])

        s_ind = 0
        for i, out_shape in enumerate(self.target_shapes):
            m = meta[i]
            e_ind = s_ind + int(np.prod(out_shape))

            curr_w_val = w_val_list[i]
            curr_w_var = w_var_list[i]
            curr_b_val = b_val_list[i]
            curr_b_var = b_var_list[i]

            if curr_w_val is None:
                c_bias = 2 if layer_has_bias[m['layer']] else 1

                if m['name'] == 'bias':
                    m_fan_out = out_shape[0]

                    # NOTE For the hyperfan-out init, we also need to know the
                    # fan-in of the layer.
                    if m['layer'] != -1:
                        m_fan_in, _ = iutils.calc_fan_in_and_out( \
                            layer_shapes[m['layer']])
                    else:
                        # FIXME Quick-fix.
                        m_fan_in = m_fan_out

                    var_in = c_relu / (2. * fan_in * input_variance)
                    num = c_relu * (1. - m_fan_in/m_fan_out)
                    denom = fan_in * input_variance
                    var_out = max(0, num / denom)

                else:
                    assert m['name'] == 'weight'
                    m_fan_in, m_fan_out = iutils.calc_fan_in_and_out(out_shape)

                    var_in = c_relu / (c_bias * m_fan_in * fan_in * \
                                       input_variance)
                    var_out = c_relu / (m_fan_out * fan_in * input_variance)

                if method == 'in':
                    var = var_in
                elif method == 'out':
                    var = var_out
                elif method == 'harmonic':
                    var = 2 * (1./var_in + 1./var_out)
                else:
                    raise ValueError('Method %s invalid.' % method)

                # Initialize output head weight tensor using `var`.
                std = math.sqrt(var)
                a = math.sqrt(3.0) * std
                torch.nn.init._no_grad_uniform_( \
                    self.layer_weight_tensors[-1][s_ind:e_ind, :], -a, a)
            else:
                if curr_w_var == 0:
                    nn.init.constant_(
                        self.layer_weight_tensors[-1][s_ind:e_ind, :],
                        curr_w_val)
                else:
                    std = math.sqrt(curr_w_var)
                    a = math.sqrt(3.0) * std
                    torch.nn.init._no_grad_uniform_( \
                        self.layer_weight_tensors[-1][s_ind:e_ind, :],
                        curr_w_val-a, curr_w_val+a)

                if curr_b_var == 0:
                    nn.init.constant_(
                        self.layer_bias_vectors[-1][s_ind:e_ind],
                        curr_b_val)
                else:
                    std = math.sqrt(curr_b_var)
                    a = math.sqrt(3.0) * std
                    torch.nn.init._no_grad_uniform_( \
                        self.layer_bias_vectors[-1][s_ind:e_ind],
                        curr_b_val-a, curr_b_val+a)

            s_ind = e_ind

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
