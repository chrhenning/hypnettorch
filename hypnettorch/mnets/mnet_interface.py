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
# @title          :mnets/mnet_interface.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :09/20/2019
# @version        :1.0
# @python_version :3.6.8
"""
Main-Network Interface
----------------------

The module :mod:`mnets.mnet_interface` contains an interface for main networks.
The interface ensures that we can consistently use these networks without
knowing their specific implementation.
"""
from abc import ABC, abstractmethod
import numpy as np
from warnings import warn
import torch

from hypnettorch.utils.batchnorm_layer import BatchNormLayer
from hypnettorch.utils.context_mod_layer import ContextModLayer
from hypnettorch.utils import misc
from hypnettorch.utils.torch_utils import init_params

class MainNetInterface(ABC):
    """A general interface for main networks, that can be used stand-alone
    (i.e., having their own weights) or with no (or only some) internal
    weights, such that the remaining weights have to be passed through the
    forward function (e.g., they may be generated through a hypernetwork).
    """
    def __init__(self):
        super(MainNetInterface, self).__init__()

        ### IMPORTANT NOTE FOR DEVELOPERS IMPLEMENTING THIS INTERFACE ###
        ### The following member variables have to be set by all classes that
        ### implement this interface.
        ### Please always verify your implementation using the method
        ### `_is_properly_setup` at the end the constructor of any class
        ### implementing this interface.
        self._internal_params = None
        self._param_shapes = None
        # You don't have to implement this following attribute, but it might
        # be helpful, for instance for hypernetwork initialization.
        self._param_shapes_meta = None
        self._hyper_shapes_learned = None
        # You don't have to implement this following attribute, but it might
        # be helpful, for instance for hypernetwork initialization.
        self._hyper_shapes_learned_ref = None
        self._hyper_shapes_distilled = None
        self._has_bias = None
        self._has_fc_out = None
        self._mask_fc_out = None
        self._has_linear_out = None
        self._layer_weight_tensors = None
        self._layer_bias_vectors = None
        self._batchnorm_layers = None
        self._context_mod_layers = None

        ### The rest will be taken care of automatically.
        # This will be set automatically based on attribute `_param_shapes`.
        self._num_params = None
        # This will be set automatically based on attribute `_weights`.
        self._num_internal_params = None

        # Deprecated, use `_hyper_shapes_learned` instead.
        self._hyper_shapes = None
        # Deprecated, use `_param_shapes` instead.
        self._all_shapes = None
        # Deprecated, use `_internal_params` instead.
        self._weights = None

    def _is_properly_setup(self, check_has_bias=True):
        """This method can be used by classes that implement this interface to
        check whether all required properties have been set."""
        assert(self._param_shapes is not None or self._all_shapes is not None)
        if self._param_shapes is None:
            warn('Private member "_param_shapes" should be specified in each ' +
                 'sublcass that implements this interface, since private ' +
                 'member "_all_shapes" is deprecated.', DeprecationWarning)
            self._param_shapes = self._all_shapes

        if self._hyper_shapes is not None or \
                self._hyper_shapes_learned is not None:
            if self._hyper_shapes_learned is None:
                warn('Private member "_hyper_shapes_learned" should be ' +
                     'specified in each sublcass that implements this ' +
                     'interface, since private member "_hyper_shapes" is ' +
                     'deprecated.', DeprecationWarning)
                self._hyper_shapes_learned = self._hyper_shapes
            # FIXME we should actually assert equality if
            # `_hyper_shapes_learned` was not None.
            self._hyper_shapes = self._hyper_shapes_learned

        assert self._weights is None or self._internal_params is None
        if self._weights is not None and self._internal_params is None:
            # Note, in the future we might throw a deprecation warning here,
            # once "weights" becomes deprecated.
            self._internal_params = self._weights

        assert self._internal_params is not None or \
               self._hyper_shapes_learned is not None

        if self._hyper_shapes_learned is None and \
                self.hyper_shapes_distilled is None:
            # Note, `internal_params` should only contain trainable weights and
            # not other things like running statistics. Thus, things that are
            # passed to an optimizer.
            assert len(self._internal_params) == len(self._param_shapes)

        if self._param_shapes_meta is None:
            # Note, this attribute was inserted post-hoc.
            # FIXME Warning is annoying, programmers will notice when they use
            # this functionality.
            #warn('Attribute "param_shapes_meta" has not been implemented!')
            pass
        else:
            assert(len(self._param_shapes_meta) == len(self._param_shapes))
            for dd in self._param_shapes_meta:
                assert isinstance(dd, dict)
                assert 'name' in dd.keys() and 'index' in dd.keys() and \
                    'layer' in dd.keys()
                assert dd['name'] is None or \
                       dd['name'] in ['weight', 'bias', 'bn_scale', 'bn_shift',
                                      'cm_scale', 'cm_shift', 'embedding']

                assert isinstance(dd['index'], int)
                if self._internal_params is None:
                    assert dd['index'] == -1
                else:
                    assert dd['index'] == -1 or \
                        0 <= dd['index'] < len(self._internal_params)

                assert isinstance(dd['layer'], int)
                assert dd['layer'] == -1 or dd['layer'] >= 0

        if self._hyper_shapes_learned is not None:
            if self._hyper_shapes_learned_ref is None:
                # Note, this attribute was inserted post-hoc.
                # FIXME Warning is annoying, programmers will notice when they
                # use this functionality.
                #warn('Attribute "hyper_shapes_learned_ref" has not been ' +
                #     'implemented!')
                pass
            else:
                assert isinstance(self._hyper_shapes_learned_ref, list)
                for ii in self._hyper_shapes_learned_ref:
                    assert isinstance(ii, int)
                    assert ii == -1 or 0 <= ii < len(self._param_shapes)

        assert(isinstance(self._has_fc_out, bool))
        assert(isinstance(self._mask_fc_out, bool))
        assert(isinstance(self._has_linear_out, bool))

        assert(self._layer_weight_tensors is not None)
        assert(self._layer_bias_vectors is not None)

        # Note, you should overwrite the `has_bias` attribute if you do not
        # follow this requirement.
        if check_has_bias:
            assert isinstance(self._has_bias, bool)
            if self._has_bias:
                assert len(self._layer_weight_tensors) == \
                       len(self._layer_bias_vectors)

    @property
    def internal_params(self):
        """A list of all internally maintained parameters of the main network
        currently in use. If all parameters are assumed to be generated
        externally, then this attribute will be ``None``.

        Simply speaking, the parameters listed here should be passed to
        the optimizer.

        Note:
            In most cases, the attribute will contain the same set of parameter
            objects as the method :meth:`torch.nn.Module.parameters` would
            return. Though, there  might be future use-cases where the
            programmer wants to hide parameters from the optimizer in a task-
            or time-dependent manner.

        :type: torch.nn.ParameterList or None
        """
        return self._internal_params

    @property
    def weights(self):
        """Same as :attr:`internal_params`.

        .. deprecated:: 1.0
            Please use attribute :attr:`internal_params` instead.

        :type: torch.nn.ParameterList or None
        """
        warn('Use attribute "internal_params" rather than "weigths", as ' +
             '"weights" might be removed in the future.', DeprecationWarning)
        return self.internal_params

    @property
    def internal_params_ref(self):
        """A list of integers. Each entry either represents an index within
        attribute :attr:`param_shapes` or is set to ``-1``.

        Can only be spacified if :attr:`internal_params` is not ``None``.

        .. note::

            The possibility that entries may be ``-1`` should account for
            unforeseeable flexibility that programmers may need.

        :type: list or None
        """
        if self.internal_params is None:
            return None

        if len(self.internal_params) == 0:
            return []

        # Note, programmers are not forced (just encouraged) to implement
        # `param_shapes_meta`.
        try:
            psm = self.param_shapes_meta
        except:
            raise NotImplementedError('Attribute "internal_params_ref" ' +
                'requires that attribute "param_shapes_meta" is implemented ' +
                'for this network.')

        ret_dict = {}

        for i, m in enumerate(psm):
            if m['index'] != -1:
                assert m['index'] not in ret_dict.keys()
                ret_dict[m['index']] = i

        assert np.all(np.isin(np.arange(len(self.internal_params)),
                              list(ret_dict.keys())))
        return np.sort(list(ret_dict.keys())).tolist()

    @property
    def param_shapes(self):
        """A list of lists of integers. Each list represents the shape of a
        parameter tensor. Note, this attribute is independent of the attribute
        :attr:`internal_params`, it always comprises the shapes of all parameter
        tensors as if the network would be stand-alone (i.e., no weights being
        passed to the :meth:`forward` method).

        :type: list
        """
        return self._param_shapes

    @property
    def param_shapes_meta(self):
        """A list of dictionaries. The length of the list is equal to the length
        of the list :attr:`param_shapes` and each entry of this list provides
        meta information to the corresponding entry in :attr:`param_shapes`.
        Each dictionary contains the keys ``name``, ``index`` and ``layer``.
        The key ``name`` is a string and refers to the type of weight tensor
        that the shape corresponds to.

        Possible values are:

        - ``'weight'``: A weight tensor of a standard layer as those
          stored in attribute :attr:`layer_weight_tensors`.
        - ``'bias'``: A bias vector of a standard layer as those
          stored in attribute :attr:`layer_bias_vectors`.
        - ``'bn_scale'``: The weights for scaling activations in a
          batchnorm layer :class:`utils.batchnorm_layer.BatchNormLayer`.
        - ``'bn_shift'``: The weights for shifting activations in a
          batchnorm layer :class:`utils.batchnorm_layer.BatchNormLayer`.
        - ``'cm_scale'``: The weights for scaling activations in a
          context-mod layer
          :class:`utils.context_mod_layer.ContextModLayer`.
        - ``'cm_shift'``: The weights for shifting activations in a
          context-mod layer
          :class:`utils.context_mod_layer.ContextModLayer`.
        - ``'embedding'``: The parameters represent embeddings.
        - ``None``: Not specified!

        The key ``index`` might refer to the index of the corresponding
        parameter tensor (if existing) inside the :attr:`internal_params`
        list. It is ``-1`` if the parameter tensor is not internally
        maintained.

        The key ``layer`` is an integer. Shapes with the same ``layer``
        entry are supposed to reside in the same layer. For instance, a
        ``'weight'`` and a ``'bias'`` with the same entry for key ``layer``
        are supposed to be the weight tensor and bias vector in the same
        layer. The value ``-1`` refers to `not specified`!

         :type: list
        """
        if self._param_shapes_meta is None:
            raise NotImplementedError('Attribute not implemented for this ' +
                                      'network.')

        return self._param_shapes_meta

    @property
    def hyper_shapes_learned(self):
        """A list of lists of integers. Each list represents the shape of a
        weight tensor that has to be passed to the :meth:`forward` method during
        training. If all weights are maintained internally, then this attribute
        will be ``None``.

        :type: list
        """
        return self._hyper_shapes_learned

    @property
    def hyper_shapes_learned_ref(self):
        """A list of integers. Each entry either represents an index within
        attribute :attr:`param_shapes` or is set to ``-1``.

        .. note::

            The possibility that entries may be ``-1`` should account for
            unforeseeable flexibility that programmers may need.

        :type: list
        """
        if self._hyper_shapes_learned is not None and \
                self._hyper_shapes_learned_ref is None:
            raise NotImplementedError('Attribute not implemented for this ' +
                                      'network')

        return self._hyper_shapes_learned_ref

    @property
    def hyper_shapes_distilled(self):
        """A list of lists of integers. This attribute is complementary to
        attribute :attr:`hyper_shapes_learned`, which contains shapes of tensors
        that are learned through the hypernetwork. In contrast, this attribute
        should contain the shapes of tensors that are not needed by the main
        network during training (as it learns or calculates the tensors
        itself), but should be distilled into a hypernetwork after training
        in order to avoid increasing memory consumption.

        The attribute is ``None`` if no tensors have to be distilled into
        a hypernetwork.

        For instance, if batch normalization is used, then the attribute
        :attr:`hyper_shapes_learned` might contain the batch norm weights
        whereas the attribute :attr:`hyper_shapes_distilled` contains the
        running statistics, which are first estimated by the main network
        during training and later distilled into the hypernetwork.

        :type: list or None
        """
        return self._hyper_shapes_distilled

    @property
    def has_bias(self):
        """Whether layers in this network have bias terms.

        :type: bool
        """
        return self._has_bias

    @property
    def has_fc_out(self):
        """Whether the output layer of the network is a fully-connected layer.

        :type: bool
        """
        return self._has_fc_out

    @property
    def mask_fc_out(self):
        """If this attribute is set to ``True``, it is implicitly assumed that
        if :attr:`hyper_shapes_learned` is not ``None``, the last two entries of
        :attr:`hyper_shapes_learned` are the weights and biases of the final
        fully-connected layer.

        This attribute is helpful, for instance, in multi-head continual
        learning settings. In case we regularize task-specific main network
        weights, it is important to know which weights are specific for an
        output head (as determined by the weights of the final layer).

        .. note::
            Only applies if attribute :attr:`has_fc_out` is ``True``.

        :type: bool
        """
        return self._mask_fc_out

    @property
    def has_linear_out(self):
        """Is ``True`` if no nonlinearity is applied in the output layer.

        :type: bool
        """
        return self._has_linear_out

    @property
    def num_params(self):
        """The total number of weights in the parameter tensors described by the
        attribute :attr:`param_shapes`.

        :type: int
        """
        if self._num_params is None:
            self._num_params = MainNetInterface.shapes_to_num_weights( \
                self.param_shapes)
        return self._num_params

    @property
    def num_internal_params(self):
        """The number of internally maintained parameters as prescribed by
        attribute :attr:`internal_params`.

        :type: int
        """
        if self._num_internal_params is None:
            if self.internal_params is None:
                self._num_internal_params = 0
            else:
                # FIXME should we distinguish between trainable and
                # non-trainable parameters (`p.requires_grad`)?
                self._num_internal_params = int(sum(p.numel() for p in \
                                                    self.internal_params))
        return self._num_internal_params

    @property
    def layer_weight_tensors(self):
        """These are the actual weight tensors used in layers (e.g., weight
        matrix in fully-connected layer, kernels in convolutional layer, ...).

        This attribute is useful when applying a custom initialization to
        these layers.

        :type: torch.nn.ParameterList
        """
        return self._layer_weight_tensors

    @property
    def layer_bias_vectors(self):
        """Similar to attribute :attr:`layer_weight_tensors` but for the bias
        vectors in each layer. List should be empty in case :attr:`has_bias` is
        ``False``.

        Note:
            There might be cases where some weight matrices in attribute
            :attr:`layer_weight_tensors` have no bias vectors, in which
            case elements of this list might be ``None``.

        :type: torch.nn.ParameterList
        """
        return self._layer_bias_vectors

    @property
    def batchnorm_layers(self):
        """A list of instances of class
        :class:`utils.batchnorm_layer.BatchNormLayer` in case batch
        normalization is used in this network.

        .. note::
            We explicitly do not support the usage of PyTorch its batchnorm
            layers as class :class:`utils.batchnorm_layer.BatchNormLayer`
            represents a hypernet compatible wrapper for them.

        :type: torch.nn.ModuleList
        """
        return self._batchnorm_layers

    @property
    def context_mod_layers(self):
        """A list of instances of class
        :class:`utils.context_mod_layer.ContextModLayer` in case these are
        used in this network.

        :type: torch.nn.ModuleList
        """
        return self._context_mod_layers

    @abstractmethod
    def distillation_targets(self):
        """Targets to be distilled after training.

        If :attr:`hyper_shapes_distilled` is not ``None``, then this method
        can be used to retrieve the targets that should be distilled into an
        external hypernetwork after training.

        The shapes of the returned tensors have to match the shapes specified in
        :attr:`hyper_shapes_distilled`.

        Example:

            Assume a continual learning scenario with a main network that uses
            batch normalization (and tracks running statistics). Then this
            method should be called right after training on a task in order to
            retrieve the running statistics, such that they can be distilled
            into a hypernetwork.

        Returns:
            The target tensors corresponding to the shapes specified in
            attribute :attr:`hyper_shapes_distilled`.
        """
        raise NotImplementedError('TODO implement function')

    @abstractmethod
    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            x: The inputs :math:`x` to the network.
            weights (optional): List of weight tensors, that are used as network
                parameters. If attribute :attr:`hyper_shapes_learned` is not
                ``None``, then this argument is non-optional and the shapes
                of the weight tensors have to be as specified by
                :attr:`hyper_shapes_learned`.

                Otherwise, this option might still be set but the weight tensors
                must follow the shapes specified by attribute
                :attr:`param_shapes`.
            distilled_params (optional): May only be passed if attribute
                :attr:`hyper_shapes_distilled` is not ``None``.

                If not passed but the network relies on those parameters
                (e.g., batchnorm running statistics), then this method simply
                chooses the current internal representation of these parameters
                as returned by :meth:`distillation_targets`.
            condition (optional): Sometimes, the network will have to be
                conditioned on contextual information, which can be passed via
                this argument and depends on the actual implementation of this
                interface.

                For instance, when using batch normalization in a continual
                learning scenario, where running statistics have been
                checkpointed for every task, then this ``condition`` might be
                the actual task ID, that is passed as the argument ``stats_id``
                of the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward`.

        Returns:
            The output :math:`y` of the network.
        """
        raise NotImplementedError('TODO implement function')

    @staticmethod
    def shapes_to_num_weights(dims):
        """The number of parameters contained in a list of tensors with the
        given shapes.

        Args:
            dims: List of tensor shapes. For instance, the attribute
                :attr:`hyper_shapes_learned`.

        Returns:
            (int)
        """
        return int(np.sum([np.prod(l) for l in dims]))

    def custom_init(self, normal_init=False, normal_std=0.02, zero_bias=True):
        """Initialize weight tensors in attribute :attr:`layer_weight_tensors`
        using Xavier initialization and set bias vectors to 0.

        Note:
            This method will override the default initialization of the network,
            which is often based on :func:`torch.nn.init.kaiming_uniform_`
            for weight tensors (i.e., attribute :attr:`layer_weight_tensors`)
            and a uniform init based on fan-in/fan-out for bias vectors
            (i.e., attribute :attr:`layer_bias_vectors`).

        Args:
            normal_init (bool): Use normal initialization rather than Xavier.
            normal_std (float): The standard deviation when choosing
                ``normal_init``.
            zero_bias (bool): Whether bias vectors should be initialized to
                zero. If ``False``, then bias vectors are left untouched.
        """
        for w in self.layer_weight_tensors:
            if normal_init:
                torch.nn.init.normal_(w, mean=0, std=normal_std)
            else:
                torch.nn.init.xavier_uniform_(w)

        if zero_bias:
            for b in self.layer_bias_vectors:
                if b is not None:
                    torch.nn.init.constant_(b, 0)

    def get_output_weight_mask(self, out_inds=None, device=None):
        """Create a mask for selecting weights connected solely to certain
        output units.

        This method will return a list of the same length as
        :attr:`param_shapes`. Entries in this list are either ``None`` or
        masks for the corresponding parameter tensors. For all parameter
        tensors that are not directly connected to output units, the
        corresponding entry will be ``None``. If ``out_inds is None``, then all
        output weights are selected by a masking value ``1``. Otherwise, only
        the weights connected to the output units in ``out_inds`` are selected,
        the rest is masked out.

        Note:
            This method only works for networks with a fully-connected output
            layer (see :attr:`has_fc_out`), that have the attribute
            :attr:`mask_fc_out` set. Otherwise, the method has to be overwritten
            by an implementing class.

        Args:
            out_inds (list, optional): List of integers. Each entry denotes an
                output unit.
            device: Pytorch device. If given, the created masks will be moved
                onto this device.

        Returns:
            (list): List of masks with the same length as :attr:`param_shapes`.
            Entries whose corresponding parameter tensors are not connected to
            the network outputs are ``None``.
        """
        if not (self.has_fc_out and self.mask_fc_out):
            raise NotImplementedError('Method not applicable for this ' +
                                      'network type.')

        ret = [None] * len(self.param_shapes)

        obias_ind = len(self.param_shapes)-1 if self.has_bias else None
        oweights_ind = len(self.param_shapes)-2 if self.has_bias \
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

    @staticmethod
    def _parse_context_mod_args(cm_kwargs):
        """Parse context-modulation arguments for a class.

        This function first loads the default values of all context-mod
        arguments passed to class :class:`mnets.mlp.MLP`. If any of these
        arguments is not occurring in the dictionary ``cm_kwargs``, then they
        will be added using the default value from class :class:`mnets.mlp.MLP`.

        Args:
            cm_kwargs (dict): A dictionary, that is modified in place (i.e.,
                missing keys are added).

        Returns:
            (list): A list of key names from ``cm_kwargs`` that are not related
            to context-modulation, i.e., unknown to this function.
        """
        from hypnettorch.mnets.mlp import MLP

        # All context-mod related arguments in `mnets.mlp.MLP.__init__`.
        cm_keys = ['use_context_mod',
                   'context_mod_inputs',
                   'no_last_layer_context_mod',
                   'context_mod_no_weights',
                   'context_mod_post_activation',
                   'context_mod_gain_offset',
                   'context_mod_gain_softplus']

        default_cm_kwargs = misc.get_default_args(MLP.__init__)

        for k in cm_keys:
            assert k in default_cm_kwargs.keys()
            if k not in cm_kwargs.keys():
                cm_kwargs[k] = default_cm_kwargs[k]

        # Extract keyword arguments that do not belong to context-mod.
        unknown_kwargs = []
        for k in cm_kwargs.keys():
            if k not in default_cm_kwargs.keys():
                unknown_kwargs.append(k)

        return unknown_kwargs

    def _add_context_mod_layers(self, cm_shapes, cm_layers=None):
        """Add context mod layers to the network.

        Note:
            This method should only be called inside the constructor of any
            class that implements this interface.

        Note:
            This method assumes that the context-mod related arguments of
            class :class:`mnets.mlp.MLP` are properly set as private attributes.

        Note:
            This method will set attributes :attr:`param_shapes_meta` and
            :attr:`hyper_shapes_learned_ref` correctly only if they are not
            ``None``.

        Args:
            cm_shapes (list): List of list of intergers. The shapes of each
                context-mod layer that should be added.
            cm_layers (list, optional): List of integers. Can be provided to
                specify the key ``layer`` for the attribute
                :attr:`param_shapes_meta`. Otherwise, the values of key
                ``layer`` are simply ``-1``.
        """
        assert cm_layers is None or len(cm_layers) == len(cm_shapes)

        for i, s in enumerate(cm_shapes):
            cmod_layer = ContextModLayer(s,
                no_weights=self._context_mod_no_weights,
                apply_gain_offset=self._context_mod_gain_offset,
                apply_gain_softplus=self._context_mod_gain_softplus)
            assert len(cmod_layer.param_shapes) in [1, 2]
            self._context_mod_layers.append(cmod_layer)

            self.param_shapes.extend(cmod_layer.param_shapes)
            if self._param_shapes_meta is not None:
                for ii, ps_name in enumerate(cmod_layer.param_shapes_meta):
                    assert ps_name in ['gain', 'shift']
                    self._param_shapes_meta.append(
                        {'name': 'cm_scale' if ps_name == 'gain' \
                             else 'cm_shift',
                         'index': -1 if self._context_mod_no_weights else \
                             len(self._internal_params) + ii,
                         'layer': -1 if cm_layers is None else cm_layers[i]
                        })

            if self._context_mod_no_weights:
                self._hyper_shapes_learned.extend(cmod_layer.param_shapes)
                if self._hyper_shapes_learned_ref is not None:
                    self._hyper_shapes_learned_ref.extend(range( \
                        len(self.param_shapes)-len(cmod_layer.param_shapes),
                        len(self.param_shapes)))
            else:
                self._internal_params.extend(cmod_layer.weights)

    def _num_context_mod_shapes(self):
        """The number of entries in :attr:`param_shapes` associated with
        context-modulation.

        Returns:
            (int): Returns ``0`` if :attr:`context_mod_layers` is ``None``.
        """
        if self.context_mod_layers is None:
            return 0

        ret = 0
        for cm_layer in self.context_mod_layers:
            ret += len(cm_layer.param_shapes)

        return ret

    def _add_batchnorm_layers(self, bn_sizes, bn_no_weights, bn_layers=None,
                              distill_bn_stats=False, bn_track_stats=True):
        """Add batchnorm layers to the network.

        Note:
            This method should only be called inside the constructor of any
            class that implements this interface.

        Note:
            This method will set attributes :attr:`param_shapes_meta` and
            :attr:`hyper_shapes_learned_ref` correctly only if they are not
            ``None``.

        Args:
            bn_sizes (list): List of intergers denoting the feature size of
                each batchnorm layer.
            bn_no_weights (bool): If ``True``, batchnorm layers will be
                generated without internal parameters :attr:`internal_params`.
            bn_layers (list, optional): See attribute ``cm_layers`` of method
                :meth:`_add_context_mod_layers`.
            distill_bn_stats (bool): If ``True``, the stats shapes will be
                appended to :attr:`hyper_shapes_distilled`.
            bn_track_stats (bool): Will be passed as argument
                ``track_running_stats`` to class
                :class:`utils.batchnorm_layer.BatchNormLayer`.
        """
        assert bn_layers is None or len(bn_layers) == len(bn_sizes)

        if self._batchnorm_layers is None and len(bn_sizes) > 0:
            self._batchnorm_layers = torch.nn.ModuleList()

        if distill_bn_stats and self._hyper_shapes_distilled is None:
            self._hyper_shapes_distilled = []

        for i, n in enumerate(bn_sizes):
            bn_layer = BatchNormLayer(n, affine=not bn_no_weights,
                    track_running_stats=bn_track_stats)
            self._batchnorm_layers.append(bn_layer)

            assert len(bn_layer.param_shapes) == 2
            self.param_shapes.extend(bn_layer.param_shapes)
            if self._param_shapes_meta is not None:
                self._param_shapes_meta.extend([
                    {'name': 'bn_scale',
                     'index': -1 if bn_no_weights else \
                         len(self._internal_params),
                     'layer': -1 if bn_layers is None else bn_layers[i]},
                    {'name': 'bn_shift',
                     'index': -1 if bn_no_weights else \
                         len(self._internal_params)+1,
                     'layer': -1 if bn_layers is None else bn_layers[i]},
                ])

            if bn_no_weights:
                self._hyper_shapes_learned.extend(bn_layer.param_shapes)
                if self._hyper_shapes_learned_ref is not None:
                    self._hyper_shapes_learned_ref.extend(range( \
                        len(self.param_shapes)-len(bn_layer.param_shapes),
                        len(self.param_shapes)))
            else:
                self._internal_params.extend(bn_layer.weights)

            if distill_bn_stats:
                self._hyper_shapes_distilled.extend( \
                    [list(p.shape) for p in bn_layer.get_stats(0)])

    def _add_fc_layers(self, in_sizes, out_sizes, no_weights, fc_layers=None):
        """Add fully-connected layers to the network.

        This method will set the weight requirements for fully-connected layers
        correctly. During the :meth:`forward` computation, those weights can be
        used in combination with :func:`torch.nn.functional.linear`.

        Note:
            This method should only be called inside the constructor of any
            class that implements this interface.

        Note:
            Bias weights are handled based on attribute :attr:`has_bias`.

        Note:
            This method will assumes attributes :attr:`param_shapes_meta` and
            :attr:`hyper_shapes_learned_ref` exist already.

        Note:
            Generated weights will be automatically added to attributes
            :attr:`layer_bias_vectors` and :attr:`layer_weight_tensors`.

        Note:
            Standard initialization will be applied to created weights.

        Args:
            in_sizes (list): List of intergers denoting the input size of each
                added fc-layer.
            out_sizes (list): List of intergers denoting the output size of each
                added fc-layer.
            no_weights (bool): If ``True``, fc-layers will be generated without
                internal parameters :attr:`internal_params`.
            fc_layers (list, optional): See attribute ``cm_layers`` of method
                :meth:`_add_context_mod_layers`.
        """
        assert len(in_sizes) == len(out_sizes)
        assert fc_layers is None or len(fc_layers) == len(in_sizes)
        assert self._param_shapes_meta is not None
        assert not no_weights or self._hyper_shapes_learned_ref is not None

        if self._layer_weight_tensors is None:
            self._layer_weight_tensors = torch.nn.ParameterList()
        if self._layer_bias_vectors is None:
            self._layer_bias_vectors = torch.nn.ParameterList()

        for i, n_in in enumerate(in_sizes):
            n_out = out_sizes[i]

            s_w = [n_out, n_in]
            s_b = [n_out] if self.has_bias else None

            for j, s in enumerate([s_w, s_b]):
                if s is None:
                    continue

                is_bias = True
                if j % 2 == 0:
                    is_bias = False

                if not no_weights:
                    self._internal_params.append(torch.nn.Parameter( \
                        torch.Tensor(*s), requires_grad=True))
                    if is_bias:
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
                    'name': 'bias' if is_bias else 'weight',
                    'index': -1 if no_weights else len(self._internal_params)-1,
                    'layer': -1 if fc_layers is None else fc_layers[i]
                })

            if not no_weights:
                init_params(self._layer_weight_tensors[-1],
                    self._layer_bias_vectors[-1] if self.has_bias else None)

    def overwrite_internal_params(self, new_params):
        """Overwrite the values of all internal parameters.

        This will affect all parameters maintained in attribute
        :attr:`internal_params`.

        An example usage of this method is the initialization of a standalone
        main network with weights that have been previously produced by a
        hypernetwork.

        Args:
            new_params: A list of parameter values that are used to initialize
                 the network internal parameters is expected.
        """
        assert len(new_params) == len(self.internal_params)
        for i in range(len(new_params)):
            assert np.all(np.equal(list(new_params[i].shape),
                                   self.internal_params[i].shape))
            self.internal_params[i].data = new_params[i]

    @staticmethod
    def flatten_params(params, param_shapes=None, unflatten=False):
        """Flatten a list of parameter tensors.

        This function will take a list of parameter tensors and flatten them
        into a single vector. This flattening operation can also be undone using
        the argument ``unflatten``.

        Args:
            params (list): A list of tensors. Those tensors will be flattened
                and concatenated into a tensor. If ``unflatten=True``, then
                ``params`` is expected to be a flattened tensor, which will be
                split into a list of tensors according to ``param_shapes``.
            param_shapes (list): List of parameter tensor shapes. Required when
                unflattening a flattened parameter tensor.
            unflatten (bool): If ``True``. the flattening operation will be
                reversed.

        Returns:
            (torch.Tensor): The flattened tensor. If ``unflatten=True``, a list
            of tensors will be returned.
        """
        if unflatten:
            assert param_shapes is not None

            ret = []

            ind = 0
            for s in param_shapes:
                num = int(np.prod(s))

                p = params[ind:ind+num]
                p = p.view(*s)
                ret.append(p)

                ind += num

            return ret

        else:
            return torch.cat([p.flatten() for p in params])

if __name__ == '__main__':
    pass


