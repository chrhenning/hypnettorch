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
# @title          :hnets/hnet_container.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :07/30/2020
# @version        :1.0
# @python_version :3.6.9
"""
Hypernetwork-container that wraps a mixture of hypernets
--------------------------------------------------------

The module :mod:`hnets.hnet_container` contains a hypernetwork container,
i.e., a hypernetwork that produces weights by internally using a mixture of
hypernetworks that implement the interface
:class:`hnets.hnet_interface.HyperNetInterface`. The container also allows the
specification of shared or condition-specific weights.

Example:
    Assume a target network with shapes
    ``target_shapes=[[10, 5], [5], [5], [5], [5, 5]]``, where the first 4
    tensors represent the weight matrix, bias vector and batch norm scale and
    shift, while the last tensor is the linear output layer's weight matrix.

    We consider two usecase scenarios. In the first one, the first layer weights
    (matrix and bias vector) are generated by a hypernetwork, while the batch
    norm weights should be realized via a fixed set of shared weights. The
    output weights shall be condition-specific:

    .. code-block:: python

        from hnets import HMLP

        # First-layer weights.
        fl_hnet = HMLP([[10, 5], [5]], num_cond_embs=5)

        def assembly_fct(list_of_hnet_tensors, uncond_tensors, cond_tensors):
            assert len(list_of_hnet_tensors) == 1
            return list_of_hnet_tensors[0] + uncond_tensors + cond_tensors

        hnet = HContainer([[10, 5], [5], [5], [5], [5, 5]], assembly_fct,
                          hnets=[fl_hnet], uncond_param_shapes=[[5], [5]],
                          cond_param_shapes=[[5, 5]],
                          uncond_param_names=['bn_scale', 'bn_shift'],
                          cond_param_names=['weight'], num_cond_embs=5)

    In the second usecase scenario, we utilize two separate hypernetworks, one
    as above and a second one for the condition-specific output weights.
    Batchnorm weights remain to be realized via a single set of shared weights.

    .. code-block:: python

        from hnets import HMLP

        # First-layer weights.
        fl_hnet = HMLP([[10, 5], [5]], num_cond_embs=5)
        # Last-layer weights.
        ll_hnet = HMLP([[5, 5]], num_cond_embs=5)

        def assembly_fct(list_of_hnet_tensors, uncond_tensors, cond_tensors):
            assert len(list_of_hnet_tensors) == 2
            return list_of_hnet_tensors[0] + uncond_tensors + \\
                list_of_hnet_tensors[1]

        hnet = HContainer([[10, 5], [5], [5], [5], [5, 5]], assembly_fct,
                          hnets=[fl_hnet, ll_hnet],
                          uncond_param_shapes=[[5], [5]],
                          uncond_param_names=['bn_scale', 'bn_shift'],
                          num_cond_embs=5)
"""
import numpy as np
import torch
import torch.nn as nn
from warnings import warn

from hypnettorch.hnets.hnet_interface import HyperNetInterface

class HContainer(nn.Module, HyperNetInterface):
    """Implementation of a wrapper that abstracts the use of a set of
    hypernetworks.

    Note:
        Parameter tensors instantiated by this constructor are initialized via
        a normal distribution :math:`\\mathcal{N}(0, 0.02)`.

    Args:
        (....): See constructor arguments of class
            :class:`hnets.mlp_hnet.HMLP`.
        assembly_fct (func): A function handle that takes the produced tensors
            of each internal `hypernet` (see arguments ``hnets``,
            ``uncond_param_shapes`` and ``cond_param_shapes``) and converts them
            into tensors with shapes ``target_shapes``.

            The function handle must have the signature:
            ``assembly_fct(list_of_hnet_tensors, uncond_tensors, cond_tensors)``
            . The first argument is a list of lists of tensors, the reamining
            two are lists of tensors. ``hnet_tensors`` contains the output of
            each hypernetwork in ``hnets``. ``uncond_tensors`` contains all
            internally maintained unconditional weights as specified by
            ``uncond_param_shapes``. ``cond_tensors`` contains the internally
            maintained weights corresponding to the selected condition and as
            specified by argument ``cond_param_shapes``. The function is
            expected to return a list of tensors, each of them having a shape as
            specified by ``target_shapes``.

            Example:
                Assume ``target_shapes=[[3], [3], [10, 5], [5]]`` and that
                ``hnets`` is made up of two hypernetworks with output shapes
                ``[[3]]`` and ``[[3], [10, 5]]``. In addition
                ``cond_param_shapes=[[5]]``.
                Then the argument ``hnet_tensors`` will be a list of lists of
                tensors as follows:
                ``[[tensor(3)], [tensor(3), tensor(10, 5)]``, ``uncond_tensors``
                will be an empty list and ``cond_tensors`` will be list of
                tensors: ``[[tensor(5)]]``.

                The output of ``assembly_fct`` is expected to be a list of
                tensors as follows:
                ``[tensor(3), tensor(3), tensor(10, 5), tensor(5)]``.

            Note:
                This function considers one sample at a time, even if a batch
                of inputs is processed.

            Note:
                It is assumed that ``assembly_fct`` does not further process the
                incoming weights. Otherwise, the attributes
                :attr:`mnets.mnet_interface.MainNetInterface.has_fc_out` and
                :attr:`mnets.mnet_interface.MainNetInterface.has_linear_out`
                might be invalid.
        hnets (list, optional): List of instances of class
            :class:`hnets.hnet_interface.HyperNetInterface`. All these
            hypernetworks are assumed to produce a part of the weights that are
            then assembled to a common hypernetwork output via the
            ``assembly_fct``.
        uncond_param_shapes (list, optional): List of lists of integers. Each
            entry in the list encodes the shape of an (unconditional) parameter
            tensor that will be added to attribute
            :attr:`hnets.hnet_interface.HyperNetInterface.unconditional_params`
            and additionally will also become an output of this hypernetwork
            that is passed to the ``assembly_fct``.

            Hence, these parameters are independent of the hypernetwork input.
            Thus, they are just treated as normal weights as if they were part
            of the main network. This option therefore only provides the
            convinience of mimicking the behavior weights would elicit if they
            were part of the main network without needing to change the main
            network its implementation.
        cond_param_shapes (list, optional): List of lists of integers. Each
            entry in the list encodes the shape of a (conditional) parameter
            tensor that will be added to attribute
            :attr:`hnets.hnet_interface.HyperNetInterface.conditional_params`
            (how often it will be added is determined by argument
            ``num_cond_embs``). It is otherwise similar to option
            ``uncond_param_shapes``.

            Note:
                If this option is specified, then argument ``cond_id`` of
                :meth:`forward` has to be specified.
        uncond_param_names (list, optional): If provided, it must have the same
            length as ``uncond_param_shapes``. It will contain a list of strings
            that are used as values for key ``name`` in attribute
            :attr:`hnets.hnet_interface.HyperNetInterface.param_shapes_meta`.

            If not provided, shapes with more than 1 element are assigned value
            ``weights`` and all others are assigned value ``bias``.
        cond_param_names (list, optional): Same as argument
            ``uncond_param_names`` for argument ``cond_param_shapes``.
    """
    def __init__(self, target_shapes, assembly_fct, hnets=None,
                 uncond_param_shapes=None, cond_param_shapes=None,
                 uncond_param_names=None, cond_param_names=None,
                 verbose=True, no_uncond_weights=False, no_cond_weights=False,
                 num_cond_embs=1):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        HyperNetInterface.__init__(self)

        if hnets is None and uncond_param_shapes is None and \
                cond_param_shapes is None:
            raise ValueError('Not specified how to produce hypernet output.')

        assert hnets is None or (isinstance(hnets, (list, tuple)) and \
                                 len(hnets) > 0)
        self._hnets = [] if hnets is None else hnets
        assert uncond_param_shapes is None or \
            (isinstance(uncond_param_shapes, (list, tuple)) and \
             len(uncond_param_shapes) > 0)
        self._uncond_ps = [] if uncond_param_shapes is None else \
            uncond_param_shapes
        assert cond_param_shapes is None or \
            (isinstance(cond_param_shapes, (list, tuple)) and \
             len(cond_param_shapes) > 0)
        self._cond_ps = [] if cond_param_shapes is None else \
            cond_param_shapes

        self._assembly_fct = assembly_fct

        assert uncond_param_names is None or (uncond_param_shapes is not None \
            and len(uncond_param_names) == len(uncond_param_shapes))
        assert cond_param_names is None or (cond_param_shapes is not None \
            and len(cond_param_names) == len(cond_param_shapes))

        ##############################
        ### Setup class attributes ###
        ##############################
        self._target_shapes = target_shapes
        self._num_known_conds = num_cond_embs

        # As we just append the weights of the internal hypernets we will have
        # output weights all over the place.
        # Additionally, it would be complicated to assign outputs to target
        # outputs, as we do not know, what is happening in the `assembly_fct`.
        self._mask_fc_out = False

        self._unconditional_param_shapes_ref = []
        self._param_shapes = []
        self._param_shapes_meta = []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        hnets_tmp = [] if hnets is None else hnets
        for i, hnet in enumerate(hnets_tmp):
            # Note, it is important to convert lists into new object and not
            # just copy references!
            # Note, we have to adapt all references if `i > 0`.

            ps_len_old = len(self._param_shapes)

            if i == 0 and cond_param_shapes is None:
                self._num_known_conds = hnet.num_known_conds
            else:
                # We have to enforce this, as we pass the same conditional IDs
                # `cond_id` to the `hnet`'s forward method. We could also
                # check whether `hnet` even accepts conditional inputs and if
                # not, we just don't pass `cond_id`.
                assert self._num_known_conds == hnet.num_known_conds

            for ref in hnet._unconditional_param_shapes_ref:
                self._unconditional_param_shapes_ref.append(ref + ps_len_old)

            if hnet._internal_params is not None:
                if self._internal_params is None:
                    self._internal_params = nn.ParameterList()
                ip_len_old = len(self._internal_params)
                self._internal_params.extend( \
                    nn.ParameterList(hnet._internal_params))
            self._param_shapes.extend(list(hnet._param_shapes))
            for meta in hnet.param_shapes_meta:
                # FIXME Fixed key names will lead to conflicts when stacking
                # multiple containers.
                assert 'celement_type' not in meta.keys() # Container element
                assert 'celement_ind' not in meta.keys()
                assert 'layer' in meta.keys()
                assert 'index' in meta.keys()
                new_meta = dict(meta)
                new_meta['celement_type'] = 'hnet'
                new_meta['celement_ind'] = i
                if i > 0:
                    # FIXME We should properly adjust colliding `layer` IDs.
                    new_meta['layer'] = -1
                new_meta['index'] = meta['index'] + ip_len_old
                self._param_shapes_meta.append(new_meta)

            if hnet._hyper_shapes_learned is not None:
                if self._hyper_shapes_learned is None:
                    self._hyper_shapes_learned = []
                    self._hyper_shapes_learned_ref = []
                self._hyper_shapes_learned.extend( \
                    list(hnet._hyper_shapes_learned))
                for ref in hnet._hyper_shapes_learned_ref:
                    self._hyper_shapes_learned_ref.append(ref + ps_len_old)
            if hnet._hyper_shapes_distilled is not None:
                if self._hyper_shapes_distilled is None:
                    self._hyper_shapes_distilled = []
                self._hyper_shapes_distilled.extend( \
                    list(hnet._hyper_shapes_distilled))

            if self._has_bias is None:
                self._has_bias = hnet._has_bias
            elif self._has_bias != hnet._has_bias:
                self._has_bias = False
                # FIXME We should overwrite the getter and throw an error!
                warn('Some internally maintained hypernetworks use biases, ' +
                     'while others don\'t. Setting attribute "has_bias" to ' +
                     'False.')

            if self._has_fc_out is None:
                self._has_fc_out = hnet._has_fc_out
            else:
                assert self._has_fc_out == hnet._has_fc_out

            if self._has_linear_out is None:
                self._has_linear_out = hnet._has_linear_out
            else:
                assert self._has_linear_out == hnet._has_linear_out

            self._layer_weight_tensors.extend( \
                nn.ParameterList(hnet._layer_weight_tensors))
            self._layer_bias_vectors.extend( \
                nn.ParameterList(hnet._layer_bias_vectors))
            if hnet._batchnorm_layers is not None:
                if self._batchnorm_layers is None:
                    self._batchnorm_layers = nn.ModuleList()
                self._batchnorm_layers.extend( \
                    nn.ModuleList(hnet._batchnorm_layers))
            if hnet._context_mod_layers is not None:
                if self._context_mod_layers is None:
                    self._context_mod_layers = nn.ModuleList()
                self._context_mod_layers.extend( \
                    nn.ModuleList(hnet._context_mod_layers))

        if self._hyper_shapes_distilled is not None:
            raise NotImplementedError('Distillation of parameters not ' +
                                      'supported yet!')

        if hnets is None:
            self._has_bias = False
            
            self._has_fc_out = False
            self._has_linear_out = False

            has_int_cond_weights = cond_param_shapes is not None and \
                not no_cond_weights
            has_int_uncond_weights = uncond_param_shapes is not None and \
                not no_uncond_weights
            self._internal_params = nn.ParameterList() if has_int_cond_weights \
                or has_int_uncond_weights else None
            self._hyper_shapes_learned = None if has_int_cond_weights and \
                has_int_uncond_weights else []
            self._hyper_shapes_learned_ref = None if \
                self._hyper_shapes_learned is None else []
            self._hyper_shapes_distilled is None

        elif cond_param_shapes is not None or uncond_param_shapes is not None:
            self._has_fc_out = False
            self._has_linear_out = False

        ###########################################################
        ### Initialize conditional and unconditional parameters ###
        ###########################################################
        if uncond_param_shapes is not None:
            for i, s in enumerate(uncond_param_shapes):
                if not no_uncond_weights:
                    self._internal_params.append(nn.Parameter( \
                        data=torch.Tensor(*s), requires_grad=True))
                    torch.nn.init.normal_(self._internal_params[-1], mean=0.,
                                          std=.02)
                else:
                    self._hyper_shapes_learned.append(s)
                    self._hyper_shapes_learned_ref.append( \
                        len(self.param_shapes))

                self._unconditional_param_shapes_ref.append( \
                    len(self.param_shapes))

                self._param_shapes.append(s)

                if uncond_param_names is not None:
                    pname = uncond_param_names[i]
                else:
                    pname = 'weight' if len(s) > 1 else 'bias'

                self._param_shapes_meta.append({
                    'name': pname,
                    'index': -1 if no_uncond_weights else \
                        len(self._internal_params)-1,
                    'layer': -1,
                    'celement_type': 'uncond'
                })

        if cond_param_shapes is not None:
            for cind in range(self.num_known_conds):
                for i, s in enumerate(cond_param_shapes):
                    if not no_cond_weights:
                        self._internal_params.append(nn.Parameter( \
                            data=torch.Tensor(*s), requires_grad=True))
                        torch.nn.init.normal_(self._internal_params[-1],
                                              mean=0., std=.02)
                    else:
                        self._hyper_shapes_learned.append(s)
                        self._hyper_shapes_learned_ref.append( \
                            len(self.param_shapes))

                    self._param_shapes.append(s)

                    if cond_param_names is not None:
                        pname = cond_param_names[i]
                    else:
                        pname = 'weight' if len(s) > 1 else 'bias'

                    self._param_shapes_meta.append({
                        'name': pname,
                        'index': -1 if no_cond_weights else \
                            len(self._internal_params)-1,
                        'layer': -1,
                        'celement_type': 'cond',
                        'celement_cind': cind
                    })

        #############################
        ### Finalize construction ###
        #############################
        self._is_properly_setup()
        print('Created Hypernet Container for %d hypernet(s).' \
              % len(self._hnets) + \
              (' Container maintains %d plain unconditional parameter ' \
               % len(self._uncond_ps) if len(self._uncond_ps) > 0 else '') + \
               'tensors.' + \
              (' Container maintains %d plain conditional parameter ' \
               % len(self._cond_ps) + 'tensors for each of %d condiditions.' \
               % self.num_known_conds if len(self._cond_ps) > 0 else ''))
        print(self)

    @property
    def internal_hnets(self):
        """The list of internal hypernetworks provided via constructor argument
        ``hnets``.

        If ``hnets`` was not provided, the attribute is an empty list.

        :type: list
        """
        return self._hnets

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed'):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.mlp_hnet.HMLP.forward`. Some further information
                is provided below.
            uncond_input (optional): Passed to underlying hypernetworks (see
                constructor argument ``hnets``).
            cond_input (optional): Passed to underlying hypernetworks (see
                constructor argument ``hnets``).
            cond_id (int or list, optional): Only passed to underlying
                hypernetworks (see constructor argument ``hnets``) if
                ``cond_input`` is ``None``.
            weights (list or dict, optional): If provided as ``dict`` then
                an additional key ``hnets`` can be specified, which has to a
                list of the same length as the constructor argument ``hnets``
                containing dictionaries as entries that will be concatenated
                to the extracted (hnet-specific) keys ``uncond_weights`` and
                ``cond_weights``.

                For instance, for an instance of class
                :class:`hnets.chunked_mlp_hnet.ChunkedHMLP` the additional key
                ``chunk_embs`` might be added.
            condition (optional): Will be passed to the underlying hypernetworks
                (see constructor argument ``hnets``).

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """
        if distilled_params is not None:
            raise NotImplementedError('Hypernet does not support ' +
                                      '"distilled_params" yet!')

        if len(self._cond_ps) > 0 and cond_id is None:
            raise ValueError('"cond_id" needs to be provided if plain ' +
                             'conditional parameters are maintained.')

        _, _, uncond_weights, cond_weights = \
            self._preprocess_forward_args(_input_required=False,
                _parse_cond_id_fct=None, uncond_input=None, cond_input=None,
                cond_id=None, weights=weights,
                distilled_params=distilled_params, condition=condition,
                ret_format=ret_format)
        up_ref = self.unconditional_param_shapes_ref
        cp_ref = self.conditional_param_shapes_ref
        assert uncond_weights is None or len(up_ref) == len(uncond_weights)
        assert cond_weights is None or len(cp_ref) == len(cond_weights)

        ###############################################################
        ### Split into weights belonging to hnets and plain weights ###
        ###############################################################
        hnets_uncond_weights = [[] for _ in range(len(self._hnets))]
        hnets_cond_weights = [[] for _ in range(len(self._hnets))]
        plain_uncond_weights = []
        plain_cond_weights = [[] for _ in range(self.num_known_conds)]

        for i in range(len(self.param_shapes)):
            meta = self.param_shapes_meta[i]
            if up_ref is not None and i in up_ref:
                idx = up_ref.index(i)
                if meta['celement_type'] == 'hnet':
                    hnets_uncond_weights[meta['celement_ind']].append( \
                        uncond_weights[idx])
                else:
                    assert meta['celement_type'] == 'uncond'
                    plain_uncond_weights.append(uncond_weights[idx])
            else:
                idx = cp_ref.index(i)
                if meta['celement_type'] == 'hnet':
                    hnets_cond_weights[meta['celement_ind']].append( \
                        cond_weights[idx])
                else:
                    assert meta['celement_type'] == 'cond'
                    plain_cond_weights[meta['celement_cind']].append( \
                        cond_weights[idx])

        #####################################
        ### Compute internal hnet outputs ###
        #####################################
        hnet_outs = []

        for i, hnet in enumerate(self._hnets):
            hnet_cond_id = cond_id if cond_input is None else None
            hnet_weights = dict()
            if len(hnets_uncond_weights[i]) > 0:
                hnet_weights['uncond_weights'] = hnets_uncond_weights[i]
            if len(hnets_cond_weights[i]) > 0:
                hnet_weights['cond_weights'] = hnets_cond_weights[i]
            if isinstance(weights, dict) and 'hnets' in weights.keys():
                assert len(weights['hnets']) == len(self._hnets)
                hnet_weights = dict(**hnet_weights, **weights['hnets'][i])
            hnet_out = hnet.forward(uncond_input=uncond_input,
                cond_input=cond_input, cond_id=hnet_cond_id,
                weights=hnet_weights, distilled_params=None,
                condition=condition, ret_format='sequential')
            hnet_outs.append(hnet_out)

        ##################################
        ### Assemble final hnet output ###
        ##################################
        batch_size = None
        if cond_id is not None:
            if isinstance(cond_id, int):
                batch_size = 1
                cond_id = [cond_id]
            else:
                batch_size = len(cond_id)
        for hout in hnet_outs:
            # FIXME Should we enforce that the length of `cond_id` is equal to
            # the batch size processed by the internal hnets?
            if batch_size is None or batch_size == 1:
                batch_size = len(hout)
            elif len(hout) > 1:
                assert batch_size == len(hout)
        if batch_size is None:
            # Can happen if 'cond_id` is `None` and we have no internal hnets.
            batch_size = 1

        full_hnet_out = []
        for i in range(batch_size):
            list_of_hnet_tensors = []
            for hout in hnet_outs:
                if len(hout) == 1:
                    list_of_hnet_tensors.append(hout[0])
                else:
                    list_of_hnet_tensors.append(hout[i])
            uncond_tensors = plain_uncond_weights
            cond_tensors = []
            if cond_id is not None:
                if len(cond_id) == 1:
                    cond_tensors = plain_cond_weights[cond_id[0]]
                else:
                    cond_tensors = plain_cond_weights[cond_id[i]]
            full_hnet_out.append(self._assembly_fct(list_of_hnet_tensors,
                uncond_tensors, cond_tensors))

            # Sanity check.
            if i == 0:
                outs = full_hnet_out[-1]
                assert len(outs) == len(self.target_shapes)
                for i, s in enumerate(self.target_shapes):
                    assert np.all(np.equal(outs[i].shape, s))

        ########################################
        ### Convert to correct output format ###
        ########################################
        ret = full_hnet_out
        assert ret_format in ['flattened', 'sequential', 'squeezed']
        if ret_format == 'sequential':
            return ret
        elif ret_format == 'squeezed':
            if batch_size == 1:
                return ret[0]
            return ret

        flat_ret = [None] * batch_size
        for bind in range(batch_size):
            for i, tensor in enumerate(ret[bind]):
                if i == 0:
                    flat_ret[bind] = tensor.flatten()
                else:
                    flat_ret[bind] = \
                        torch.cat([flat_ret[bind], tensor.flatten()], dim=0)

        return torch.stack(flat_ret, dim=0)

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This network does not have any distillation targets.

        Returns:
            ``None``
        """
        return None

if __name__ == '__main__':
    pass


