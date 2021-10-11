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
# @title          :hnets/structured_mlp_hnet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :04/17/2020
# @version        :1.0
# @python_version :3.6.10
"""
Structured Chunked MLP - Hypernetwork
-------------------------------------

The module :mod:`hnets.structured_mlp_hnet` contains a `Structured Chunked
Hypernetwork`, i.e., a hypernetwork that is aware of the target network
architecture and choses a smart way of chunking.

In contrast to the `Chunked Hypernetwork`
:class:`hnets.chunked_mlp_hnet.ChunkedHMLP`, which just flattens the
``target_shapes`` and splits them into equally sized chunks (ignoring the
underlying network structure in terms of layers or type of weight (bias, kernel,
...)), the :class:`StructuredHMLP` aims to preserve this structure when chunking
the target weights.

Example:
    Assume ``target_shapes = [[3], [3], [10, 5], [10], [20, 5], [20]]``.

    There are now many ways to split those weights into chunks. In the simplest
    case, we consider only one chunk and produce all weights at once with a
    `Full Hypernetwork` :class:`hnets.mlp_hnet.HMLP`.

    Another simple scenario would be to realize that all shapes except the first
    two are different. So, we create a total of 5 internal hypernetworks for
    those 6 weight tensors, where the first internal hypernetwork would produce
    weights of shape ``[3]`` upon receiving an external input plus an internal
    chunk embedding. See below for an example instantiation:

    .. code-block:: python

        def assembly_fct(list_of_chunks):
            assert len(list_of_chunks) == 4
            ret = []
            for chunk in list_of_chunks:
                ret.extend(chunk)
            return ret

        hnet = StructuredHMLP([[3], [3], [10, 5], [10], [20, 5], [20]],
            [[[3]], [[10, 5], [10]], [[20, 5], [20]]], [2, 1, 1], 8,
            {'layers': [10,10]}, assembly_fct, cond_chunk_embs=True,
            uncond_in_size=0, cond_in_size=0, verbose=True,
            no_uncond_weights=False, no_cond_weights=False, num_cond_embs=1)

    A smarter way of chunking would be to realize that the last two shapes are
    just twice the middle two shapes. Hence, we could instantiate two internal
    hypernetworks. The first one would be used to produce tensors of shape
    ``[3]`` and therefore require 2 chunk embeddings. The second internal
    hypernetwork would be used to create tensors of shape ``[10, 5], [10]``,
    requiring 3 chunk embeddings (the last two chunks together make up the last
    two target tensors of shape ``[20, 5], [20]``).

    .. code-block:: python

        def assembly_fct(list_of_chunks):
            assert len(list_of_chunks) == 5
            ret = [*list_of_chunks[0], *list_of_chunks[1], *list_of_chunks[2]]
            for t, tensor in enumerate(list_of_chunks[3]):
                ret.append(torch.cat([tensor, list_of_chunks[4][t]], dim=0))
            return ret

        hnet = StructuredHMLP([[3], [3], [10, 5], [10], [20, 5], [20]],
            [[[3]], [[10, 5], [10]]], [2, 3], 8,
            {'layers': [10,10]}, assembly_fct, cond_chunk_embs=True,
            uncond_in_size=0, cond_in_size=0, verbose=True,
            no_uncond_weights=False, no_cond_weights=False, num_cond_embs=1)

Example:
    This hypernetwork can also be used to realize soft-sharing via templates as
    proposed in `Savarese et al. <https://arxiv.org/abs/1902.09701>`__

    Assume a target network with 3 layers of identical weight shapes
    ``target_shapes=[s, s, s]``, where ``s`` denotes a weight shape.

    If we want to create these 3 weight tensors via a linear combination of two
    templates, we could create an instance of :class:`StructuredHMLP` as
    follows:

    .. code-block:: python

        def assembly_fct(list_of_chunks):
            assert len(list_of_chunks) == 3
            return [list_of_chunks[0][0], list_of_chunks[1][0],
                    list_of_chunks[2][0]]

        hnet = StructuredHMLP([s, s, s], [[s]], [3], 2,
            {'layers': [], 'use_bias': False}, assembly_fct
            cond_chunk_embs=True, uncond_in_size=0, cond_in_size=0,
            verbose=True, no_uncond_weights=False, no_cond_weights=False,
            num_cond_embs=1)

    There will be one underlying linear hypernetwork, that expects a
    2-dimensional embedding input. The computation of the linear hypernetwork
    can be seen as :math:`t_i = W e_i`. Where :math:`t_i` is a tensor of shape
    ``s`` containing the weights of the :math:`i`-th chunk (with chunk embedding
    :math:`e_i`).

    The 2 templates are encoded in the hypernetwork weights :math:`W`, whereas
    the chunk embedding represents the coefficients of the linear combination.
"""
import numpy as np
import torch
import torch.nn as nn
from warnings import warn

from hypnettorch.hnets.hnet_interface import HyperNetInterface
from hypnettorch.hnets.mlp_hnet import HMLP

class StructuredHMLP(nn.Module, HyperNetInterface):
    """Implementation of a `structured chunked fully-connected hypernet`.

    This network builds a series of full hypernetworks internally (hidden from
    the user). There will be one internal hypernetwork for each element of
    ``chunk_shapes``. Those internal hypernetworks can produce an arbitrary
    amount of chunks (as defined by ``num_per_chunk``). All those chunks are
    finally assembled by function ``assembly_fct`` to produce tensors according
    to ``target_shapes``.

    Note:
        It is possible to set ``uncond_in_size`` and ``cond_in_size`` to zero
        if ``cond_chunk_embs`` is ``True`` and there are no zeroes in argument
        ``chunk_emb_sizes``.

    Args:
        (....): See constructor arguments of class
            :class:`hnets.mlp_hnet.HMLP`.
        chunk_shapes (list): List of lists of lists of integers. Each chunk will
            be produced by its own internal hypernetwork (instance of class
            :class:`hnets.mlp_hnet.HMLP`). Hence, this list can be seen as a
            list of ``target_shapes``, passed to the underlying internal
            hypernets.
        num_per_chunk (list): List of the same length as ``chunk_shapes``, that
            determines how often each of these chunks has to be produced.
        chunk_emb_sizes (list or int): List with the same length as
            ``chunk_shapes`` or single integer that will be expanded to this
            length. Determines the chunk embedding size per internal
            hypernetwork.

            Note:
                Embeddings will be initialized with a normal distribution using
                zero mean and unit variance.

            Note:
                If the corresponding entry in ``num_per_chunk`` is ``1``, then
                an embedding size might be ``0``, which means there won't be
                chunk embeddings for the corresponding internal hypernetwork.
        hmlp_kwargs (list or dict): List of dictionaries or a single dictionary
            that will be expanded to such a list. Those dictionaries may contain
            keyword arguments for each instance of class
            :class:`hnets.mlp_hnet.HMLP` that will be generated.

            The following keys are **not permitted** in these dictionaries:
            - ``uncond_in_size``
            - ``cond_in_size``
            - ``no_uncond_weights``
            - ``no_cond_weights``
            - ``num_cond_embs``
            Those arguments will be determined by the corresponding keyword
            arguments of this class!
        assembly_fct (func): A function handle that takes the produced chunks
            and converts them into tensors with shapes ``target_shapes``.

            The function handle must have the signature:
            ``assembly_fct(list_of_chunks)``.
            The argument ``list_of_chunks`` is a list of lists of tensors. The
            function is expected to return a list of tensors, each of them
            having a shape as specified by ``target_shapes``.

            Example:
                Assume ``chunk_shapes=[[[3]], [[10, 5], [5]]]`` and
                ``num_per_chunk=[2, 1]``. Then the argument ``list_of_chunks``
                will be a list of lists of tensors as follows:
                ``[[tensor(3)], [tensor(3)], [tensor(10, 5), tensor(5)]]``.

                If ``target_shapes=[[3], [3], [10, 5], [5]]``, then the output
                of ``assembly_fct`` is expected to be a list of tensors as
                follows: ``[tensor(3), tensor(3), tensor(10, 5), tensor(5)]``.

            Note:
                This function considers one sample at a time, even if a batch
                of inputs is processed.

            Note:
                It is assumed that ``assembly_fct`` does not further process the
                incoming weights. Otherwise, the attributes
                :attr:`mnets.mnet_interface.MainNetInterface.has_fc_out` and
                :attr:`mnets.mnet_interface.MainNetInterface.has_linear_out`
                might be invalid.
        cond_chunk_embs (bool): See documentation of class
            :class:`hnets.chunked_mlp_hnet.ChunkedHMLP`
    """
    def __init__(self, target_shapes, chunk_shapes, num_per_chunk,
                 chunk_emb_sizes, hmlp_kwargs, assembly_fct,
                 cond_chunk_embs=False, uncond_in_size=0, cond_in_size=8,
                 verbose=True, no_uncond_weights=False, no_cond_weights=False,
                 num_cond_embs=1):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        HyperNetInterface.__init__(self)

        ### Basic checks for user inputs ###
        assert isinstance(chunk_shapes, (list, tuple)) and len(chunk_shapes) > 0
        num_chunk_weights = 0
        for chunk in chunk_shapes: # Each chunk is a list of shapes!
            assert isinstance(chunk, (list, tuple)) and len(chunk) > 0
            num_chunk_weights += StructuredHMLP.shapes_to_num_weights(chunk)
        num_trgt_weights = StructuredHMLP.shapes_to_num_weights(target_shapes)
        if num_trgt_weights > num_chunk_weights:
            # TODO Should we display a warning? The user might actively want
            # to reuse the same weights in the target network. In the end, the
            # user should be completely free on how he assembles the chunks to
            # weights within the `assembly_fct`.
            pass
        assert isinstance(num_per_chunk, (list, tuple)) and \
            len(num_per_chunk) == len(chunk_shapes)
        if 0 in num_per_chunk:
            raise ValueError('Option "num_per_chunk" may not contains 0s. ' +
                             'Each internal hypernetwork must create at ' +
                             'least one chunk!')
        assert isinstance(chunk_emb_sizes, (int, list, tuple))
        if isinstance(chunk_emb_sizes, int):
            chunk_emb_sizes = [chunk_emb_sizes] * len(chunk_shapes)
        assert len(chunk_emb_sizes) == len(chunk_shapes)
        if 0 in chunk_emb_sizes and uncond_in_size == 0 and cond_in_size == 0:
            raise ValueError('Argument "chunk_emb_sizes" may not contain ' +
                             '0s if "uncond_in_size" and "cond_in_size" are ' +
                             '0!')
        for i, s in enumerate(chunk_emb_sizes):
            if s == 0 and num_per_chunk[i] != 1:
                raise ValueError('Option "chunk_emb_sizes" may only contain ' +
                                 'zeroes if the corresponding entry in ' +
                                 '"num_per_chunk" is 1.')
        assert isinstance(hmlp_kwargs, (dict, list, tuple))
        if isinstance(hmlp_kwargs, dict):
            hmlp_kwargs = [dict(hmlp_kwargs) for _ in range(len(chunk_shapes))]
        assert len(hmlp_kwargs) == len(chunk_shapes)
        for hkwargs in hmlp_kwargs:
            assert isinstance(hkwargs, dict)
            forbidden = ['uncond_in_size', 'cond_in_size', 'no_uncond_weights',
                         'no_cond_weights', 'num_cond_embs']
            for kw in forbidden:
                if kw in hkwargs.keys():
                    raise ValueError('Key %s may not be passed with argument ' \
                                     % kw + '"hmlp_kwargs"!')

            if 'verbose' not in hkwargs.keys():
                hkwargs['verbose'] = False

        ### Make constructor arguments internally available ###
        self._chunk_shapes = chunk_shapes
        self._num_per_chunk = num_per_chunk
        self._chunk_emb_sizes = chunk_emb_sizes
        #self._hkwargs = hkwargs
        self._assembly_fct = assembly_fct
        self._cond_chunk_embs = cond_chunk_embs
        self._uncond_in_size = uncond_in_size
        self._cond_in_size = cond_in_size
        self._no_uncond_weights = no_uncond_weights
        self._no_cond_weights = no_cond_weights
        self._num_cond_embs = num_cond_embs

        ### Create underlying full hypernets ###
        num_hnets = len(chunk_shapes)
        self._hnets = []

        for i in range(num_hnets):
            # Note, even if chunk embeddings are considered conditional, they
            # are maintained in this object and just fed as an external input
            # to the underlying hnet.
            hnet_uncond_in_size = uncond_in_size + chunk_emb_sizes[i]

            # Conditional inputs (`cond_in_size`) will be maintained by the
            # first internal hypernetwork.
            if i == 0:
                hnet_no_cond_weights = no_cond_weights
                hnet_num_cond_embs = num_cond_embs
                if cond_chunk_embs and cond_in_size == 0:
                    # If there are no other conditional embeddings except the
                    # chunk embeddings, we tell the first underlying hnet
                    # explicitly that it doesn't need to maintain any
                    # conditional weights to avoid that it will throw a warning.
                    hnet_num_cond_embs = 0
            else:
                # All other hypernetworks will be passed the conditional
                # embeddings from the first hypernet as input.
                hnet_no_cond_weights = True
                hnet_num_cond_embs = 0

            self._hnets.append(HMLP(chunk_shapes[i],
                uncond_in_size=hnet_uncond_in_size, cond_in_size=cond_in_size,
                no_uncond_weights=no_uncond_weights,
                no_cond_weights=hnet_no_cond_weights,
                num_cond_embs=hnet_num_cond_embs, **hmlp_kwargs[i]))

        ### Setup attributes required by interface ###
        # Most of these attributes are taken over from the internally
        # maintained hypernetworks.
        self._target_shapes = target_shapes
        self._num_known_conds = self._num_cond_embs

        # As we just append the weights of the internal hypernets we will have
        # output weights all over the place.
        # Additionally, it would be complicated to assign outputs to target
        # outputs, as we do not know, what is happening in the `assembly_fct`.
        # Also, keep in mind that we will append chunk embeddings at the end
        # of `param_shapes`.
        self._mask_fc_out = False

        self._unconditional_param_shapes_ref = []
        self._param_shapes = []
        self._param_shapes_meta = []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        for i, hnet in enumerate(self._hnets):
            # Note, it is important to convert lists into new object and not
            # just copy references!
            # Note, we have to adapt all references if `i > 0`.

            ps_len_old = len(self._param_shapes)

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
                assert 'hnet_ind' not in meta.keys()
                assert 'layer' in meta.keys()
                assert 'index' in meta.keys()
                new_meta = dict(meta)
                new_meta['hnet_ind'] = i
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

        ### Create chunk embeddings ###
        if cond_in_size == 0 and uncond_in_size == 0 and 0 in chunk_emb_sizes:
            raise ValueError('At least one internal hypernetwork has no ' +
                             'chunk embedding(s). Therefore, the input size ' +
                             'might not be 0.')
        if cond_in_size == 0 and uncond_in_size == 0 and not cond_chunk_embs:
            # Note, we could also allow this case. It would be analoguous to
            # creating a full hypernet with no unconditional input and one
            # conditional embedding. But the user can explicitly achieve that
            # as noted below.
            raise ValueError('If no external (conditional or unconditional) ' +
                             'input is provided to the hypernetwork, then ' +
                             'it can only learn a fixed output. If this ' +
                             'behavior is desired, please enable ' +
                             '"cond_chunk_embs" and set "num_cond_embs=1".')

        chunk_emb_shapes = []
        # To which internal hnet does the corresponding chunk shape belong to.
        chunk_emb_refs = []
        for i, size in enumerate(chunk_emb_sizes):
            if size == 0:
                # No chunk embeddings for internal hnet `i`.
                continue

            chunk_emb_refs.append(i)
            assert num_per_chunk[i] > 0
            chunk_emb_shapes.append([num_per_chunk[i], size])
        self._chunk_emb_shapes = chunk_emb_shapes
        self._chunk_emb_refs = chunk_emb_refs

        # How often do we have to instantiate the chunk embeddings prescribed by
        # `chunk_emb_shapes`?
        num_cemb_weights = 1
        no_cemb_weights = no_uncond_weights
        if cond_chunk_embs:
            num_cemb_weights = num_cond_embs
            no_cemb_weights = no_cond_weights

        # Number of conditional and unconditional parameters so far.
        tmp_num_uncond = len(self._unconditional_param_shapes_ref)
        tmp_num_cond = len(self._param_shapes) - tmp_num_uncond

        # List of lists of inds.
        # Indices of chunk embedding per condition within
        # `conditional_param_shapes`, if chunk embeddings are conditional.
        # Otherwise, indices of chunk embeddings within
        # `unconditional_param_shapes`.
        self._chunk_emb_inds = [[] for _ in range(num_cemb_weights)]

        for i in range(num_cemb_weights):
            for j, shape in enumerate(chunk_emb_shapes):
                if not no_cemb_weights:
                    self._internal_params.append(nn.Parameter( \
                        data=torch.Tensor(*shape), requires_grad=True))
                    torch.nn.init.normal_(self._internal_params[-1], mean=0.,
                                          std=1.)
                else:
                    self._hyper_shapes_learned.append(shape)
                    self._hyper_shapes_learned_ref.append( \
                        len(self.param_shapes))

                if not cond_chunk_embs:
                    self._unconditional_param_shapes_ref.append( \
                        len(self.param_shapes))

                self._param_shapes.append(shape)
                # In principle, these embeddings also belong to the input, so we
                # just assign them as "layer" 0 (note, the underlying hnets use
                # the same layer ID for its embeddings.
                self._param_shapes_meta.append({
                    'name': 'embedding',
                    'index': -1 if no_cemb_weights else \
                        len(self._internal_params)-1,
                    'layer': 0,
                    'info': 'chunk embeddings',
                    'hnet_ind': chunk_emb_refs[j],
                    'cond_id': i if cond_chunk_embs else -1
                })

                if cond_chunk_embs:
                    self._chunk_emb_inds[i].append(tmp_num_cond)
                    tmp_num_cond += 1
                else:
                    self._chunk_emb_inds[i].append(tmp_num_uncond)
                    tmp_num_uncond += 1

        assert len(self.param_shapes) == tmp_num_uncond + tmp_num_cond

        ### Finalize construction ###
        self._is_properly_setup()

        if verbose:
            print('Created Structured Chunked MLP Hypernet.')
            print('It manages %d full hypernetworks internally that produce ' \
                  % (num_hnets) + '%s chunks in total.' % (self.num_chunks))
            print('The internal hypernetworks have a combined output size of ' +
                  '%d compared to %d weights produced by this network.' \
                  % (num_chunk_weights, self.num_outputs))
            print(self)

    @property
    def num_chunks(self):
        """The total number of chunks that make up the hypernet output.

        This attribute simply corresponds to ``np.sum(num_per_chunk)``.

        :type: int
        """
        return int(np.sum(self._num_per_chunk))

    @property
    def chunk_emb_shapes(self):
        """List of lists of integers. The list contains the shape of the chunk
        embeddings required per forward sweep.

        Note:
            Some internal hypernets might not need chunk embeddings if the
            corresponding entry in ``chunk_emb_sizes`` is zero.

        :type: list
        """
        return self._chunk_emb_shapes

    @property
    def cond_chunk_embs(self):
        """Whether chunk embeddings are unconditional (``False``) or conditional
        (``True``) parameters. See constructor argument ``cond_chunk_embs``.

        :type: bool
        """
        return self._cond_chunk_embs

    @property
    def internal_hnets(self):
        """ The list of internal hypernetworks (instances of class
        :class:`hnets.mlp_hnet.HMLP`) which are created to produce the
        individual chunks according to constructor argument ``chunk_shapes``.

        :type: list
        """
        return self._hnets

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed'):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.mlp_hnet.HMLP.forward`.
            weights (list or dict, optional): If provided as ``dict`` and
                chunk embeddings are considered conditional (see constructor
                argument ``cond_chunk_embs``), then the additional key
                ``chunk_embs`` can be used to pass a batch of chunk embeddings.
                This option is mutually exclusive with the option of passing
                ``cond_id``. Note, if conditional inputs via ``cond_input`` are
                expected, then the batch sizes must agree.

                A batch of chunk embeddings is expected to be a list of tensors
                of shape
                ``[B, *ce_shape]``, where ``B`` denotes the batch size and
                ``ce_shape`` is a shape from list :attr:`chunk_emb_shapes`.

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """
        if distilled_params is not None:
            raise NotImplementedError('Hypernet does not support ' +
                                      '"distilled_params" yet!')

        # Note, the network does not necessarily have chunk embeddings.
        has_chunk_embs = len(self.chunk_emb_shapes) > 0

        cond_chunk_embs = None
        if isinstance(weights, dict):
            if 'chunk_embs' in weights.keys():
                assert has_chunk_embs
                cond_chunk_embs = weights['chunk_embs']
                if not self._cond_chunk_embs:
                    raise ValueError('Key "chunk_embs" for argument ' +
                                     '"weights" is only allowed if chunk ' +
                                     'embeddings are conditional.')
                assert isinstance(cond_chunk_embs, (list, tuple))
                batch_size = None
                for i, s in self.chunk_emb_shapes:
                    assert len(cond_chunk_embs[i].shape) == 3 and \
                        np.all(np.equal(cond_chunk_embs.shape[1:], s))
                    if i == 0:
                        batch_size = cond_chunk_embs[i].shape[0]
                    else:
                        assert cond_chunk_embs[i].shape[0] == batch_size

                if cond_id is not None:
                    raise ValueError('Option "cond_id" is mutually exclusive ' +
                                     'with key "chunk_embs" for argument ' +
                                     '"weights".')
                assert cond_input is None or \
                    cond_input.shape[0] == batch_size

                # Remove `chunk_embs` from dictionary, since upper class parser
                # doesn't know how to deal with it.
                del weights['chunk_embs']
                if len(weights.keys()) == 0: # Empty dictionary.
                    weights = None

        if cond_input is not None and self._cond_chunk_embs and \
                has_chunk_embs and cond_chunk_embs is None:
            raise ValueError('Conditional chunk embeddings have to be ' +
                             'provided via "weights" if "cond_input" is ' +
                             'specified.')

        _input_required = self._cond_in_size > 0 or self._uncond_in_size > 0
        # We parse `cond_id` afterwards if chunk embeddings are also
        # conditional.
        if self._cond_chunk_embs:
            _parse_cond_id_fct = lambda x, y, z: None
        else:
            _parse_cond_id_fct = None

        uncond_input, cond_input, uncond_weights, cond_weights = \
            self._preprocess_forward_args(_input_required=_input_required,
                _parse_cond_id_fct=_parse_cond_id_fct,
                uncond_input=uncond_input, cond_input=cond_input,
                cond_id=cond_id, weights=weights,
                distilled_params=distilled_params, condition=condition,
                ret_format=ret_format)
                #ext_inputs=ext_inputs, task_emb=task_emb,
                #task_id=task_id, theta=theta, dTheta=dTheta)

        ### Translate IDs to conditional inputs ###
        if cond_id is not None and self._cond_chunk_embs:
            assert cond_input is None and cond_chunk_embs is None
            cond_id = [cond_id] if isinstance(cond_id, int) else cond_id

            if cond_weights is None:
                raise ValueError('Forward option "cond_id" can only be ' +
                                 'used if conditional parameters are ' +
                                 'maintained internally or passed to the ' +
                                 'forward method via option "weights".')

            if has_chunk_embs:
                cond_chunk_embs = [[] for _ in \
                                   range(len(self.chunk_emb_shapes))]
            cond_input = [] if self._cond_in_size > 0 else None

            for i, cid in enumerate(cond_id):
                if cid < 0 or cid >= self._num_cond_embs:
                    raise ValueError('Condition %d not existing!' % (cid))

                # Note, we do not necessarily have conditional embeddings.
                if self._cond_in_size > 0:
                    cond_input.append(cond_weights[cid])

                for j, pind in enumerate(self._chunk_emb_inds[cid]):
                    cond_chunk_embs[j].append(cond_weights[pind])

            if self._cond_in_size > 0:
                cond_input = torch.stack(cond_input, dim=0)

            for i in range(len(self.chunk_emb_shapes)):
                cond_chunk_embs[i] = torch.stack(cond_chunk_embs[i], dim=0)

        ### Finalize input parsing ###
        batch_size = None
        if cond_input is not None:
            batch_size = cond_input.shape[0]
        if cond_chunk_embs is not None:
            assert batch_size is None or batch_size == \
                cond_chunk_embs[0].shape[0]
            batch_size = cond_chunk_embs[0].shape[0]
        if uncond_input is not None:
            if batch_size is None:
                batch_size = uncond_input.shape[0]
            else:
                assert batch_size == uncond_input.shape[0]
        assert batch_size is not None

        chunk_embs = None
        if self._cond_chunk_embs:
            assert cond_chunk_embs is not None or not has_chunk_embs
            assert self._cond_in_size == 0 or cond_input is not None
            chunk_embs = cond_chunk_embs
        else:
            assert cond_chunk_embs is None
            chunk_embs = []
            for i, pind in enumerate(self._chunk_emb_inds[0]):
                chunk_embs.append(uncond_weights[pind])
                # Insert batch dimension.
                chunk_embs[-1] = chunk_embs[-1].expand(batch_size, \
                    *self.chunk_emb_shapes[i])

        # We now have the following setup:
        # cond_input: [batch_size, cond_in_size] or None
        # uncond_input: [batch_size, uncond_in_size] or None
        # chunk_embs is a list with an entry for all hypernets `i`, that have
        # chunk embeddings, the list has a tensor of shape:
        # [batch_size, num_chunks[i], chunk_emb_size[i]]

        ### Compute output chunks ###
        # I.e., iterate over internal hypernets.

        # A list of chunks for each sample in the input batch. Those will be
        # later processed by the `assembly_fct`.
        chunks = [[] for _ in range(batch_size)]

        cemb_ind = 0
        for i, hnet in enumerate(self._hnets):
            ### Assemble input for i-th hypernet ###
            requires_cemb_input = i in self._chunk_emb_refs

            if requires_cemb_input:
                # Append chunk embeddings to unconditional input.
                ce_shape = self.chunk_emb_shapes[cemb_ind]
                curr_chunk_embs = chunk_embs[cemb_ind]
                num_chunks = ce_shape[0]

                # We now first copy the hypernet inputs for each chunk, arriving
                # at
                # cond_input: [batch_size, num_chunks, cond_in_size] or None
                # uncond_input: [batch_size, num_chunks, uncond_in_size] or None
                hnet_cond_input = None
                if cond_input is not None:
                    hnet_cond_input = cond_input.reshape(batch_size, 1, -1)
                    hnet_cond_input = hnet_cond_input.expand(batch_size,
                        num_chunks, self._cond_in_size)
                if uncond_input is not None:
                    hnet_uncond_input = uncond_input.reshape(batch_size, 1, -1)
                    hnet_uncond_input = hnet_uncond_input.expand(batch_size,
                        num_chunks, self._uncond_in_size)
                    # The chunk embeddings are considered unconditional inputs
                    # to the underlying hypernetwork.
                    hnet_uncond_input = torch.cat([hnet_uncond_input,
                                                   curr_chunk_embs], dim=2)
                else:
                    hnet_uncond_input = curr_chunk_embs

                # Now we build one big batch for the underlying hypernetwork,
                # with batch size: `batch_size * num_chunks`.
                if hnet_cond_input is not None:
                    hnet_cond_input = hnet_cond_input.reshape( \
                        batch_size * num_chunks, -1)
                hnet_uncond_input = hnet_uncond_input.reshape( \
                    batch_size * num_chunks, -1)

                cemb_ind += 1
            else:
                num_chunks = 1
                hnet_cond_input = cond_input
                hnet_uncond_input = uncond_input

            ### Extract weights for i-th hypernet ###
            hnet_weights = dict()
            # Note, only the first hnet has its own conditional weights.
            if i == 0:
                if cond_weights is not None and self._cond_chunk_embs:
                    hnet_weights['cond_weights'] = \
                        self._hnets[0].conditional_params
                elif cond_weights is not None:
                    hnet_weights['cond_weights'] = cond_weights
            assert uncond_weights is not None
            hnet_weights['uncond_weights'] = []
            assert len(uncond_weights) == \
                len(self.unconditional_param_shapes_ref)
            for j, ref in enumerate(self.unconditional_param_shapes_ref):
                meta = self.param_shapes_meta[ref]
                if 'hnet_ind' in meta.keys() and meta['hnet_ind'] == i:
                    if 'info' in meta.keys() and \
                            meta['info'] == 'chunk embeddings':
                        continue
                    hnet_weights['uncond_weights'].append(uncond_weights[j])

            ### Process i-th chunks ###
            hnet_out = hnet.forward(uncond_input=hnet_uncond_input,
                cond_input=hnet_cond_input, cond_id=None, weights=hnet_weights,
                distilled_params=None, condition=condition,
                ret_format='sequential')
            assert isinstance(hnet_out, list) and \
                len(hnet_out) == batch_size * num_chunks
            for bind in range(batch_size):
                for cind in range(num_chunks):
                    chunks[bind].append(hnet_out[bind*num_chunks + cind])

        ### Retrieve hypernet output ###
        ret = []
        for bind in range(batch_size):
            assert len(chunks[bind]) == self.num_chunks
            ret.append(self._assembly_fct(chunks[bind]))
            if bind == 0:
                outs = ret[-1]
                assert len(outs) == len(self.target_shapes)
                for i, s in enumerate(self.target_shapes):
                    assert np.all(np.equal(outs[i].shape, s))

        ### Convert to correct output format ###
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

    def get_cond_in_emb(self, cond_id):
        """Get the ``cond_id``-th (conditional) input embedding.

        Args:
            (....): See docstring of method
                :meth:`hnets.mlp_hnet.HMLP.get_cond_in_emb`.

        Returns:
            (torch.nn.Parameter)
        """
        return self._hnets[0].get_cond_in_emb(cond_id)

    def get_chunk_embs(self, cond_id=None):
        """Get the chunk embeddings.

        Args:
            cond_id (int): Is mandatory if constructor argument
                ``cond_chunk_embs`` was set. Determines the set of chunk
                embeddings to be considered.

        Returns:
            (list): A list of tensors with shapes prescribed by
            :attr:`chunk_emb_shapes`.
        """
        ret = []
        if self._cond_chunk_embs:
            if cond_id is None:
                raise RuntimeError('Option "cond_id" has to be set if chunk ' +
                                   'embeddings are conditional parameters!')
            if self.conditional_params is None:
                raise RuntimeError('Conditional chunk embeddings are not ' +
                                   'internally maintained!')
            if not isinstance(cond_id, int) or cond_id < 0 or \
                    cond_id >= self._num_cond_embs:
                raise RuntimeError('Option "cond_id" must be between 0 and ' +
                                   '%d!' % (self._num_cond_embs-1))

        else:
            assert cond_id is None
            if self.unconditional_params is None:
                raise RuntimeError('Chunk embeddings are not internally ' +
                                   'maintained!')

        for meta in self.param_shapes_meta:
            if 'info' in meta.keys() and meta['info'] == 'chunk embeddings':
                if cond_id is not None:
                    assert meta['cond_id'] != -1
                    if cond_id != meta['cond_id']:
                        continue
                assert meta['index'] != -1

                shape = self.chunk_emb_shapes[len(ret)]
                ret.append(self.internal_params[meta['index']])
                assert np.all(np.equal(ret[-1].shape, shape))

        return ret

if __name__ == '__main__':
    pass


