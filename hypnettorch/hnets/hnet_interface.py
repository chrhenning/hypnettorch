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
# @title          :hnets/hnet_interface.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :09/20/2019
# @version        :1.0
# @python_version :3.6.8
"""
Hypernetwork Interface
----------------------

The module :mod:`hnets.hnet_interface` contains an interface for hypernetworks.

A hypernetworks is a special type of neural network that produces the weights of
another neural network (called the main or target networks, see
:mod:`mnets.mnet_interface`). The name "hypernetworks" was introduced in

    `Ha et al., "Hypernetworks", 2016. <https://arxiv.org/abs/1609.09106>`

The interface ensures that we can consistently use different types of these
networks without knowing their specific implementation details (as long as we
only use functionalities defined in class :class:`HyperNetInterface`).
"""
from abc import abstractmethod
import numpy as np
import torch
from warnings import warn

from hypnettorch.mnets.mnet_interface import MainNetInterface

class HyperNetInterface(MainNetInterface):
    r"""A general interface for hypernetworks.

    Note:
        Previous implementations of hypernetworks used the deprecated interface
        :mod:`utils.module_wrappers.CLHyperNetInterface`, which was specialized
        for the design of task-conditioned hypernetworks. This new interface
        is more general but includes all previous capabilities.
    """
    def __init__(self):
        super(HyperNetInterface, self).__init__()

        ### IMPORTANT NOTE FOR DEVELOPERS IMPLEMENTING THIS INTERFACE ###
        # The following member variables have to be set by all classes  #
        # that implement this interface (IN ADDITION to the attributes  #
        # that must be set according to the base interface              #
        # MainNetInterface).                                            #
        # Please always verify your implementation using the method     #
        # `_is_properly_setup` at the end the constructor of any class  #
        # implementing this interface.                                  #
        #################################################################
        # The indices of parameter shapes within `param_shapes` that are
        # associated with parameters that are considered "unconditional". The
        # remaining indices (not in this list) are considered to be associated
        # with conditional parameters.
        self._unconditional_param_shapes_ref = None # list
        # The output shapes of the hypernetwork.
        self._target_shapes = None # list
        # The maximum `cond_id` that can be passed to the forward method will be
        # `self._num_known_conds-1`.
        self._num_known_conds = None

        ### THE FOLLOWING ATTRIBUTES WILL BE SET AUTOMATICALLY IF ###
        ### `_param_shapes_meta` IS PROPERLY SETUP.               ###
        # Otherwise, they have to be set manually!                  #
        #############################################################
        # The indices of parameters within `internal_params` that are considered
        # "unconditional". The remaining indices (not in this list) are
        # considered conditional parameters.
        # Note, if `param_shapes_meta` is specified then in conjunction with
        # `unconditional_param_shapes_ref` we can automatically infer which
        # parameters inside `internal_params` are unconditional.
        # Attribute is a list of `len(unconditional_params) > 0` else `None`.
        self._unconditional_params_ref = None # list or None

        ### THE FOLLOWING ATTRIBUTES WILL BE SET AUTOMATICALLY ###
        # Please do not overwrite the default values.            #
        ##########################################################
        # ... (reserved for future use)

    def _is_properly_setup(self):
        """This method can be used by classes that implement this interface to
        check whether all required properties have been set."""

        MainNetInterface._is_properly_setup(self)

        assert isinstance(self._num_known_conds, int) and \
            self._num_known_conds >= 0

        assert isinstance(self._unconditional_param_shapes_ref, list)
        np_ups_ref = np.unique(self._unconditional_param_shapes_ref)
        assert self.internal_params is not None
        assert len(np_ups_ref) == len(self._unconditional_param_shapes_ref)
        assert np.all(np_ups_ref >= 0) and \
               np.all(np_ups_ref < len(self.param_shapes))

        if self._unconditional_params_ref is not None:
            np_up_ref = np.unique(self._unconditional_params_ref)
            assert self.internal_params is not None
            assert len(np_up_ref) == len(self._unconditional_params_ref)
            assert np.all(np_up_ref >= 0) and \
                   np.all(np_up_ref < len(self.internal_params))
        elif self.internal_params is not None:
            # Note, it might be that `internal_params` only contains conditional
            # parameters, thus `unconditional_params_ref` is intentionally left
            # to be `None`. However, without having access to
            # `param_shapes_meta` we have no way to check this. Therefore, we
            # require its existence.
            assert self._param_shapes_meta is not None, \
                '"_unconditional_params_ref" has to be manually specified ' + \
                'if "_param_shapes_meta" is not specified.'

            self._unconditional_params_ref = []
            for idx in self._unconditional_param_shapes_ref:
                meta = self.param_shapes_meta[idx]
                if meta['index'] != -1:
                    self._unconditional_params_ref.append(meta['index'])

            if len(self._unconditional_params_ref) == 0:
                self._unconditional_params_ref = None

        assert self._target_shapes is not None

    @property
    def unconditional_params(self):
        r"""Internally maintained parameters of the hypernetwork **excluding**
        parameters that may be specific to a given condition, e.g., task
        embeddings in continual learning.

        Hence, it is the portion of parameter tensors from attribute
        :attr:`mnets.mnet_interface.MainNetInterface.internal_params` that
        is not specific to a certain task/condition.

        Note:
            This attribute is ``None`` if there are no unconditional
            parameters that are internally maintained.

        Example:
            An example use-case for a hypernetwork :math:`h` could be the
            following: :math:`h(x, e_i; \theta)`, where :math:`x` is an
            arbitrary input, :math:`e_i` is a learned embedding (condition)
            and :math:`\theta` are the internal "unconditional" parameters
            of the hypernetwork. In some cases (for simplicity), the
            conditions :math:`e_i` as well as the parameters :math:`\theta`
            are maintained internally by this class. This attribute can be
            used to gain access to the "unconditional" parameters
            :math:`\theta`, while
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params`
            would return all "conditional" parameters :math:`e_i` as well
            as the "unconditional" parameters :math:`\theta`.

        :type: list or None
        """
        if self.unconditional_params_ref is None:
            return None
        ret = []
        for idx in self.unconditional_params_ref:
            ret.append(self.internal_params[idx])

        return ret

    @property
    def unconditional_params_ref(self):
        """A list of integers that has the same length as
        :attr:`unconditional_params`. Each entry represents an index within
        attribute :attr:`mnets.mnet_interface.MainNetInterface.internal_params`.

        If :attr:`unconditional_params` is ``None``, the this attribute is
        ``None`` as well.

        Example:
            Using an instance ``hnet`` that implements this interface, the
            following is ``True``.

            .. code-block:: python

                hnet.internal_params[hnet.unconditional_params_ref[i]] is \
                hnet.unconditional_params[i]

        Note:
            This attribute has different semantics compared to
            :attr:`unconditional_param_shapes_ref` which points to locations
            within
            :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`,
            wheras this attribute points to locations within
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params`.

        :type: list or None
        """
        return self._unconditional_params_ref

    @property
    def unconditional_param_shapes(self):
        """A list of lists of integers denoting the shape of every parameter
        tensor belonging to the `unconditional` parameters associated with this
        hypernetwork. Note, the returned list is a subset of the shapes
        maintained in :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`
        and is independent whether these parameters are internally maintained
        (i.e., occuring within :attr:`unconditional_params`).

        :type: list
        """
        ret = []
        for idx in self.unconditional_param_shapes_ref:
            ret.append(self.param_shapes[idx])
        return ret

    @property
    def unconditional_param_shapes_ref(self):
        """A list of integers that has the same length as
        :attr:`unconditional_param_shapes`. Each entry represents an index
        within attribute
        :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.

        :type: list
        """
        return self._unconditional_param_shapes_ref

    @property
    def num_outputs(self):
        """The total number of output neurons (number of weights generated for
        the target network). This quantity can be computed based on attribute
        :attr:`target_shapes`.

        :type: int
        """
        return MainNetInterface.shapes_to_num_weights(self.target_shapes)

    @property
    def target_shapes(self):
        """A list of list of integers representing the shapes of weight tensors
        generated, i.e., the hypernet output, which could be, for instance, the
        :attr:`mnets.mnet_interface.MainNetInterface.hyper_shapes_learned`
        of another network whose weights this hypernetwork is producing.

        :type: list
        """
        return self._target_shapes

    @property
    def conditional_params(self):
        """The complement of the internally maintained parameters hold by
        attribute :attr:`unconditional_params`.

        A typical example of these parameters are embedding vectors. In
        continual learning, for instance, there could be a separate task-
        embedding per task used as hypernet input, see

            von Oswald et al., "Continual learning with hypernetworks",
            ICLR 2020. https://arxiv.org/abs/1906.00695

        Note:
            This attribute is ``None`` if there are no conditional
            parameters that are internally maintained.

        :type: list or None
        """
        if self.internal_params is None:
            return None

        uc_indices = self.unconditional_params_ref
        if uc_indices is None:
            uc_indices = []

        ret = []
        for idx in range(len(self.internal_params)):
            if idx not in uc_indices:
                ret.append(self.internal_params[idx])

        return ret

    @property
    def conditional_param_shapes(self):
        """A list of lists of integers denoting the shape of every parameter
        tensor belonging to the `conditional` parameters associated with this
        hypernetwork (i.e., the complement of those returned by
        :attr:`unconditional_param_shapes`). Note, the returned list is a subset
        of the shapes maintained in
        :attr:`mnets.mnet_interface.MainNetInterface.param_shapes` and is
        independent whether these parameters are internally maintained
        (i.e., occuring within :attr:`conditional_params`).

        :type: list
        """
        ret = []
        for idx in self.conditional_param_shapes_ref:
            ret.append(self.param_shapes[idx])
        return ret

    @property
    def conditional_param_shapes_ref(self):
        """A list of integers that has the same length as
        :attr:`conditional_param_shapes`. Each entry represents an index within
        attribute :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.

        It can be used to gain access to meta information about conditional
        parameters via attribute
        :attr:`mnets.mnet_interface.MainNetInterface.param_shapes_meta`.

        :type: list
        """
        ret = []
        for i in range(len(self.param_shapes)):
            if i not in self.unconditional_param_shapes_ref:
                ret.append(i)
        return ret

    @property
    def num_known_conds(self):
        """The number of conditions known to this hypernetwork. If the number of
        conditions is discrete and internally maintained by the hypernetwork,
        then this attribute specifies how many conditions the hypernet manages.

        Note:
            The option does not have to agree with the length of attribute
            :attr:`conditional_params`. For instance, in certain cases there
            are multiple conditional weights maintained per condition.

        :type: int
        """
        return self._num_known_conds

    def get_task_embs(self):
        """Returns attribute :attr:`conditional_params`.

        .. deprecated:: 1.0
            Please access attribute :attr:`conditional_params` directly, as the
            conditional parameters do not have to correspond to task embeddings.

        Returns:
            (list or None)
        """
        # FIXME Method provided for legacy reasons (existed in deprecated
        # interface "CLHyperNetInterface").
        warn('Please use attribute "conditional_params" rather than this ' +
             'method.', DeprecationWarning)
        if len(self.conditional_params) != self.num_known_conds:
            raise RuntimeError('Do not know how to extract task embeddings ' +
                               'from this network.')
        return self.conditional_params

    def get_task_emb(self, task_id):
        """Returns the ``task_id``-th element from attribute
        :attr:`conditional_params`.

        .. deprecated:: 1.0
            Please access elements of attribute :attr:`conditional_params`
            directly, as the conditional parameters do not have to correspond to
            task embeddings.

        Args:
            task_id (int): Determines which element of
                :attr:`conditional_params` should be returned.

        Returns:
            (torch.nn.Parameter)
        """
        # FIXME Method provided for legacy reasons (existed in deprecated
        # interface "CLHyperNetInterface").
        warn('Please use attribute "conditional_params" rather than this ' +
             'method.', DeprecationWarning)

        if self.conditional_params is None:
            raise ValueError('No conditional parameters to be returned!')
        if len(self.conditional_params) != self.num_known_conds:
            raise RuntimeError('Do not know how to extract task embeddings ' +
                               'from this network.')
        return self.conditional_params[task_id]

    @abstractmethod
    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed'):
        """Perform a pass through the hypernetwork.

        Args:
            uncond_input (optional): The unconditional input to the
                hypernetwork.
                
                Note:
                    Not all scenarios require a hypernetwork with unconditional
                    inputs. For instance, a `task-conditioned hypernetwork \
<https://arxiv.org/abs/1906.00695>`__ only receives a task-embedding
                    (a conditional input) as input.
            cond_input (optional): If applicable, the conditional input to
                the hypernetwork.
            cond_id (int or list, optional): The ID of the condition to be
                applied. Only applicable if conditional inputs/weights are
                maintained internally and conditions are discrete.

                Can also be a list of IDs if a batch of weights should be
                produced.

                Condition IDs have to be between 0 and :attr:`num_conditions`.

                Note:
                    Option is mutually exclusive with option ``cond_input``.
            weights (list or dict, optional): List of weight tensors, that are
                used as hypernetwork parameters. If not all weights are
                internally maintained, then this argument is non-optional.

                If a ``list`` is provided, then it either has to match the
                length of :attr:`mnets.mnet_interface.MainNetInterface.\
hyper_shapes_learned` (if specified) or the length of attribute
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.

                If a ``dict`` is provided, it must have at least one of the
                following keys specified:
                - ``'uncond_weights'`` (list): Contains unconditional weights.
                - ``'cond_weights'`` (list): Contains conditional weights.
            distilled_params (optional): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`.
            condition (optional): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`.
            ret_format (str): The format in which the generated weights are
                returned. The following options are available.

                - ``'flattened'``: The hypernet output will be a tensor of shape
                  ``[batch_size, num_outputs]`` (see :attr:`num_outputs`).
                - ``'sequential'``: A list of length `batch size` is returned
                  that contains lists of length ``len(target_shapes)``, which
                  contain tensors with shapes determined by attribute
                  :attr:`target_shapes`. Hence, each entry of the returned list
                  contains the weights for one sample in the input batch.
                - ``'squeezed'``: Same as ``'sequential'``, but if the batch
                  size is ``1``, the list will be unpacked, such that a list of
                  tensors is returned (rather than a list of list of tensors).

                Example:
                    Assume :attr:`target_shapes` to be ``[[10, 5], [10]]`` and
                    ``cond_input`` to be the only input to the hypernetwork,
                    which is a batch of embeddings ``[B, E]``, where ``B`` is
                    the batch size and ``E`` is the embedding size.

                    Note, ``num_outputs = 60`` in this case
                    (cmp. :attr:`num_outputs`).

                    If ``'flattened'`` is used, a tensor of shape ``[B, 60]`` is
                    returned. If ``'sequential'`` or ``'squeezed'`` is used and
                    ``B > 1`` (e.g., ``B=3``), then a list of lists of tensors
                    (here denoted by their shapes) is returned
                    ``[[[10, 5], [10]], [[10, 5], [10]], [[10, 5], [10]]]``.
                    However, if ``B == 1`` and ``'squeezed'`` is used, then a
                    list of tensors is returned, e.g., ``[[10, 5], [10]]``.

        Returns:
            (list or torch.Tensor): See description of argument ``ret_format``.
        """
        # The method `_preprocess_forward_args` might be helpful if an
        # implementing subclass 
        raise NotImplementedError('TODO implement function')

        # You should first preprocess the kwargs. Either as follows:
        #uncond_input, cond_input, uncond_weights, cond_weights = \
        #    self._preprocess_forward_args(uncond_input=uncond_input,
        #        cond_input=cond_input, cond_id=cond_id, weights=weights,
        #        distilled_params=distilled_params, condition=condition,
        #        squeeze=squeeze)
        # Or like this.
        #kwarg_names = inspect.signature(HyperNetInterface.forward).\
        #    parameters.keys()
        #kwarg_vals = dict(locals())
        #uncond_input, cond_input, uncond_weights, cond_weights = \
        #    self._preprocess_forward_args(\
        #        {k: kwarg_vals[k] for k in kwarg_names})

    def add_to_uncond_params(self, dparams, params=None):
        r"""Add perturbations to unconditional parameters.

        This method simply adds a perturbation ``dparams`` (:math:`d\theta`) to
        the unconditional parameters :math:`\theta`.

        Args:
            dparams (list): List of tensors.
            params (list, optional): List of tensors. If unspecified, attribute
                :attr:`unconditional_params` is taken instead. Otherwise, the
                method simply returns ``params + dparams``.

        Returns:
            (list): List were elements of ``dparams`` and unconditional params
            (or ``params``) are summed together.
        """
        if params is None:
            if self.unconditional_params is None:
                raise ValueError('Method requires option "params" if there ' +
                                 'are no internally maintained unconditional ' +
                                 'parameters.')
            params = self.unconditional_params

        if len(params) != len(dparams):
            raise ValueError('Lengths of lists to be added must match!')
        return [p + dp for p, dp in zip(params, dparams)]

    def _preprocess_forward_args(self, _input_required=True,
                                 _parse_cond_id_fct=None, **kwargs):
        """Parse all :meth:`forward` arguments.

        Note:
            This method is currently not considering the arguments
            ``distilled_params`` and ``condition``.

        Args:
            _input_required (bool): Whether at least one of the forward
                arguments ``uncond_input``, ``cond_input`` and ``cond_id`` has
                to be not ``None``.
            _parse_cond_id_fct (func): A function with signature
                ``_parse_cond_id_fct(self, cond_ids, cond_weights)``, where
                ``self`` is the current object, ``cond_ids`` is a ``list`` of
                integers and ``cond_weights`` are the parsed conditional weights
                if any (see return values).
                The function is expected to parse argument ``cond_id`` of the
                :meth:`forward` method. If not provided, we simply use the
                indices within ``cond_id`` to stack elements of
                :attr:`conditional_params`.
            **kwargs: All keyword arguments passed to the :meth:`forward`
                method.

        Returns:
            (tuple): Tuple containing:

            - **uncond_input**: The argument ``uncond_input`` passed to the
              :meth:`forward` method.
            - **cond_input**: If provided, then this is just argument
              ``cond_input`` of the :meth:`forward` method. Otherwise, it is
              either ``None`` or if provided, the conditional input will be
              assembled from the parsed conditional weights ``cond_weights``
              using :meth:`forward` argument ``cond_id``.
            - **uncond_weights**: The unconditional weights :math:`\\theta` to
              be used during forward processing (they will be assembled from
              internal and given weights).
            - **cond_weights**: The conditional weights if tracked be the
              hypernetwork. The parsing is done analoguously as for
              ``uncond_weights``.
        """
        if kwargs['ret_format'] not in ['flattened', 'sequential', 'squeezed']:
            raise ValueError('Return format %s unknown.' \
                             % (kwargs['ret_format']))

        #####################
        ### Parse Weights ###
        #####################
        # We first parse the weights as they night be needed later to choose
        # inputs via `cond_id`.
        uncond_weights = self.unconditional_params
        cond_weights = self.conditional_params
        if kwargs['weights'] is not None:
            if isinstance(kwargs['weights'], dict):
                assert 'uncond_weights' in kwargs['weights'].keys() or \
                       'cond_weights' in kwargs['weights'].keys()

                if 'uncond_weights' in kwargs['weights'].keys():
                    # For simplicity, we assume all unconditional parameters
                    # are passed. This might have to be adapted in the
                    # future.
                    assert len(kwargs['weights']['uncond_weights']) == \
                           len(self.unconditional_param_shapes)
                    uncond_weights = kwargs['weights']['uncond_weights']
                if 'cond_weights' in kwargs['weights'].keys():
                    # Again, for simplicity, we assume all conditional weights
                    # have to be passed.
                    assert len(kwargs['weights']['cond_weights']) == \
                           len(self.conditional_param_shapes)
                    cond_weights = kwargs['weights']['cond_weights']

            else: # list
                if self.hyper_shapes_learned is not None and \
                        len(kwargs['weights']) == \
                        len(self.hyper_shapes_learned):
                    # In this case, we build up conditional and
                    # unconditional weights from internal and given weights.
                    weights = []
                    for i in range(len(self.param_shapes)):
                        if i in self.hyper_shapes_learned_ref:
                            idx = self.hyper_shapes_learned_ref.index(i)
                            weights.append(kwargs['weights'][idx])
                        else:
                            meta = self.param_shapes_meta[i]
                            assert meta['index'] != -1
                            weights.append( \
                                self.internal_params[meta['index']])
                else:
                    if len(kwargs['weights']) != len(self.param_shapes):
                        raise ValueError('The length of argument ' +
                            '"weights" does not meet the specifications.')
                    # In this case, we simply split the given weights into
                    # conditional and unconditional weights.
                    weights = kwargs['weights']
                assert len(weights) == len(self.param_shapes)

                # Split 'weights' into conditional and unconditional weights.
                up_ref = self.unconditional_param_shapes_ref
                cp_ref = self.conditional_param_shapes_ref

                if up_ref is not None:
                    uncond_weights = [None] * len(up_ref)
                else:
                    up_ref = []
                    uncond_weights = None
                if cp_ref is not None:
                    cond_weights = [None] * len(cp_ref)
                else:
                    cp_ref = []
                    cond_weights = None

                for i in range(len(self.param_shapes)):
                    if i in up_ref:
                        idx = up_ref.index(i)
                        assert uncond_weights[idx] is None
                        uncond_weights[idx] = weights[i]
                    else:
                        assert i in cp_ref
                        idx = cp_ref.index(i)
                        assert cond_weights[idx] is None
                        cond_weights[idx] = weights[i]

        ####################
        ### Parse Inputs ###
        ####################
        if _input_required and kwargs['uncond_input'] is None and \
                kwargs['cond_input'] is None and kwargs['cond_id'] is None:
            raise RuntimeError('No hypernet inputs have been provided!')

        # No further preprocessing required.
        uncond_input = kwargs['uncond_input']

        if kwargs['cond_input'] is not None and kwargs['cond_id'] is not None:
            raise ValueError('You cannot provide arguments "cond_input" and ' +
                             '"cond_id" simultaneously!')

        cond_input = None
        if kwargs['cond_input'] is not None:
            cond_input = kwargs['cond_input']
            if len(cond_input.shape) == 1:
                raise ValueError('Batch dimension for conditional inputs is ' +
                                 'missing.')
        if kwargs['cond_id'] is not None:
            assert isinstance(kwargs['cond_id'], (int, list))
            cond_ids = kwargs['cond_id']
            if isinstance(cond_ids, int):
                cond_ids = [cond_ids]

            if _parse_cond_id_fct is not None:
                cond_input = _parse_cond_id_fct(self, cond_ids, cond_weights)
            else:
                if cond_weights is None:
                    raise ValueError('Forward option "cond_id" can only be ' +
                                     'used if conditional parameters are ' 
                                     'maintained internally or passed to the ' +
                                     'forward method via option "weights".')

                assert len(cond_weights) == len(self.conditional_param_shapes)
                if len(cond_weights) != self.num_known_conds:
                    raise RuntimeError('Do not know how to translate IDs to ' +
                                       'conditional inputs.')

                cond_input = []
                for i, cid in enumerate(cond_ids):
                    if cid < 0 or cid >= self.num_known_conds:
                        raise ValueError('Condition %d not existing!' % (cid))

                    cond_input.append(cond_weights[cid])
                    if i > 0:
                        # Assumption when not providing `_parse_cond_id_fct`.
                        assert np.all(np.equal(cond_input[0].shape,
                                               cond_input[i].shape))

                cond_input = torch.stack(cond_input, dim=0)

        # If we are given both, unconditional and conditional inputs, we
        # have to ensure that they use the same batch size.
        if cond_input is not None and uncond_input is not None:
            # We assume the first dimension being the batch dimension.
            # Note, some old hnet implementations could only process one
            # embedding at a time and it was ok to not have a dedicated
            # batch dimension. To avoid nasty bugs we enforce a separate
            # batch dimension.
            assert len(cond_input.shape) > 1 and len(uncond_input.shape) > 1
            if cond_input.shape[0] != uncond_input.shape[0]:
                # If one batch-size is 1, we just repeat the input.
                if cond_input.shape[0] == 1:
                    batch_size = uncond_input.shape[0]
                    cond_input = cond_input.expand(batch_size,
                                                   *cond_input.shape[1:])
                elif uncond_input.shape[0] == 1:
                    batch_size = cond_input.shape[0]
                    uncond_input = uncond_input.expand(batch_size,
                        *uncond_input.shape[1:])
                else:
                    raise RuntimeError('Batch dimensions of hypernet ' +
                                       'inputs do not match!')
            assert cond_input.shape[0] == uncond_input.shape[0]

        return uncond_input, cond_input, uncond_weights, cond_weights

    def convert_out_format(self, hnet_out, src_format, trgt_format):
        """Convert the hypernetwork output into another format.

        This is a helper method to easily convert the output of a hypernetwork
        into different formats. Cf. argument ``ret_format`` of method
        :meth:`forward`.

        Args:
            hnet_out (list or torch.Tensor): See return value of method
                :meth:`forward`.
            src_format (str): The format of argument ``hnet_out``. See argument
                ``ret_format`` of method :meth:`forward`.
            trgt_format (str): The target format in which ``hnet_out`` should be
                converted. See argument ``ret_format`` of method
                :meth:`forward`.

        Returns:
            (list or torch.Tensor): The input ``hnet_out`` converted into the
                target format ``trgt_format``.
        """
        assert src_format in ['flattened', 'sequential', 'squeezed']
        assert trgt_format in ['flattened', 'sequential', 'squeezed']

        if src_format == trgt_format:
            return hnet_out

        elif src_format == 'flattened':
            self._flat_to_ret_format(hnet_out, trgt_format)

        else:
            assert trgt_format == 'flattened'
            if src_format == 'squeezed':
                if len(hnet_out) > 0 and isinstance(hnet_out[0], torch.Tensor):
                    hnet_out = [hnet_out]

            ret = []
            for w in hnet_out:
                ret.append(torch.cat([p.flatten() for p in w]))
            return torch.stack(ret)

    def _flat_to_ret_format(self, flat_out, ret_format):
        """Helper function to convert flat hypernet output into desired output
        format.

        Args:
            flat_out (torch.Tensor): The flat output tensor corresponding to
                ``ret_format='flattened'``.
            ret_format (str): The target output format. See docstring of method
                :meth:`forward`.

        Returns:
            (list or torch.)
        """
        assert ret_format in ['flattened', 'sequential', 'squeezed']
        assert len(flat_out.shape) == 2
        batch_size = flat_out.shape[0]

        if ret_format == 'flattened':
            return flat_out

        ret = [[] for _ in range(batch_size)]
        ind = 0
        for s in self.target_shapes:
            num = int(np.prod(s))

            W = flat_out[:, ind:ind+num]
            W = W.view(batch_size, *s)

            for bind, W_b in enumerate(torch.split(W, 1, dim=0)):
                W_b = torch.squeeze(W_b, dim=0)
                assert np.all(np.equal(W_b.shape, s))
                ret[bind].append(W_b)

            ind += num

        if ret_format == 'squeezed' and batch_size == 1:
            return ret[0]

        return ret

    def __str__(self):
        """Print network information."""
        num_uncond = MainNetInterface.shapes_to_num_weights( \
            self.unconditional_param_shapes)
        num_cond = MainNetInterface.shapes_to_num_weights( \
            self.conditional_param_shapes)
        num_uncond_internal = 0
        num_cond_internal = 0
        if self.unconditional_params is not None:
            num_uncond_internal = MainNetInterface.shapes_to_num_weights( \
                [p.shape for p in self.unconditional_params])
        if self.unconditional_params is not None:
            num_cond_internal = MainNetInterface.shapes_to_num_weights( \
                [p.shape for p in self.conditional_params])
        msg = 'Hypernetwork with %d weights and %d outputs (compression ' + \
            'ratio: %.2f).\nThe network consists of %d unconditional ' + \
            'weights (%d internally maintained) and %d conditional '+ \
            'weights (%d internally maintained).'
        return msg % (self.num_params, self.num_outputs,
            self.num_params/self.num_outputs, num_uncond, num_uncond_internal,
            num_cond, num_cond_internal)

if __name__ == '__main__':
    pass
