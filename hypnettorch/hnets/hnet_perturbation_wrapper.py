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
# @title          :hnets/hnet_perturbation_wrapper.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :12/17/2020
# @version        :1.0
# @python_version :3.6.10
"""
Hypernetwork-wrapper for input-preprocessing and output-postprocessing
----------------------------------------------------------------------

The module :mod:`hnets.hnet_perturbation_wrapper` implements a wrapper for
hypernetworks that implement the interface
:class:`hnets.hnet_interface.HyperNetInterface`. By default, the wrapper is
meant for perturbing hypernetwork outputs, such that an implicit distribution
(realized via a hypernetwork) with low-dimensional support can be inflated to
have support in the full weight space.

However, the wrapper allows in general to pass function handles that preprocess
inputs and/or postprocess hypernetwork outputs.
"""
import torch.nn as nn

from hypnettorch.hnets.hnet_interface import HyperNetInterface

class HPerturbWrapper(nn.Module, HyperNetInterface):
    r"""Hypernetwork wrapper for output perturbation.

    This wrapper is meant as a helper for hypernetworks that represent
    implicit distributions, i.e., distributions that transform a simple base
    distribution :math:`p_Z(z)` into a complex target distributions

    .. math::

        w \sim q_{\theta}(W) \Leftrightarrow w = h_{\theta}(z) \quad
        \text{,} \quad z \sim p_Z(Z)

    However, the wrapper is more versatile and can also become handy in a
    variety of other use cases. Yet, in the following we concentrate on
    implicit distributions and their practical challenges. One main challenge
    is typically that the density :math:`q_\theta(W)` is only defined on a
    lower-dimensional manifold of the weight space. This is often an undesirable
    property (e.g., such implicit distributions are often not amenable for
    optimization with standard divergence measures, such as the KL).

    A simple way to overcome this issue is to add noise perturbations to the
    output of the hypernetwork, such that the perturbations itself origin from
    a full-support distribution. By default, this hypernetwork wrapper adjusts
    the sampling procedure above in the following way

    .. math::
        :label: eqdefaultsampling

        w \sim \tilde{q}_{\theta}(W) \Leftrightarrow w = h_{\theta}(z_{:n}) +
        \sigma_{\text{noise}}^2 z \equiv \tilde{h}_{\theta}(z) \quad \text{,}
        \quad z \sim p_Z(Z)

    where now :math:`\dim(\mathcal{W}) = \dim(\mathcal{Z})`,
    :math:`\sigma_\text{noise}` is a hyperparameter that controls the
    perturbation strength, and :math:`z_{:n}` are the :math:`n` first entries
    of the vector :math:`z`.

    By default, the unconditional input size of this hypernetwork will be
    of size ``hnet.num_outputs`` (if ``input_handler`` is not provided) and the
    output size will be of the same size.

    Args:
        hnet (hnets.hnet_interface.HyperNetInterface): The hypernetwork around
            which this wrapper should be wrapped.
        hnet_uncond_in_size (int): This argument refers to :math:`n` from Eq.
            :eq:`eqdefaultsampling`. If ``input_handler`` is provided, this
            argument will be ignored.
        sigma_noise (float): The perturbation strength
            :math:`\sigma_\text{noise}` from Eq. :eq:`eqdefaultsampling`. If
            ``output_handler`` is provided, this argument will be ignored.
        input_handler (func, optional): A function handler to process the
            inputs to the :meth:`hnets.hnet_interface.HyperNetInterface.forward`
            method of ``hnet``. The function handler should have the following
            signature

            .. code-block:: python

                uncond_input_int, cond_input_int, cond_id_int = input_handler( \
                    uncond_input=None, cond_input=None, cond_id=None)

            The returned values will be passed to :attr:`internal_hnet`.

            Example:
                For instance, to reproduce the behavior depicted in Eq.
                :eq:`eqdefaultsampling` one could provide the following handler

                .. code-block:: python

                    def input_handler(uncond_input=None, cond_input=None,
                                      cond_id=None):
                        assert uncond_input is not None
                        n = 5
                        return uncond_input[:, :n], cond_input, cond_id
        output_handler (func, optional): A function handler to postprocess the
            outputs of the internal hypernetwork :attr:`internal_hnet`.

            A function handler with the following signature is expected.

            .. code-block:: python

                hnet_out = output_handler(hnet_out_int, uncond_input=None,
                                          cond_input=None, cond_id=None)

            where ``hnet_out_int`` is the output of the internal hypernetwork
            :attr:`internal_hnet` and the remaining arguments are the original
            arguments passed to method :meth:`forward`. ``hnet_out_int`` will
            always have the format ``ret_format='flattened'`` and is also
            expected to return this format.

            Example:
                Deviating from Eq. :eq:`eqdefaultsampling`, let's say we want
                to implement the following sampling behavior
                
                .. math::

                    w \sim \hat{q}_\theta(W) \Leftrightarrow
                    w = h_\theta(z) + \epsilon_w \quad \text{,} \quad 
                    z \sim p_Z(Z) \text{ and } \epsilon_w \sim p_\text{noise}(W)

                In this case the unconditional input ``uncond_input`` to the
                :meth:`forward` method is expected to have size
                :math:`\dim(\mathcal{Z}) + \dim(\mathcal{W})`.

                .. code-block:: python

                    def input_handler(uncond_input=None, cond_input=None,
                                      cond_id=None):
                        assert uncond_input is not None
                        return uncond_input[:, :dim_z], cond_input, cond_id

                .. code-block:: python

                    def output_handler(hnet_out_int, uncond_input=None,
                                       cond_input=None, cond_id=None):
                        assert uncond_input is not None
                        return hnet_out_int + uncond_input[:, dim_z:]
        verbose (bool): Whether network information should be printed during
            network creation.
    """
    def __init__(self, hnet, hnet_uncond_in_size=None, sigma_noise=0.02,
                 input_handler=None, output_handler=None, verbose=True):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        HyperNetInterface.__init__(self)

        assert isinstance(hnet, HyperNetInterface)
        self._hnet = hnet
        self._hnet_uncond_in_size = hnet_uncond_in_size
        self._sigma_noise = sigma_noise
        self._input_handler = input_handler
        self._output_handler = output_handler

        if input_handler is None and hnet_uncond_in_size is None:
            raise ValueError('Either "input_handler" or "hnet_uncond_in_size"' +
                             ' has to be specified.')

        ### Setup attributes required by interface ###
        # Most of these attributes are taken over from `self._hnet`
        self._target_shapes = hnet.target_shapes
        self._num_known_conds = self._hnet.num_known_conds
        self._unconditional_param_shapes_ref = \
            list(self._hnet.unconditional_param_shapes_ref)

        if self._hnet.internal_params is not None:
            self._internal_params = \
                nn.ParameterList(self._hnet.internal_params)
        self._param_shapes = list(self._hnet.param_shapes)
        self._param_shapes_meta = list(self._hnet.param_shapes_meta)
        if self._hnet.hyper_shapes_learned is not None:
            self._hyper_shapes_learned = list(self._hnet.hyper_shapes_learned)
            self._hyper_shapes_learned_ref = \
                list(self._hnet.hyper_shapes_learned_ref)
        if self._hnet.hyper_shapes_distilled is not None:
            self._hyper_shapes_distilled = \
                list(self._hnet.hyper_shapes_distilled)
        self._has_bias = self._hnet.has_bias
        # A noise perturbed output can't be considered an FC output anymore.
        self._has_fc_out = False
        self._mask_fc_out = self._hnet.mask_fc_out
        # Guess that's the safest answer.
        self._has_linear_out = False
        self._layer_weight_tensors = \
            nn.ParameterList(self._hnet.layer_weight_tensors)
        self._layer_bias_vectors = \
            nn.ParameterList(self._hnet.layer_bias_vectors)
        if self._hnet.batchnorm_layers is not None:
            self._batchnorm_layers = nn.ModuleList(self._hnet.batchnorm_layers)
        if self._hnet.context_mod_layers is not None:
            self._context_mod_layers = \
                nn.ModuleList(self._hnet.context_mod_layers)

        ### Finalize construction ###
        self._is_properly_setup()

        if verbose:
            print('Wrapped a perturbation interface around a hypernetwork.')
            #print(self)


    @property
    def internal_hnet(self):
        """The underlying hypernetwork that was passed via constructor argument
        ``hnet``.

        :type: hnets.hnet_interface.HyperNetInterface
        """
        return self._hnet

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed'):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.hnet_interface.HyperNetInterface.forward`.

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """
        if self._input_handler is not None:
            uncond_input_int, cond_input_int, cond_id_int = \
                self._input_handler(uncond_input=uncond_input,
                                    cond_input=cond_input, cond_id=cond_id)
        else:
            assert uncond_input is not None
            uncond_input_int = uncond_input[:, :self._hnet_uncond_in_size]
            cond_input_int = cond_input
            cond_id_int = cond_id

        hnet_out_int = self._hnet.forward(uncond_input=uncond_input_int,
            cond_input=cond_input_int, cond_id=cond_id_int, weights=weights,
            distilled_params=distilled_params, condition=condition,
            ret_format='flattened')

        if self._output_handler is not None:
            hnet_out = self._output_handler(hnet_out_int,
                uncond_input=uncond_input, cond_input=cond_input,
                cond_id=cond_id)
        else:
            assert hnet_out_int.shape == uncond_input.shape
            hnet_out = hnet_out_int + self._sigma_noise * uncond_input

        ### Split output into target shapes ###
        hnet_out = self._flat_to_ret_format(hnet_out, ret_format)

        return hnet_out

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        Returns:
            Simply returns the ``distillation_targets`` of the internal hypernet
            :attr:`internal_hnet``.
        """
        return self.internal_hnet.distillation_targets()

if __name__ == '__main__':
    pass


