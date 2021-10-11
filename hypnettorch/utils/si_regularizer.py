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
# @title          :utils/si_regularizer.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :02/14/2020
# @version        :1.0
# @python_version :3.6.9
r"""
Synaptic Intelligence
---------------------

The module :mod:`utils.si_regularizer` implements the Synaptic Intelligence (SI)
regularizer proposed in

    Zenke et al., "Continual Learning Through Synaptic Intelligence", 2017.
    https://arxiv.org/abs/1703.04200

Note:
    We aim to follow the suggested implementation from appendix section A.2.3 in

        van de Ven et al., "Three scenarios for continual learning", 2019.
        https://arxiv.org/pdf/1904.07734.pdf

    We additionally ensure that importance weights :math:`\Omega` are positive.

Note:
    This implementation has the following memory requirements. Let :math:`n`
    denote the number of parameters to be regularized.

    We always need to store the importance weights :math:`\Omega` and the
    checkpointed weights after learning the last task
    :math:`\theta_\text{prev}`.

    We also need to checkpoint the weights right before the optimizer step is
    performed :math:`\theta_\text{pre\_step}` in order to update the running
    importance estimate :math:`\omega`.

    Hence, we keep an additional memory of :math:`4n`.

.. autosummary::

    hypnettorch.utils.si_regularizer.si_pre_optim_step
    hypnettorch.utils.si_regularizer.si_post_optim_step
    hypnettorch.utils.si_regularizer.si_compute_importance
    hypnettorch.utils.si_regularizer.si_regularizer
"""

import torch

def si_pre_optim_step(net, params, params_name=None, no_pre_step_ckpt=False):
    r"""Prepare SI importance estimate before running the optimizer step.

    This function has to be called before running the optimizer step in order
    to checkpoint :math:`\theta_\text{pre\_step}`.

    Note:
        When this function is called the first time (for the first task), the
        given parameters will also be checkpointed as the initial weights,
        which are required to normalize importances :math:\Omega` after
        training.

    Args:
        net (torch.nn.Module): A network required to store buffers (i.e., the
            running variables that SI needs to keep track of).
        params (list): A list of parameter tensors. For each parameter tensor
            in this list that ``requires_grad`` the importances will be
            measured.
        params_name (str, optional): In case SI should be performed for
            multiple parameter groups ``params``, one has to assign names to
            each group via this option.
        no_pre_step_ckpt (bool): If ``True``, then this function will not
            checkpoint :math:`\theta_\text{pre\_step}`. Instead, option
            ``delta_params`` of function :func:`si_post_optim_step` is expected
            to be set.

            Note:
                One still has to call this function once before updating the
                parameters of the first task for the first time.
    """
    for i, p in enumerate(params):
        _, prev_theta_name, _, pre_step_theta_name = _si_buffer_names(i,
            params_name=params_name)

        if p.requires_grad:
            if not hasattr(net, prev_theta_name):
                # Note, this condition should only be True when calling this
                # function for the very first time. It is required to later
                # normalize Omega.
                net.register_buffer(prev_theta_name, p.detach().clone())

            if not no_pre_step_ckpt:
                net.register_buffer(pre_step_theta_name, p.detach().clone())

def si_post_optim_step(net, params, params_name=None, delta_params=None):
    r"""Update running importance estimate :math:`\omega`.

    This function is called after an optimizer update step has been performed.
    It will perform an update of the internal running variable :math:\omega`
    using the current parameter values, the checkpointed parameter values
    before the optimizer step (:math:`\theta_\text{pre\_step}`, see function
    :func:`si_pre_optim_step`) and the negative gradients accumulated in the
    ``grad`` variables of the parameters.

    Args:
        (....): See docstring of function :func:`si_pre_optim_step`.
        delta_params (list): One may pass the parameter update step directly.
            In this case. the difference between the current parameter values
            and the previous ones :math:`\theta_\text{pre\_step}` will not be
            computed.

            Note:
                One may use the functions provided in module
                :mod:`utils.optim_step` to calculate ``delta_params``

            Note:
                When this option is used, it is not required to explicitly call
                the optimizer its ``step`` function. Though, it is still
                required that gradients are computed and accumulated in the
                ``grad`` variables of the parameters in ``params``.

            Note:
                This option is particularly interesting if importances should
                only be estimated wrt to a part of the total loss function,
                e.g., the task-specific part, ignoring other parts of the loss
                (e.g., regularizers).
    """
    for i, p in enumerate(params):
        _, _, running_omega_name, pre_step_theta_name = _si_buffer_names(i,
            params_name=params_name)

        if p.requires_grad:
            if p.grad is None:
                raise ValueError('Function "si_post_optim_step" expects that ' +
                                 'gradients wrt the loss have been computed.')
            if not hasattr(net, running_omega_name) or \
                    getattr(net, running_omega_name) is None:
                omega = torch.zeros_like(p).to(p.device)
            else:
                omega = getattr(net, running_omega_name)

            if delta_params is None:
                if not hasattr(net, pre_step_theta_name) or \
                        getattr(net, pre_step_theta_name) is None:
                    raise ValueError('Function "si_post_optim_step" requires ' +
                                     'that function "si_pre_optim_step" has ' +
                                     'been called or "delta_params" was set.')
                delta_p = (p.detach() - getattr(net, pre_step_theta_name))
                # Allows us to detect inconsistent use of functions and to
                # reduce memory footprint during testing.
                setattr(net, pre_step_theta_name, None)
            else:
                delta_p = delta_params[i]

            omega += delta_p * (-p.grad)
            net.register_buffer(running_omega_name, omega)

def si_compute_importance(net, params, params_name=None, epsilon=1e-3):
    r"""Compute weight importance :math:`\Omega` after training a task.

    Note:
        This function is assumed to be called after the training on the current
        task finished. It will set the variable :math:`\theta_\text{prev}` to
        the current parameter value.

    Args:
        (....): See docstring of function :func:`si_pre_optim_step`.
        epsilon (float): Damping parameter used to ensure numerical stability
            when normalizing weight importance.
    """
    for i, p in enumerate(params):
        if not p.requires_grad:
            continue

        omega_name, prev_theta_name, running_omega_name, _ = _si_buffer_names(i,
            params_name=params_name)

        if not hasattr(net, prev_theta_name):
            raise ValueError('SI importance weights can only be computed if ' +
                             'function "si_pre_optim_step" has been called ' +
                             'at the beginning of training the first task.')
        if not hasattr(net, running_omega_name):
            raise ValueError('SI importance weights can only be computed if ' +
                             'function "si_post_optim_step" has been ' +
                             'correctly used during training.')

        prev_theta = getattr(net, prev_theta_name)
        running_omega = getattr(net, running_omega_name)

        if not hasattr(net, omega_name):
            omega = torch.zeros_like(p).to(p.device)
        else:
            omega = getattr(net, omega_name)

        total_change = p.detach() - prev_theta
        omega_current = running_omega / (total_change**2 + epsilon)

        # Ensure, that we only add positive importance weights (otherwise, we
        # would drive weights away from the previous solution).
        omega += torch.clamp(omega_current, min=0)

        net.register_buffer(omega_name, omega)

        # Update theta_prev which is important next time this function is
        # called.
        net.register_buffer(prev_theta_name, p.detach().clone())

        # Important, we have to reset the running importance estimate before
        # starting training on the next task.
        setattr(net, running_omega_name, None)

def si_regularizer(net, params, params_name=None):
    """Apply synaptic intelligence regularizer.

    This function computes the SI regularizer. Note, a regularization strength
    should be multiplied by the returned loss post-hoc, to tune the strength.

    Args:
        (....): See docstring of function :func:`si_pre_optim_step`.

    Returns:
        (torch.Tensor): The regularizer as scalar value.
    """
    reg = 0.

    for i, p in enumerate(params):
        if not p.requires_grad:
            continue

        omega_name, prev_theta_name, _, _ = _si_buffer_names(i,
            params_name=params_name)

        if not hasattr(net, omega_name) or not hasattr(net, prev_theta_name):
            raise ValueError('Function "si_regularizer" can only be used ' +
                             'after function "si_compute_importance" has ' +
                             'been called at least once.')

        prev_theta = getattr(net, prev_theta_name)
        omega = getattr(net, omega_name)

        reg += (omega * (p - prev_theta)**2).sum()

    return reg

def _si_buffer_names(param_id, params_name=None):
    r"""The names of the buffers used to store SI variables.

    Args:
        param_id (int): Identifier of parameter tensor.
        params_name (str, optional): Name of the parameter group.

    Returns:
        (tuple): Tuple containing:

        - **omega_name**: Buffer name of :math:`\Omega`.
        - **prev_theta_name**: Buffer name of :math:`\theta_\text{prev}`.
        - **running_omega_name**: Buffer name of :math:\omega`.
        - **pre_step_theta_name**: Buffer name of
          :math:`\theta_\text{pre\_step}`.
    """
    pname = '' if params_name is None else '_%s' % params_name

    omega_name = 'si_omega{}_weights_{}'.format(pname, param_id)
    prev_theta_name = 'si_prev_theta{}_weights_{}'.format(pname, param_id)
    running_omega_name = 'si_running_omega{}_weights_{}'.format(pname, param_id)
    pre_step_theta_name = 'si_pre_step_theta{}_weights_{}'.format(pname,
                                                                  param_id)

    return omega_name, prev_theta_name, running_omega_name, pre_step_theta_name

if __name__ == '__main__':
    pass


