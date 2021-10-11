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
# @title           :utils/ewc_regularizer.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :05/07/2019
# @version         :1.0
# @python_version  :3.6.8
"""
Elastic Weight Consolidation
----------------------------

Implementation of EWC:
    https://arxiv.org/abs/1612.00796

Note, these implementation are based on the descriptions provided in:
    https://arxiv.org/abs/1809.10635

The code is inspired by the corresponding implementation:
    https://git.io/fjcnL
"""
import torch
from torch.nn import functional as F
from warnings import warn

from hypnettorch.mnets.mnet_interface import MainNetInterface

def compute_fisher(task_id, data, params, device, mnet, hnet=None,
                   empirical_fisher=True, online=False, gamma=1., n_max=-1,
                   regression=False, time_series=False,
                   allowed_outputs=None, custom_forward=None, custom_nll=None,
                   pass_ids=False, proper_scaling=False, prior_strength=None,
                   regression_lvar=1., target_manipulator=None):
    r"""Compute estimates of the diagonal elements of the Fisher information
    matrix, as needed as importance-weights by elastic weight consolidation
    (EWC).

    The Fisher matrix for a conditional distribution :math:`p(y \mid \theta, x)`
    (i.e., the model likelihood for a model with parameters :math:`\theta`) is
    defined as follows at location :math:`x`

    .. math::

        \mathcal{F}(x) &= \textrm{Var} \big[ \nabla_{\theta}
        \log p(y \mid \theta, x) \big] \\
        &= \mathbb{E}_{p(y \mid \theta, x)} \big[
        \nabla_{\theta}\log p(y \mid \theta, x)
        \nabla_{\theta}\log p(y \mid \theta, x)^T\big]

    In practice, we are often interested in the Fisher averaged over locations

    .. math::

        \mathcal{F} = \mathbb{E}_{p(x)} [ \mathcal{F}(x) ]

    Since the model is trained, such that in-distribution the model likelihood
    :math:`p(y \mid \theta, x)` and the ground-truth likelihood
    :math:`p(y \mid x)` agree, people often refer to the empirical Fisher, which
    utilizes the dataset for computation and therewith doesn't require sampling
    from the model likelihood. Note, EWC anyway assumes that in-distribution
    :math:`p(y \mid \theta, x) = p(y \mid x)` in order to be able to replace
    the Hessian by the Fisher matrix.

    .. math::

        \mathcal{F}_{emp} &= \mathbb{E}_{p(x,y)}
        \big[ \nabla_{\theta}\log p(y \mid \theta, x)
        \nabla_{\theta}\log p(y \mid \theta, x)^T\big] \\
        &= \mathbb{E}_{p(x)} \Big[  \mathbb{E}_{p(y \mid x)}
        \big[ \nabla_{\theta}\log p(y \mid \theta, x)
        \nabla_{\theta}\log p(y \mid \theta, x)^T\big]  \Big] \\
        &\approx \frac{1}{|\mathcal{D}|} \sum_{(x_n, y_n) \sim \mathcal{D}}
        \big[ \nabla_{\theta}\log p(y_n \mid \theta, x_n)
        \nabla_{\theta}\log p(y_n \mid \theta, x_n)^T\big]  \Big]

    Note:
        This method registers buffers in the given module (storing the
        current parameters and the estimate of the Fisher diagonal elements),
        i.e., the ``mnet`` if ``hnet`` is ``None``, otherwise the ``hnet``.

    Args:
        task_id: The ID of the current task, needed to store the computed
            tensors with a unique name. When ``hnet`` is given, it is used as
            input to the ``hnet`` forward method to select the current task
            embedding.
        data: A data handler. We will compute the Fisher estimate across the
            whole training set (except ``n_max`` is specified).
        params: A list of parameter tensors from the module of which we aim to
            compute the Fisher for. If ``hnet`` is given, then these are assumed
            to be the "theta" parameters, that we pass to the forward function
            of the hypernetwork. Otherwise, these are the "weights" passed to
            the forward method of the main network.
            Note, they might not be detached from their original parameters,
            because we use ``backward()`` on the computational graph to read out
            the ``.grad`` variable.
            Note, the order in which these parameters are passed to this method
            and the corresponding EWC loss function must not change, because
            the index within the "params" list will be used as unique
            identifier.
        device: Current PyTorch device.
        mnet: The main network. If ``hnet`` is ``None``, then ``params`` are
            assumed to belong to this network. The fisher estimate will be
            computed accordingly.
            Note, ``params`` might be the output of a task-conditioned
            hypernetwork, i.e., weights for a specific task. In this case,
            "online"-EWC doesn't make much sense, as we don't follow the
            Bayesian view of using the old task weights as prior for the current
            ones. Instead, we have a new set of weights for all tasks.
        hnet (optional): If given, ``params`` is assumed to correspond to the
            unconditional weights :math:`\theta` (which does not include, for
            instance, task embeddings) of the hypernetwork. In this case, the
            diagonal Fisher entries belong to weights of the hypernetwork. The
            Fisher will then be computed based on the probability
            :math:`p(y \mid x, \text{task\_id})`, where ``task_id`` is just a
            constant input (representing the corresponding conditional weights,
            e.g., task embedding) in addition to the training samples :math:`x`.
        empirical_fisher: If ``True``, we compute the Fisher based on training
            targets.
        online: If ``True``, then we use online EWC, hence, there is only one
            diagonal Fisher approximation and one target parameter value stored
            at the time, rather than for all previous tasks.
        gamma: The gamma parameter for online EWC, controlling the gradual decay
            of previous tasks.
        n_max (optional): If not ``-1``, this will be the maximum amount of
            samples considered for estimating the Fisher.
        regression: Whether the task at hand is a classification or regression
            task. If ``True``, a regression task is assumed. For simplicity, we
            assume the following probabilistic model
            :math:`p(y \mid x) = \mathcal{N}\big(f(x), I\big)` with :math:`I`
            being the identity matrix. In this case, the only term of the log
            probability that influence the gradient is the MSE:
            :math:`\log p(y \mid x) = \lVert f(x) - y \rVert^2 + \text{const}`
        time_series (bool): If ``True``, the output of the main network
            ``mnet`` is expected to be a time series. In particular, we
            assume that the output is a tensor of shape ``[S, N, F]``,
            where ``S`` is the length of the time series, ``N`` is the batch
            size and ``F`` is the size of each feature vector (e.g., in
            classification, ``F`` would be the number of classes).

            Let :math:`\mathbf{y} = (\mathbf{y}_1, \dots \mathbf{y}_S)` be the
            output of the main network. We denote the parameters ``params`` by
            :math:`\theta` and the input by :math:`\mathbf{x}` (which we do not
            consider as random). We use the following decomposition of the
            likelihood
            
            .. math::
                
                p(\mathbf{y} \mid \theta; \mathbf{x}) =
                \prod_{i=1}^S p(\mathbf{y}_i \mid \mathbf{y}_1, \dots,
                \mathbf{y}_{i-1}, \theta; \mathbf{x}_i)

            **Classification:** If
            :math:`f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta)` denotes the output
            of the main network ``mnet`` for timestep :math:`i` (assuming
            :math:`\mathbf{h}_{i-1}` is the most recent hidden state), we assume

            .. math::

                p(\mathbf{y}_i \mid \mathbf{y}_1, \dots, \mathbf{y}_{i-1},
                \theta; \mathbf{x}_i) \equiv \text{softmax} \big(
                f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta) \big)

            Hence, we assume that we can write the negative log-likelihood (NLL)
            as follows given a label :math:`t \in [1, \dots, F]^S`:

            .. math::

                \text{NLL} &= - \log p(Y = t \mid \theta; \mathbf{x}) \\
                &= \sum_{i=1}^S - \text{softmax} \big(
                f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta)_{t_i} \big) \\
                &= \sum_{i=1}^S \text{cross\_entropy} \big(
                f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta), t_i \big)

            Thus, we simply sum the cross-entropy losses per time-step to
            estimate the NLL, which we then backpropagate through in order to
            compute the diagonal Fisher elements.
        allowed_outputs (optional): A list of indices, indicating which output
            neurons of the main network should be taken into account when
            computing the log probability. If not specified, all output neurons
            are considered.
        custom_forward (optional): A function handle that can replace the
            default procedure of forwarding samples through the given
            network(s).

            The default forward procedure if ``hnet`` is ``None`` is

            .. code:: python

                Y = mnet.forward(X, weights=params)

            Otherwise, the default forward procedure is

            .. code:: python

                weights = hnet.forward(task_id, theta=params)
                Y = mnet.forward(X, weights=weights)

            The signature of this function should be as follows.
                - ``hnet`` is ``None``: :code:`@fun(mnet, params, X)`
                - ``hnet`` is not ``None``:
                  :code:`@fun(mnet, hnet, task_id, params, X)`

            where :code:`X` denotes the input batch to the main network (usually
            consisting of a single sample).

            Example:
                Imagine a situation where the main network uses context-
                dependent modulation (cmp.
                :class:`utils.context_mod_layer.ContextModLayer`) and the
                parameters of these context-mod layers are produced by the
                hypernetwork ``hnet``, whereas the remaining weights of the
                main network ``mnet`` are maintained internally and passed as
                argument ``params`` to this method.

                In particular, we look at a main network that is an instance
                of class :class:`mnets.mlp.MLP`. The forward pass through this
                combination of networks should be handled as follows in order
                to compute the correct fisher matrix:

                .. code:: python

                    def custom_forward(mnet, hnet, task_id, params, X):
                        mod_weights = hnet.forward(task_id)
                        weights = {
                            'mod_weights': mod_weights,
                            'internal_weights': params
                        }
                        Y = mnet.forward(X, weights=weights)
                        return Y
        custom_nll (optional): A function handle that can replace the default 
            procedure of computing the negative-log-likelihood (NLL), which is
            required to compute the Fisher.

            The signature of this function should be as follows:
                :code:`@fun(Y, T, data, allowed_outputs, empirical_fisher)`

            where ``Y`` are the outputs of the main network. Note,
            ``allowed_outputs`` have already been applied to ``Y``, if given.
            ``T`` is the target provided by the dataset ``data``, transformed as
            follows:

            .. code:: python

                T = data.output_to_torch_tensor(batch[1], device,
                                                mode='inference')

            The arguments ``data``, ``allowed_outputs`` and ``empirical_fisher``
            are only passed for convinience (e.g., to apply simple sanity checks
            using assertions).

            The output of the function handle should be the NLL for the given
            sample.
        pass_ids (bool): If a ``custom_nll`` is used and this flag is ``True``,
            then the signature of the ``cutom_nll`` is expected to be:

            .. code:: python

                @fun(Y, T, data, allowed_outputs, empirical_fisher, batch_ids)

            where ``batch_ids`` are the unique identifiers as returned by
            option ``return_ids`` of method
            :meth:`data.dataset.Dataset.next_train_batch` corresponding to the
            provided samples.

            Example:
                In sequential datasets, target sequences ``T`` might be padded
                to the same length. Though, if the unpadded length should be
                used for NLL computation, then the ``custom_nll`` function needs
                the ability to request this information (sequence length) from
                ``data``.

            Also, the signatures of ``custom_forward`` are expected to be
            different.

            The signature of this function should be as follows.

            - ``hnet`` is ``None``: ``@fun(mnet, params, X, data, batch_ids)``
            - ``hnet`` is not ``None``:
              ``@fun(mnet, hnet, task_id, params, X, data, batch_ids)``
        proper_scaling (bool): The algorithm `Online EWC` is based on a Taylor
            approximation of the posterior that leads to the following
            estimate

            .. math::
                \log p(\theta \mid \mathcal{D}_1, \cdots, \mathcal{D}_T)
                \approx \log p(\mathcal{D}_T \mid \theta) -
                \frac{1}{2}\sum_i \bigg( \sum_{t < T} N_t
                \mathcal{F}_{emp \hspace{1mm}t, i}  +
                \frac{1}{\sigma_{prior}^2}   \bigg)
                (\theta_i - \theta_{S, i}^*)^2 + \text{const}

            Due to the presentation of the algorithm in the paper and inspired
            by multiple publicly implementations, we approximate the
            regularization strength in practice via

            .. math::

                \sum_{t < T} N_t \mathcal{F}_{emp \hspace{1mm}t, i}  +
                \frac{1}{\sigma_{prior}^2} \approx \lambda
                \sum_{t < T} \mathcal{F}_{emp \hspace{1mm}t, i}

            where :math:`\lambda` is a hyperparameter.

            If this argument is ``True``, then the sum of Fisher matrices is
            properly weighted by the dataset size (independent of argument
            ``n_max``).
        prior_strength (float or list, optional): Either a scalar or a list of
            Tensors with the same shapes as ``params``. Only applies to
            `Online EWC`. One can specify an offset for all Fisher values, e.g.,
            :math:`\frac{1}{\sigma_{prior}^2}`. See argument ``proper_scaling``
            for details.
        regression_lvar (float): In regression, this refers to the variance of
            the likelihood.
        target_manipulator (func, optional): A function with signature

            .. code:: python
    
                T = target_manipulator(T)
    
            That may manipulate the targets coming from the dataset.
    """
    assert isinstance(mnet, MainNetInterface)
    assert mnet.has_linear_out

    # TODO
    if hnet is not None:
        raise NotImplementedError()

    assert hnet is None or task_id is not None
    assert not online or (gamma >= 0. and gamma <= 1.)
    assert n_max == -1 or n_max > 0

    if time_series and regression:
        raise NotImplementedError('Computing the Fisher for a recurrent ' +
                                  'regression task is not yet implemented.')

    if not online:
        if proper_scaling:
            # Doesn't hurt, we can get rid of warning.
            warn('Argument "proper_scaling" is only well justified in ' +
                 'Online EWC.')
        if prior_strength is not None:
            # We have a separate Fisher per task for EWC.
            raise ValueError('Option "prior_strength" only applicable to ' +
                             'Online EWC.')

    n_samples = data.num_train_samples
    if n_max != -1:
        n_samples = min(n_samples, n_max)

    mnet_mode = mnet.training
    mnet.eval()
    if hnet is not None:
        hnet_mode = hnet.training
        hnet.eval()

    fisher = []
    for ii, p in enumerate(params):
        if prior_strength is None:
            fisher.append(torch.zeros_like(p))
        elif isinstance(prior_strength, (list, tuple)):
            assert len(prior_strength) == len(params) and \
                prior_strength[ii].shape == p.shape
            fisher.append(prior_strength[ii].clone())
        else:
            fisher.append(torch.ones_like(p) * prior_strength)

        assert p.requires_grad # Otherwise, we can't compute the Fisher.

    # Ensure, that we go through all training samples (note, that training
    # samples are always randomly shuffled when using "next_train_batch", but
    # we always go through the complete batch before reshuffling the samples.)
    # If `n_max` was specified, we always go through a different random
    # subsample of the training set.
    data.reset_batch_generator(train=True, test=False, val=False)

    # Since the PyTorch grad function accumulates gradients, we have to go
    # through single training samples.
    for s in range(n_samples):
        batch = data.next_train_batch(1, return_ids=pass_ids)
        X = data.input_to_torch_tensor(batch[0], device, mode='inference')
        T = data.output_to_torch_tensor(batch[1], device, mode='inference')
        if target_manipulator is not None:
            T = target_manipulator(T)

        if hnet is None:
            if custom_forward is None:
                Y = mnet.forward(X, weights=params)
            else:
                if pass_ids:
                    Y = custom_forward(mnet, params, X, data, batch[2])
                else:
                    Y = custom_forward(mnet, params, X)
        else:
            if custom_forward is None:
                weights = hnet.forward(task_id, theta=params)
                Y = mnet.forward(X, weights=weights)
            else:
                if pass_ids:
                    Y = custom_forward(mnet, hnet, task_id, params, X, data,
                                       batch[2])
                else:
                    Y = custom_forward(mnet, hnet, task_id, params, X)

        if not time_series:
            assert(len(Y.shape) == 2)
        else:
            assert(len(Y.shape) == 3)

        if allowed_outputs is not None:
            if not time_series:
                Y = Y[:, allowed_outputs]
            else:
                Y = Y[:, :, allowed_outputs]

        ### Compute negative log-likelihood.
        if custom_nll is not None:
            if pass_ids:
                nll = custom_nll(Y, T, data, allowed_outputs, empirical_fisher,
                                 batch[2])
            else:
                nll = custom_nll(Y, T, data, allowed_outputs, empirical_fisher)

        elif regression:
            # Note, if regression, we don't have to modify the targets.
            # Thus, through "allowed_outputs" Y has been brought into the same
            # shape as T.

            if empirical_fisher:
                # The term that doesn't vanish in the gradient of the log
                # probability is the squared L2 norm between Y and T.
                nll = 0.5 / regression_lvar * (Y - T).pow(2).sum()
            else:
                raise NotImplementedError('Only empirical Fisher is ' +
                                          'implemented so far!')

        else:
            # Note, we assume the output of the main network is linear, such
            # that we can compute the log probabilities by applying the log-
            # softmax to these outputs.

            assert data.classification and len(data.out_shape) == 1
            if allowed_outputs is not None:
                assert target_manipulator is not None or \
                    len(allowed_outputs) == data.num_classes
                assert Y.shape[2 if time_series else 1] == len(allowed_outputs)

            # Targets might be labels or one-hot encodings.
            if data.is_one_hot:
                if time_series:
                    assert(len(T.shape) == 3 and T.shape[2] == Y.shape[2])
                    T = torch.argmax(T, 2)
                else:
                    # Note, this function processes always one sample at a time
                    # (batchsize=1), so `T` contains a single number.
                    T = torch.argmax(T)

            # Important, distinguish between empiricial and normal fisher!
            if empirical_fisher:
                if not time_series:
                    # For classification, only the loss associated with the
                    # target unit is taken into consideration.
                    nll = F.nll_loss(F.log_softmax(Y, dim=1),
                                     torch.tensor([T]).to(device))
                else:
                    ll = F.log_softmax(Y, dim=2) # log likelihood for all labels
                    # We need to swap dimenstions from [S, N, F] to [S, F, N].
                    # See documentation of method `nll_loss`.
                    ll = ll.permute(0, 2, 1)
                    nll = F.nll_loss(ll, T, reduction='none')
                    # Mean across batch dimension, but sum across time-series
                    # dimension.
                    assert(len(nll.shape) == 2)
                    nll = nll.mean(dim=1).sum()
            else:
                raise NotImplementedError('Only empirical Fisher is ' +
                                          'implemented so far!')

        ### Compute gradient of negative log likelihood to estimate Fisher
        mnet.zero_grad()
        if hnet is not None:
            hnet.zero_grad()
        torch.autograd.backward(nll, retain_graph=False, create_graph=False)

        for i, p in enumerate(params):
            fisher[i] += torch.pow(p.grad.detach(), 2)

        # This version would not require use to call zero_grad and hence, we
        # wouldn't fiddle with internal variables, but it would require us to
        # loop over tensors and retain the graph in between.
        #for p in params:
        #    g = torch.autograd.grad(nll, p, grad_outputs=None,
        #                retain_graph=True, create_graph=False,
        #                only_inputs=True)[0]
        #    fisher[i] += torch.pow(g.detach(), 2)

    for i in range(len(params)):
        if not proper_scaling:
            fisher[i] /= n_samples
        elif n_samples != data.num_train_samples:
            fisher[i] *= data.num_train_samples / n_samples

    ### Register buffers to store current task weights as well as the Fisher.
    net = mnet
    if hnet is not None:
        net = hnet
    for i, p in enumerate(params):
        buff_w_name, buff_f_name = _ewc_buffer_names(task_id, i, online)

        # We use registered buffers rather than class members to ensure that
        # these variables appear in the state_dict and are thus written into
        # checkpoints.
        net.register_buffer(buff_w_name, p.detach().clone())

        # In the "online" case, the old fisher estimate buffer will be
        # overwritten.
        if online and task_id > 0:
            prev_fisher_est = getattr(net, buff_f_name)

            # Decay of previous fisher.
            fisher[i] += gamma * prev_fisher_est

        net.register_buffer(buff_f_name, fisher[i].detach().clone())

    mnet.train(mode=mnet_mode)
    if hnet is not None:
        hnet.train(mode=hnet_mode)

def ewc_regularizer(task_id, params, mnet, hnet=None,
                    online=False, gamma=1.):
    """Compute the EWC regularizer, that can be added to the remaining loss.
    Note, the hyperparameter, that trades-off the regularization strength is
    not yet multiplied by the loss.

    This loss assumes an appropriate use of the method "compute_fisher". Note,
    for the current task "compute_fisher" has to be called after calling this
    method.

    If `online` is False, this method implements the loss proposed in eq. (3) in
    [EWC2017]_, except for the missing hyperparameter `lambda`.
    
    The online EWC implementation follows eq. (8) from [OnEWC2018]_ (note, that
    lambda does not appear in this equation, but it was used in their
    experiments).

    .. [EWC2017] https://arxiv.org/abs/1612.00796
    .. [OnEWC2018] https://arxiv.org/abs/1805.06370

    Args:
        (....): See docstring of method :func:`compute_fisher`.

    Returns:
        EWC regularizer.
    """
    assert(task_id > 0)

    net = mnet
    if hnet is not None:
        net = hnet

    ewc_reg = 0

    num_prev_tasks = 1 if online else task_id
    for t in range(num_prev_tasks):
        for i, p in enumerate(params):
            buff_w_name, buff_f_name = _ewc_buffer_names(t, i, online)

            prev_weights = getattr(net, buff_w_name)
            fisher_est = getattr(net, buff_f_name)
            # Note, since we haven't called "compute_fisher" yet, the forgetting
            # scalar has not been multiplied yet.
            if online:
                fisher_est *= gamma

            ewc_reg += (fisher_est * (p - prev_weights).pow(2)).sum()

    # Note, the loss proposed in the original paper is not normalized by the
    # number of tasks
    #return ewc_reg / num_prev_tasks / 2.
    return ewc_reg / 2.

def _ewc_buffer_names(task_id, param_id, online):
    """The names of the buffers used to store EWC variables.

    Args:
        task_id: ID of task (only used of `online` is False).
        param_id: Identifier of parameter tensor.
        online: Whether the online EWC algorithm is used.

    Returns:
        (tuple): Tuple containing:

        - **weight_buffer_name**
        - **fisher_estimate_buffer_name**
    """
    task_ident = '' if online else '_task_%d' % task_id

    weight_name = 'ewc_prev{}_weights_{}'.format(task_ident, param_id)
    fisher_name = 'ewc_fisher_estimate{}_weights_{}'.format(task_ident,
                                      param_id)
    return weight_name, fisher_name

def context_mod_forward(mod_weights=None):
    """Create a custom forward function for function :func:`compute_fisher`.

    See argument ``custom_forward`` of function :func:`compute_fisher` for more
    details.

    This is a helper method to quickly retrieve a function handle that manages
    the forward pass for a context-modulated main network.

    We assume that the interface of the main network is similar to the one of
    :meth:`mnets.mlp.MLP.forward`.

    Args:
        mod_weights (optional): If provided, it is assumed that
            :func:`compute_fisher` is called with ``hnet`` set to ``None``.
            Hence, the returned function handle will have the given
            context-modulation pattern hard-coded.
            If left unspecified, it is assumed that a ``hnet`` is passed to
            :func:`compute_fisher` and that this ``hnet`` computes only the
            parameters of all context-mod layers.

    Returns:
        A function handle.
    """
    def hnet_forward(mnet, hnet, task_id, params, X):
        mod_weights = hnet.forward(task_id)
        weights = {
            'mod_weights': mod_weights,
            'internal_weights': params
        }
        Y = mnet.forward(X, weights=weights)
        return Y

    def mnet_only_forward(mnet, params, X):
        weights = {
            'mod_weights': mod_weights,
            'internal_weights': params
        }
        Y = mnet.forward(X, weights=weights)
        return Y

    if mod_weights is None:
        return hnet_forward
    else:
        return mnet_only_forward

if __name__ == '__main__':
    pass


