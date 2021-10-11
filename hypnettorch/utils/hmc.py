#!/usr/bin/env python3
# Copyright 2021 Christian Henning
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
# @title          :utils/hmc.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :03/09/2021
# @version        :1.0
# @python_version :3.8.5
r"""
Hamiltonian-Monte-Carlo
-----------------------

The module :mod:`utils.hmc` implements the Hamiltonian-Monte-Carlo (HMC)
algorithm as described in

    Neal, `MCMC using Hamiltonian dynamics <https://arxiv.org/abs/1206.1901>`__,
    2012.

The pseudocode of the algorithm is described in Figure 2 of the paper. The
algorithm uses the Leapfrog algorithm to simulate the Hamiltonian dynamics in
discrete time. Therefore, two crucial hyperparameters are required: the stepsize
:math:`\epsilon` and the number of steps :math:`L`. Both hyperparameters have to
be chosen with care and can drastically influence the behavior of HMC. If the
stepsize :math:`\epsilon` is too small, we don't explore the state space
efficiently and waste computation. If it is too big, the numerical error from
the discretization might be come too huge and the acceptance rate rather low. In
addition, we want to choose :math:`L` large enough to obtain good exploration,
but if we set it too large we might loop back to the starting position.

The No-U-Turn-Sampler (NUTS) has been proposed to set :math:`L` automatically,
such that only the stepsize :math:`\epsilon` has to be chosen.

    Hoffman et al.,
    "`The No-U-Turn Sampler: <https://arxiv.org/abs/1111.4246>`__
    Adaptively Setting Path Lengths in Hamiltonian Monte Carlo", 2011.

This module provides implementations for both variants, basic :class:`HMC` and
:class:`NUTS`. Multiple parallel chains can be simulated via class
:class:`MultiChainHMC`. For Bayesian Neural Networks, the helper function
:func:`nn_pot_energy` can be used to define the potential energy.

**Notation**

We largely follow the notation from
`Neal et al. <https://arxiv.org/abs/1206.1901>`__. The variable of interest,
e.g., model parameters, are encoded by the position vector :math:`q`. In
addition, HMC requires a momentum :math:`p`. The Hamiltonian :math:`H(q, p)`
consists of two terms, the potential energy :math:`U(q)` and the kinetic energy
:math:`K(p) = p^T M^{-1} p / 2` with :math:`M` being a symmetric, p.d. "mass"
matrix.

The Hamiltonian dynamics can thus be summarized as

.. math::

    \frac{dq_i}{dt} &= \frac{\partial H}{\partial p_i} = [M^{-1} p]_i \\
    \frac{dp_i}{dt} &= -\frac{\partial H}{\partial q_i} = \
        - \frac{\partial U}{\partial q_i}

The Leapfrog algorithm is a way to discretize the differential equation above
in a way that is reversible and volumne preserving. The algorithm has two
hyperparameters: the stepsize :math:`\epsilon` and the number of steps
:math:`L`. Below, we sketch the algorithm to update momentum and position from
time :math:`t` to time :math:`t + L\epsilon`.

.. math::

    p_i(t + \frac{\epsilon}{2}) &= p_i(t) - \frac{\epsilon}{2} \
        \frac{\partial U}{\partial q_i} \big( q(t) \big) \\
    q_i(t + l\epsilon) &= q_i(t + (l-1)\epsilon) + \epsilon \
        \frac{p_i(t + (l-1)\epsilon + \epsilon/2)}{m_i} \quad \forall l = 1..L\\
    p_i(t + l\epsilon + \frac{\epsilon}{2}) &= \
        p_i(t + (l-1)\epsilon + \frac{\epsilon}{2}) - \epsilon \
        \frac{\partial U}{\partial q_i} \big( q(t+l\epsilon) \big) \
        \quad \forall l = 1..L-1\\
    p_i(t + L\epsilon) &= p_i(t + (L-1)\epsilon + \frac{\epsilon}{2}) -\
        \frac{\epsilon}{2} \frac{\partial U}{\partial q_i} \
        \big( q(t+L\epsilon) \big)

We assume a diagonal mass matrix in the position update above.

.. autosummary::

    hypnettorch.utils.hmc.HMC
    hypnettorch.utils.hmc.MCMC
    hypnettorch.utils.hmc.MultiChainHMC
    hypnettorch.utils.hmc.NUTS
    hypnettorch.utils.hmc.leapfrog
    hypnettorch.utils.hmc.log_prob_standard_normal_prior
    hypnettorch.utils.hmc.nn_pot_energy
"""
import logging
import numpy as np
from os import path
from queue import Queue
import sys
from tensorboardX import SummaryWriter
from threading import Thread
import torch
from torch.distributions import Normal, MultivariateNormal
import torch.nn.functional as F
from warnings import warn

from hypnettorch.mnets.mnet_interface import MainNetInterface

def _grad_pot_energy(pot_energy, position):
    r"""Compute the partial derivatives of the potential energy for the current
    position :math:`q(t)`.

    Args:
        (....): See docstring of function :func:`leapfrog`.

    Returns:
        (torch.Tensor): The vector of partial derivatives
        :math:`\frac{\partial U}{\partial q} \big( q(t) \big)`.
    """
    pe_val = pot_energy(position)
    pot_grad, = torch.autograd.grad(pe_val, position, only_inputs=True)
    return pot_grad

def _grad_kin_energy(momentum, inv_mass):
    r"""Compute the partial derivatives of the kinetic energy for the current
    momentum :math:`p(t)`.

    This function assumes a kinetic energy of the form
    :math:`K(p) = p^T M^{-1} p / 2`.

    Args:
        (....): See docstring of function :func:`leapfrog`.

    Returns:
        (torch.Tensor): The vector of partial derivatives
        :math:`\frac{\partial K}{\partial p} \big( p(t) \big)`.
    """
    if isinstance(inv_mass, torch.Tensor):
        if inv_mass.numel() == momentum.numel(): # Assuming diagonal mass
            grad_ke = inv_mass * momentum # Element-wise product.
        else: # Assuming full matrix
            grad_ke = torch.matmul(inv_mass, momentum)
    else: # Assuming a single scalar
        grad_ke = inv_mass * momentum

    return grad_ke

def _kin_energy(momentum, inv_mass):
    """Compute the kinetic energy for the current momentum :math:`p(t)`.

    This function assumes a kinetic energy according to
    :math:`K(p) = p^T M^{-1} p / 2`.

    Args:
        (....): See docstring of function :func:`leapfrog`.

    Returns:
        (torch.Tensor): The scalar energy value.
    """
    if isinstance(inv_mass, torch.Tensor):
        if inv_mass.numel() == momentum.numel(): # Assuming diagonal mass
            ke = .5 * torch.dot(momentum, inv_mass * momentum)
        else: # Assuming full matrix
            ke = .5 *  torch.dot(momentum, torch.matmul(inv_mass, momentum))
    else: # Assuming a single scalar
        ke = .5 * inv_mass * torch.dot(momentum, momentum)

    return ke

def leapfrog(position, momentum, stepsize, num_steps, inv_mass, pot_energy):
    r"""Implementation of the leapfrog algorithm.

    The leapfrog algorithm updates position :math:`q` and momentum :math:`p`
    variables by simulating the Hamiltonian dynamics in discrete time for a
    time window of size :math:`L\epsilon`, where :math:`L` is the number of
    leapfrog steps ``num_steps`` and :math:`\epsilon` is the ``stepsize``.

    In general, one can call this method :math:`L` times while setting
    ``num_steps=1`` in order to obtain the complete trajectory. However, if not
    necessary, we recommend setting ``num_steps=L`` to save the unnecessary
    computation of intermediate momentum variables.

    Args:
        position (torch.Tensor): The position variable :math:`q`.
        momentum (torch.Tensor): The momentum variable :math:`p`.
        stepsize (float): The leapfrog stepsize :math:`\epsilon`.
        num_steps (int): The number of leapfrog steps :math:`L`.
        inv_mass (float or torch.Tensor): The inverse mass matrix
            :math:`M^{-1}`. Can also be provided as vector, in case of a
            diagonal mass matrix, or as scalar.
        pot_energy (func): A function handle that computes the potential energy
            :math:`U\big( q(t) \big)`, receiving as only input the current
            position variable.

            Note:
                The function handle ``pot_energy`` has to be amenable to
                :mod:`torch.autograd`, as the momentum update requires the
                partial derivatives of the potential energy.

    Returns:
        (tuple): Tuple containing:

            - **position** (torch.Tensor): The updated position variable.
            - **momentum** (torch.Tensor): The updated momentum variable.
    """
    # p(t + \epsilon/2)
    momentum = momentum - .5 * stepsize * _grad_pot_energy(pot_energy, position)

    for l in range(num_steps):
        # Compute the gradient of the kinetic energy.
        grad_ke = _grad_kin_energy(momentum, inv_mass)

        # q(t + (l+1)\epsilon)
        position = position + stepsize * grad_ke

        if l < num_steps-1:
            # p(t + (l+1)\epsilon + \epsilon/2)
            momentum = momentum - \
                stepsize * _grad_pot_energy(pot_energy, position)

    # p(t + L\epsilon)
    momentum = momentum - .5 * stepsize * _grad_pot_energy(pot_energy, position)

    return position, momentum

def log_prob_standard_normal_prior(position, mean=0., std=1.):
    r"""Log-probability density of a standard normal prior.

    This function can be used to compute :math:`\log p(q)` for
    :math:`p(q) = \mathcal{N}(q; \bm{\mu}, I \bm{\sigma}^2)`, where :math:`I`
    denotes the identity matrix.

    This function can be passed to :func:`nn_pot_energy` as argument
    ``prior_log_prob_func`` using, for instance:

    .. code-block:: python

        lp_func = lambda q: log_prob_standard_normal_prior(q, mean=0., std=.02)

    Args:
        position (torch.Tensor): The position variable :math:`q`.
        mean (float or torch.Tensor): The mean of the diagonal Gaussian prior.
        std (float or torch.Tensor): The diagonal covariance of the Gaussian
            prior.
    """
    prior_dist = Normal(mean, std)

    return prior_dist.log_prob(position).sum()


def nn_pot_energy(net, inputs, targets, prior_log_prob_func, tau_pred= 1.,
                  nll_type='regression'):
    r"""The potential energy for Bayesian inference with HMC using neural
    networks.

    When obtaining samples from the posterior parameter distribution of a neural
    network via HMC, a potential energy function has to be specified that allows
    evaluating the negative log-posterior up to a constant. We consider a neural
    network with parameters :math:`W` which encodes a likelihood function
    :math:`p(y \mid W; x)` for an input :math:`x`. In addition, a prior
    :math:`p(W)` needs to be specified. Given a dataset :math:`\mathcal{D}`
    consisting of ``inputs`` :math:`x_n` and ``targets`` :math:`y_n`, we can
    specify the potential energy as (note, here :math:`q = W`)

    .. math::

        U(W) &= - \log p(\mathcal{D} \mid W) - \log p(W) \\
            &= - \sum_n \log p(y_n \mid W; x_n) - \log p(W)

    where the first term corresponds to the negative log-likelihood (NLL). The
    precise way of computing the NLL depends on which kind of likelihood
    interpretation is forced onto the network (cf. argument ``nll_type``).

    Args:
        net (mnets.mnet_interface.MainNetInterface): The considered neural
            network, whose parameters are :math:`W`.
        inputs (torch.Tensor): A tensor containing all the input sample points
            :math:`x_n` in :math:`\mathcal{D}`.
        targets (torch.Tensor): A tensor containing all the output sample points
            :math:`y_n` in :math:`\mathcal{D}`.
        prior_log_prob_func (func): Function handle that allows computing the
            log-probability density of the prior for a given position variate.
        tau_pred (float): Only applies to ``nll_type='regression'``. The inverse
            variance of the assumed Gaussian likelihood.
        nll_type (str): The type of likelihood interpretation enforced on the
            network. The following options are supported:

            - ``'regression'``: The network outputs the mean of a 1D normal
              distribution with fixed variance.

              .. math::

                  \text{NLL} = \frac{1}{2 \sigma_\text{ll}^2} \
                      \sum_{(x, y) \in \mathcal{D}} \
                      \big( f_\text{M}(x, W) - y \big)^2

              where :math:`f_\text{M}(x, W)` is the network output and
              :math:`\frac{1}{\sigma_\text{ll}^2}` corresponds to ``tau_pred``.

            - ``'classification'``: Multi-class classification with a softmax
              likelihood. Note, we assume the network has linear (logit) outputs

              .. math::

                  \text{NLL} = \sum_{(\mathbf{x}, y) \in \mathcal{D}} \bigg( \
                      \underbrace{ - \sum_{c=0}^{C-1} [c = y] \log \Big( \
                      \text{softmax} \big( f_\text{M}(\mathbf{x}, W) \big)_c \
                      }_{\text{cross-entropy loss with 1-hot targets}} \Big) \
                      \bigg)

              where :math:`C` is the number of classes and :math:`y` are
              integer labels. We assume that the neural network
              :math:`f_\text{M}(\mathbf{x}, W)` outputs logits.

              .. note::

                  We assume ``targets`` contains integer labels and **not**
                  1-hot encodings for ``'classification'``!

    Returns:
        (func): A function handle as required by constructor argument
        ``pot_energy_func`` of class :class:`HMC`.
    """
    assert nll_type in ['regression', 'classification']
    if nll_type != 'regression':
        assert tau_pred == 1.

    def pot_energy_func(position):
        weights = MainNetInterface.flatten_params(position,
            param_shapes=net.param_shapes, unflatten=True)

        preds = net(inputs, weights=weights)

        if nll_type == 'regression':
            nll = 0.5 * tau_pred * F.mse_loss(preds, targets, reduction='sum')
        elif nll_type == 'classification':
            # Note, we assume that `targets` are integer labels!
            nll = F.cross_entropy(preds, targets, reduction='sum')
        else:
            raise NotImplementedError()

        return nll - prior_log_prob_func(position)

    return pot_energy_func

class HMC:
    r"""This class represents the basic HMC algorithm.

    The algorithm is implemented as outlined in Fig. 2 of
    `Neal et al. <https://arxiv.org/abs/1206.1901>`__.

    The potential energy should be the negative log probability density of the
    target distribution to sample from (up to a constant)
    :math:`U(q) = - \log p(q) + \text{const.}`.

    Args:
        initial_position (torch.Tensor): The initial position :math:`q(0)`.

            Note:
                The position variable should be provided as vector. The weights
                of a neural network can be flattend via
                :meth:`mnets.mnet_interface.MainNetInterface.flatten_params`.
        pot_energy_func (func): A function handle computing the potential
            energy :math:`U(q)` upon receiving a position :math:`q`. To sample
            the weights of a neural network, the helper function
            :func:`nn_pot_energy` can be used. To sample via HMC from a target
            distribution implemented via
            :class:`torch.distributions.distribution.Distribution`, one can
            define a function handle as in the following example.

            Example:
                .. code-block:: python

                    d = MultivariateNormal(torch.zeros(4), torch.eye(4))
                    pot_energy_func = lambda q : - d.log_prob(q)
        stepsize (float): The stepsize :math:`\epsilon` of the :func:`leapfrog`
            algorithm.
        num_steps (int): The number of steps :math:`L` in the :func:`leapfrog`
            algorithm.
        inv_mass (float or torch.Tensor): The inverse "mass" matrix as required
            for the computation of the kinetic energy :math:`K(p)`. See argument
            ``inv_mass`` of function :func:`leapfrog` for details.
        logger (logging.Logger, optional): If provided, the progress will be
            logged.
        log_interval (int): After how many states the status should be logged.
        writer (tensorboardX.SummaryWriter, optional): A tensorboard writer.
            If given, useful simulation data will be logged, like the
            developement of the Hamiltonian.
        writer_tag (str): Will be added to the tensorboard tags.
    """
    def __init__(self, initial_position, pot_energy_func, stepsize=.02,
                 num_steps=1, inv_mass=1., logger=None, log_interval=100,
                 writer=None, writer_tag=''):
        self._position = initial_position
        if not self._position.requires_grad:
            self._position.requires_grad = True
        self._stepsize = stepsize
        self._num_steps = num_steps
        self._pot_energy_func = pot_energy_func
        self._inv_mass = inv_mass
        self._logger = logger
        self._log_interval = log_interval
        self._writer = writer
        self._writer_tag = writer_tag

        self._positions = [initial_position]
        self._num_states = 0
        self._accumulated_accept = 0

        # Define distribution from which to sample momentum.
        if isinstance(inv_mass, torch.Tensor) and len(inv_mass.shape) == 2:
            #mass = torch.inverse(inv_mass)
            self._momentum_dist = MultivariateNormal( \
                torch.zeros_like(initial_position), precision_matrix=inv_mass)
        else:
            mass = 1. / inv_mass
            # Note, that we need to pass the standard deviation and not
            # variance.
            self._momentum_dist = Normal(torch.zeros_like(initial_position),
                                         mass**0.5)

    @property
    def stepsize(self):
        """The stepsize :math:`\epsilon` of the :func:`leapfrog` algorithm.

        You may adapt the stepsize at any point.

        :type: float
        """
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value):
        self._stepsize = value

    @property
    def num_steps(self):
        """The number of steps :math:`L` in the :func:`leapfrog` algorithm.

        You may adapt the number of steps at any point.

        :type: int
        """
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value):
        self._num_steps = value

    @property
    def current_position(self):
        """The latest position :math:`q(t)` in the chain simulated so far.

        :type: torch.Tensor
        """
        return self._position

    @property
    def num_states(self):
        """The number of states in the chain visited so far.

        The counter will be increased by method :meth:`simulate_chain`.

        :type: int
        """
        return self._num_states

    @property
    def position_trajectory(self):
        """A list containing all position variables (Markov states) visited so
        far.

        New positions will be added by the method :meth:`simulate_chain`. To
        decrease the memory footprint of objects in this class, the trajectory
        can be cleared via method :meth:`clear_position_trajectory`.

        :type: list
        """
        return self._positions

    @property
    def acceptance_probability(self):
        """The fraction of states that have been accepted.

        :type: float
        """
        if self.num_states == 0:
            return 1.0

        return self._accumulated_accept / self._num_states

    def clear_position_trajectory(self, n=None):
        """Reset attribute :attr:`position_trajectory`.

        This method will no affect the counter :attr:`num_states`.

        Args:
            n (int, optional): If provided, only the first ``n`` elements of
                :attr:`position_trajectory` are discarded (e.g., the burn-in
                samples).
        """
        if n is not None:
            self._positions = self._positions[n:]
        else:
            self._positions = []

    def simulate_chain(self, n):
        """Simulate the next ``n`` states of the chain.

        The new states will be appended to attribute
        :attr:`position_trajectory`.

        Args:
            n (int): Number of HMC steps to be executed.
        """
        logger = self._logger
        writer = self._writer

        for _ in range(n):
            curr_q = self.current_position
            # Resample momentum.
            curr_p = self._momentum_dist.sample()
            #curr_p.requires_grad = True

            # Simulate Hamiltonian dynamics.
            q = curr_q.detach().clone()
            p = curr_p.detach().clone()
            if not q.requires_grad:
                q.requires_grad = True

            q, p = leapfrog(q, p, self.stepsize, self.num_steps, self._inv_mass,
                            self._pot_energy_func)

            # Negation of momentum not required in simulation.
            #p = -p

            # Evaluate Hamiltonian at beginning and end of trajectory.
            k_p_start = _kin_energy(curr_p, self._inv_mass)
            u_q_start = self._pot_energy_func(curr_q)

            k_p_proposal = _kin_energy(p, self._inv_mass)
            u_q_proposal = self._pot_energy_func(q)

            # Metropolis update.
            if torch.rand(1).to(p.device) < torch.exp(u_q_start-u_q_proposal + \
                                                      k_p_start-k_p_proposal):
                accept = True
                self._accumulated_accept += 1

                self._positions.append(q)
            else: # Reject
                accept = False
                self._positions.append(curr_q.clone())
            self._position = self._positions[-1]

            self._num_states += 1

            # Log progress.
            if accept:
                kinetic = k_p_proposal.detach().cpu().numpy()
                potential = u_q_proposal.detach().cpu().numpy()
            else:
                kinetic = k_p_start.detach().cpu().numpy()
                potential = u_q_start.detach().cpu().numpy()
            hamiltonian = kinetic + potential

            if logger is not None and \
                    (self.num_states-1) % self._log_interval == 0:
                logger.debug('HMC state %d: Current Hamiltonian: %f - ' \
                    % (self.num_states, hamiltonian) + \
                    'Acceptance probability: %.2f%%.' \
                    % (self.acceptance_probability * 100))

            if writer is not None:
                tag = self._writer_tag
                writer.add_scalar('%shmc/kinetic' % tag, kinetic,
                                  global_step=self.num_states,
                                  display_name='Kinetic Energy')
                writer.add_scalar('%shmc/potential' % tag, potential,
                                  global_step=self.num_states,
                                  display_name='Potential Energy')
                writer.add_scalar('%shmc/hamiltonian' % tag, hamiltonian,
                                  global_step=self.num_states,
                                  display_name='Hamiltonian')
                writer.add_scalar('%shmc/accept' % tag,
                                  self.acceptance_probability,
                                  global_step=self.num_states,
                                  display_name='Acceptance Probability')

class NUTS(HMC):
    r"""HMC with No U-Turn Sampler (NUTS).

    In this class, we implement the efficient version of the NUTS algorithm
    (see algorithm 3 in `Hoffman et al. <https://arxiv.org/abs/1111.4246>`__).

    NUTS eliminates the need to choose the number of Leapfrog steps :math:`L`.
    While the algorithm is more computationally expensive than basic HMC, the
    reduced hyperparameter effort has been shown to reduce the overall
    computational cost (and it requires less human intervention).

    As explained in the paper, a good heuristic to set :math:`L` is to choose
    the highest number (for given :math:`\epsilon`) before the trajectory loops
    back to the initial position :math:`q_0`, e.g., when the following quantity
    becomes negative

    .. math::
        \frac{d}{dt} \frac{1}{2} \lVert q - q_0 \rVert_2^2 = \
            \langle q- q_0, p \rangle

    Note, this equation assumes the `mass matrix` is the identity: :math:`M=I`.

    However, this approach is in general not time reversible, therefore NUTS
    proposes a recursive agorithm that allows backtracing. NUTS randomly adds
    subtrees to a balanced binary tree and stops when any of those subtrees
    starts making a "U-turn" (either forward or backward in time). This tree
    construction is fully symmetric and therefore reversible.

    Note:
        The NUTS paper also proposes to combine a heuristic approach to adapt
        the stepsize :math:`\epsilon` together with :math:`L` (e.g., see
        algorithm 6 in `Hoffman et al. <https://arxiv.org/abs/1111.4246>`__).

        Such stepsize adaptation is currently not implemented by this class!

    Args:
        (....): See docstring of class :class:`HMC`.
        delta_max (float): The nonnegative criterion :math:`\Delta_\text{max}`
            from Eq. 8 of `Hoffman et al. <https://arxiv.org/abs/1111.4246>`__,
            that should ensure that we stop NUTS if the energy becomes too big.
    """
    def __init__(self, initial_position, pot_energy_func, stepsize=.02,
                 delta_max=1000., inv_mass=1., logger=None, log_interval=100,
                 writer=None, writer_tag=''):
        HMC.__init__(self, initial_position, pot_energy_func, stepsize=stepsize,
                     num_steps=None, inv_mass=inv_mass, logger=logger,
                     log_interval=log_interval, writer=writer,
                     writer_tag=writer_tag)

        self._delta_max = delta_max

    # Overwrite base attribute.
    @property
    def num_steps(self):
        """The attribute :attr:`HMC.num_steps` does not exist for class
        :class:`NUTS`! Accessing this attribute will cause an error.
        """
        raise RuntimeError('NUTS has no attribute "num_steps".')

    @num_steps.setter
    def num_steps(self, value):
        raise RuntimeError('NUTS has no attribute "num_steps".')

    def _u_turn(self, q1, q2, p):
        """Detect whether a U-Turn has been made.

        This method simply computes

        .. math::

            \Big[ \langle q2- q_1, M^-1 p \rangle \leq 0 \Big]

        where :math:`[\cdot]` denotes the Iverson bracket.

        Returns:
            (int)
        """
        # Note that the product of inverse mass matrix and momentum is simply
        # the gradient of the kinetic energy.
        angle = torch.dot(q2 - q1, _grad_kin_energy(p, self._inv_mass))
        return int(angle >= 0)

    def _build_tree(self, q, p, u, v, j):
        """Build the NUTS tree recursively.

        See function "BuildTree" in algorithm 3 of the NUTS paper.
        """
        if j == 0:
            q1, p1 = leapfrog(q, p, v * self.stepsize, 1, self._inv_mass,
                              self._pot_energy_func)

            # The log-probability is up to additive constants the negative
            # total energy (or hamiltonian).
            k_p1 = _kin_energy(p1, self._inv_mass)
            u_q1 = self._pot_energy_func(q1)

            log_prob = -u_q1 - k_p1
            log_u = torch.log(u)

            n1 = int(log_u <= log_prob)
            s1 = int(log_prob > log_u - self._delta_max)

            return q1, p1, q1, p1, q1, n1, s1

        else:
            q_m, p_m, q_p, p_p, q1, n1, s1 = self._build_tree(q, p, u, v, j-1)
            if s1 == 1:
                if v < 0:
                    q_m, p_m, _, _, q2, n2, s2 = self._build_tree(q_m, p_m, u,
                                                                  v, j-1)
                else:
                    _, _, q_p, p_p, q2, n2, s2 = self._build_tree(q_p, p_p, u,
                                                                  v, j-1)

                n = n1 + n2
                if n > 0: # This step is a bit ambiguous in the pseudo-code!
                    if torch.rand(1) <= n2 / n:
                        q1 = q2
                s1 = s2 * self._u_turn(q_m, q_p, p_m) * \
                    self._u_turn(q_m, q_p, p_p)
                n1 = n1 + n2

            return q_m, p_m, q_p, p_p, q1, n1, s1

    def simulate_chain(self, n):
        """Simulate the next ``n`` states of the chain.

        The new states will be appended to attribute
        :attr:`position_trajectory`.

        Args:
            n (int): Number of HMC steps to be executed.
        """
        logger = self._logger
        writer = self._writer

        device = self.current_position.device

        for _ in range(n):
            curr_q = self.current_position
            # Resample momentum.
            curr_p = self._momentum_dist.sample()
            # Sample slice variable.
            curr_K = _kin_energy(curr_p, self._inv_mass)
            curr_U = self._pot_energy_func(curr_q)
            curr_prob = torch.exp(-curr_U - curr_K) # unnormalized probability
            u = torch.rand(1).to(device) * curr_prob

            # Initialize some variables.
            q_minus = curr_q
            p_minus = curr_p

            q_plus = curr_q
            p_plus = curr_p

            # The new state.
            q_new = None

            j = 0
            n = 1
            s = 1

            while s == 1:
                # Choose random direction.
                v = -1 if torch.rand(1) < .5 else 1

                if v == -1:
                    q_minus, p_minus, _, _, q_proposed, n_new, s_new = \
                        self._build_tree(q_minus, p_minus, u, v, j)
                else:
                    _, _, q_plus, p_plus, q_proposed, n_new, s_new = \
                        self._build_tree(q_plus, p_plus, u, v, j)

                if s_new == 1:
                    if torch.rand(1) < min(1, n_new/n):
                        q_new = q_proposed

                n += n_new
                s = s_new * self._u_turn(q_minus, q_plus, p_minus) * \
                    self._u_turn(q_minus, q_plus, p_plus)
                j += 1

            if q_new is not None:
                self._accumulated_accept += 1
                self._positions.append(q_new.detach().clone())

                new_U = self._pot_energy_func(q_new)

            else:
                self._positions.append(curr_q.detach().clone())

                new_U = curr_U

            if not self._positions[-1].requires_grad:
                self._positions[-1].requires_grad = True

            new_U = new_U.detach().cpu().numpy()

            self._position = self._positions[-1]
            self._num_states += 1

            # Log progress.
            if logger is not None and \
                    (self.num_states-1) % self._log_interval == 0:
                logger.debug('NUTS state %d: Current Potential Energy: %f - ' \
                    % (self.num_states, new_U) + \
                    'Acceptance probability: %.2f%%.' \
                    % (self.acceptance_probability * 100))

            if writer is not None:
                tag = self._writer_tag
                writer.add_scalar('%snuts/potential' % tag, new_U,
                                  global_step=self.num_states,
                                  display_name='Potential Energy')
                writer.add_scalar('%snuts/accept' % tag,
                                  self.acceptance_probability,
                                  global_step=self.num_states,
                                  display_name='Acceptance Probability')
                writer.add_scalar('%snuts/tree_depth' % tag, j,
                                  global_step=self.num_states,
                                  display_name='Tree Depth')

class MultiChainHMC():
    r"""Wrapper for running multiple HMC chains in parallel.

    Samples obtained via an MCMC sampler are highly auto-correlated for two
    reasons: (1) the proposal distribution is conditioned on the previous state
    and (2) because of rejection (consecutive states are identical). In
    addition, it is unclear when the chain is long enough such that sufficient
    exploration has been taking place and the sample (excluding initial burn-in)
    can be considered an i.i.d. sample from the target distribution. For this
    reason, it is recommended to obtain an MCMC sample by running multiple
    chains in parrallel, starting from varying initial postitions :math:`q(0)`.

    This class provides a simple wrapper to instantiate multiple chains from
    :class:`HMC` (and its subclasses) and provides an interface to easily
    simulate those chains.

    Args:
        initial_positions (list or tuple): A list of initial positions. The
            length of this list will determine the number of chains to be
            instantiated. Each element is an initial position as described for
            argument ``initial_position`` of class :class:`HMC`.
        pot_energy_func (func): See docstring of class :class:`HMC`. One may
            also provide a list of functions. For instance, if the potential
            energy of a Bayesian neural network should be computed, there might
            be a runtime speedup if each function uses separate model instance.
        chain_type (str): The of HMC algorithm to be used. The following options
            are available:

            - ``'hmc'``: Each chain will be an instance of class :class:`HMC`.
            - ``'nuts'``: Each chain will be an instance of class :class:`NUTS`.
        **kwargs: Keyword arguments that will be passed to the constructor when
            instantiating each chain. The following particularities should be
            noted.

            - If a ``writer`` object is passed, then a chain-specific identifier
              is added to the corresponding ``writer_tag``, except if ``writer``
              is a string. In this case, we assume ``writer`` corresponds to an
              output directory and we construct a separate object of class
              :class:`tensorboardX.SummaryWriter` per chain. In the latter case,
              the scalars logged across chains are all shown within the same
              tensorboard plot and are therefore easier comparable.
            - If a ``logger`` object is passed, then it will only be provided
              to the first chain. If a logger should be passed to multiple
              chain instances, then a list of objects from class
              :class:`logging.Logger` is required. If entries in this list are
              ``None``, then a simple console logger is generated for these
              entries that displays the chain's identity when logging a message.
    """
    def __init__(self, initial_positions, pot_energy_func, chain_type='hmc',
                 **kwargs):
        assert chain_type in ['hmc', 'nuts']

        self._num_chains = len(initial_positions)

        # Determine `writer` and `logger` per chain.
        writers = None
        if 'writer' in kwargs.keys():
            writers = kwargs['writer']
        writer_tags = None
        writer_tag = ''
        if 'writer_tag' in kwargs.keys():
            writer_tag = kwargs['writer_tag']
        loggers = None
        if 'logger' in kwargs.keys():
            loggers = kwargs['logger']

        self._close_writers = False
        if writers is not None:
            if isinstance(writers, SummaryWriter):
                # Same writer for all chains, but different tags.
                writers = [writers] * self.num_chains
                writer_tags = [writer_tag + 'chain_%d/' % ii for ii in \
                               range(self.num_chains)]
            else:
                assert isinstance(writers, str)
                summary_dir = writers
                writers = [SummaryWriter(logdir=path.join(summary_dir, \
                                                          'chain_%d' % ii)) \
                           for ii in range(self.num_chains)]
                # Since we created these writer objects, we should explicitly
                # close them when the object is destroyed.
                self._close_writers = True
                self._writers = writers

        if loggers is not None:
            if isinstance(loggers, logging.Logger):
                logger = loggers
                loggers = [None] * self.num_chains
                loggers[0] = logger
            else:
                for ii, logger in enumerate(loggers):
                    if logger is None:
                        loggers[ii] = \
                            MultiChainHMC._get_chain_specific_logger(ii)

        pot_energy_funcs = pot_energy_func
        if not isinstance(pot_energy_func, (list, tuple)):
            pot_energy_funcs = [pot_energy_func] * self.num_chains

        # Instantiate chains.
        self._chains = []
        for cidx in range(self.num_chains):
            if writers is not None:
                kwargs['writer'] = writers[cidx]
            if writer_tags is not None:
                kwargs['writer_tag'] = writer_tags[cidx]
            if loggers is not None:
                kwargs['logger'] = loggers[cidx]

            pe_func = pot_energy_funcs[cidx]

            chain = None
            if chain_type == 'hmc':
                chain = HMC(initial_positions[cidx], pe_func, **kwargs)
            elif chain_type == 'nuts':
                chain = NUTS(initial_positions[cidx], pe_func, **kwargs)

            self._chains.append(chain)

    @property
    def num_chains(self):
        """The number of chains managed by this instance.

        :type: int
        """
        return self._num_chains

    @property
    def chains(self):
        """The list of internally managed HMC objects.

        :type: list
        """
        return self._chains

    @property
    def avg_acceptance_probability(self):
        """The average fraction of states that have been accepted across all
        chains.

        :type: float
        """
        avg_ap = np.mean([c.acceptance_probability for c in self.chains])
        return float(avg_ap)

    def simulate_chains(self, num_states, num_chains=-1, num_parallel=1):
        """Simulate the chains to gather a certain number of new positions.

        This method simulates the internal chains to add ``num_states``
        positions to each considered chain.

        Args:
            num_states (int): Each considered chain will be simulated for
                this amount of HMC steps (see argument ``n`` of method
                :math:`HMC.simulate_chain`).
            num_chains (int or list): The number of chains to be considered. If
                ``-1``, then all chains will be simulated for ``num_states``
                steps. Otherwise, the ``num_chains`` chains with the lowest
                number of states so far (according to attribute
                :attr:`HMC.num_states`) is simulated. Alternatively, one may
                specify a list of chain indices (numbers between 0 and
                :attr:`num_chains`).
            num_parallel (int): How many chains should be simulated in parallel.
                If ``1``, the chains are simulated consecutively (one after
                another).
        """
        if num_chains == -1:
            chain_ids = list(range(self.num_chains))
        elif isinstance(num_chains, (list, tuple)):
            chain_ids = num_chains

            if len(chain_ids) != len(set(chain_ids)):
                warn('Duplicates found in argument "num_chains".')
                chain_ids = list(set(chain_ids))
        else:
            chain_lengths = [(i, o.num_states) \
                             for i, o in enumerate(self.chains)]
            chain_lengths.sort(key=lambda tup: tup[1], reverse=False)
            num_chains = min(num_chains, self.num_chains)
            chain_ids = [t[0] for t in chain_lengths]

        # Put all chains to be simulated in a queue.
        q = Queue()
        for i in chain_ids:
            q.put(self.chains[i])

        def worker(): # Code will be executed in separate thread.
            while not q.empty():
                hmc_obj = q.get()
                hmc_obj.simulate_chain(num_states)
                q.task_done()

        # Create threads that simulate the chains in parallel.
        threads = []
        for i in range(num_parallel):
            t = Thread(target=worker)
            t.start()
            threads.append(t)

        # Block until all queue items are marked as done.
        q.join()

        for t in threads:
            t.join()

    def __del__(self): # Destructor
        if self._close_writers:
            for writer in self._writers:
                writer.close()

    @staticmethod
    def _get_chain_specific_logger(chain_id):
        """Create a chain-specific logger instance.

        Args:
            chain_id (int): The chain identifier.

        Returns:
            (logging.Logger)
        """
        stream_formatter = logging.Formatter( \
            fmt='%(asctime)s - %(levelname)s' + ' - chain %d' % chain_id \
                + ' - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(stream_formatter)
        #stream_handler.setLevel(logging.DEBUG)

        logger = logging.getLogger('logger_chain%d' % chain_id)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)

        return logger

class MCMC:
    r"""Implementation of the Metropolis-Hastings algorithm.

    This class implements the basic Metropolis-Hastings algorithm as, for
    instance, outlined `here <https://arxiv.org/abs/1504.01896>`__ (see alg. 1).

    The Metropolis-Hastings algorithm is a simple MCMC algorithm. In contrast
    to :class:`HMC`, sampling is slow as positions follow a random walk.
    However, the algorithm does not need access to gradient information, which
    makes it applicable to a wider range of applications.

    We use a normal distribution :math:`\mathcal{N}(p, \sigma^2 I)` as proposal,
    where :math:`p` denotes the previous position (sample point). Thus, the
    proposal is symmetric, and cancels in the MH steps.

    The potential energy is expected to be passed as negative log-probability
    (up to a constant), such that

    .. math::
        \frac{\pi(\tilde{p}_t)}{\pi(p_{t-1})} \propto \
            \exp \big\{ U(p_{t-1}) - U(\tilde{p}_t) \big\}

    Args:
        (....): See docstring of class :class:`HMC`.
        proposal_std (float): The standard deviation :math:`\sigma` of the
            proposal distribution :math:`\tilde{p}_t \sim q(p \mid p_{t-1})`.
    """
    def __init__(self, initial_position, pot_energy_func, proposal_std=1.,
                 logger=None, log_interval=100, writer=None, writer_tag=''):
        self._position = initial_position
        if not self._position.requires_grad:
            self._position.requires_grad = True
        self._proposal_std = proposal_std
        self._pot_energy_func = pot_energy_func
        self._logger = logger
        self._log_interval = log_interval
        self._writer = writer
        self._writer_tag = writer_tag

        self._positions = [initial_position]
        self._num_states = 0
        self._accumulated_accept = 0

    @property
    def proposal_std(self):
        """The std :math:`\sigma` of the proposal distribution.

        :type: float
        """
        return self._proposal_std

    @proposal_std.setter
    def proposal_std(self, value):
        self._proposal_std = value

    @property
    def current_position(self):
        """The latest position :math:`q(t)` in the chain simulated so far.

        :type: torch.Tensor
        """
        return self._position

    @property
    def num_states(self):
        """The number of states in the chain visited so far.

        The counter will be increased by method :meth:`simulate_chain`.

        :type: int
        """
        return self._num_states

    @property
    def position_trajectory(self):
        """A list containing all position variables (Markov states) visited so
        far.

        New positions will be added by the method :meth:`simulate_chain`. To
        decrease the memory footprint of objects in this class, the trajectory
        can be cleared via method :meth:`clear_position_trajectory`.

        :type: list
        """
        return self._positions

    @property
    def acceptance_probability(self):
        """The fraction of states that have been accepted.

        :type: float
        """
        if self.num_states == 0:
            return 1.0

        return self._accumulated_accept / self._num_states

    def clear_position_trajectory(self, n=None):
        """Reset attribute :attr:`position_trajectory`.

        This method will no affect the counter :attr:`num_states`.

        Args:
            n (int, optional): If provided, only the first ``n`` elements of
                :attr:`position_trajectory` are discarded (e.g., the burn-in
                samples).
        """
        if n is not None:
            self._positions = self._positions[n:]
        else:
            self._positions = []

    def simulate_chain(self, n):
        """Simulate the next ``n`` states of the chain.

        The new states will be appended to attribute
        :attr:`position_trajectory`.

        Args:
            n (int): Number of MCMC steps to be executed.
        """
        logger = self._logger
        writer = self._writer

        for _ in range(n):
            curr_q = self.current_position

            # Sample new proposal.
            eps = torch.normal(torch.zeros_like(curr_q), 1)
            q = curr_q + self.proposal_std * eps

            # Evaluate Hamiltonian at beginning and end of trajectory.
            u_q_start = self._pot_energy_func(curr_q)

            u_q_proposal = self._pot_energy_func(q)

            # Metropolis update.
            if torch.rand(1).to(q.device) < min(1, torch.exp(u_q_start - \
                                                             u_q_proposal)):
                accept = True
                self._accumulated_accept += 1

                self._positions.append(q)
            else: # Reject
                accept = False
                self._positions.append(curr_q.clone())
            self._position = self._positions[-1]

            self._num_states += 1

            # Log progress.
            if accept:
                potential = u_q_proposal.detach().cpu().numpy()
            else:
                potential = u_q_start.detach().cpu().numpy()

            if logger is not None and \
                    (self.num_states-1) % self._log_interval == 0:
                logger.debug('MH state %d: Current Pot. Energy: %f - ' \
                    % (self.num_states, potential) + \
                    'Acceptance probability: %.2f%%.' \
                    % (self.acceptance_probability * 100))

            if writer is not None:
                tag = self._writer_tag
                writer.add_scalar('%smh/potential' % tag, potential,
                                  global_step=self.num_states,
                                  display_name='Potential Energy')
                writer.add_scalar('%smh/accept' % tag,
                                  self.acceptance_probability,
                                  global_step=self.num_states,
                                  display_name='Acceptance Probability')

if __name__ == '__main__':
    pass
