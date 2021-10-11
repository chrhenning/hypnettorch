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
# @title          :data/timeseries/rnd_rec_teacher.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :07/16/2020
# @version        :1.0
# @python_version :3.6.10
r"""
Dataset from random recurrent teacher networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We consider a student-teacher setup. The dataset is meant for continual
learning, such that an individual teacher (individual task) is used to determine
the computation of a subspace of the activations of a recurrent student network.

This is a synthetic dataset that will allow the manual construction of the
optimal student network that solves all tasks simultanously. As such, this
student network can be compared to trained networks (either continually or in
parallel on multiple tasks).

To be more precise, we set the teacher to be an Elman-type recurrent network
(see :class:`mnets.simple_rnn.SimpleRNN`):

.. math::

    r_t^{(k)} = \sigma (A^{(k)} r_{t-1}^{(k)} + x_t) \\
    s_t^{(k)} = \sigma (B^{(k)} r_t^{(k)}) \\
    t_t^{(k)} = C^{(k)} s_t^{(k)}

Where :math:`k` is a unique task identifier (in the context of multiple
teachers), :math:`x_t` is the network input at time :math:`t`, the recurrent
state is initialized at zero :math:`r_0^{(k)} = 0` and :math:`\sigma()` is a
user-defined non-linearity. The non-linear output computation :math:`s_t^{(k)}`
is optional.

We assume an input :math:`x_t \in \mathbb{R}^{n_\text{in}}` and a target
dimensionality of :math:`n_\text{out}`.
:math:`A^{(k)} \in \mathbb{R}^{n_\text{in} \times n_\text{in}}`,
:math:`B^{(k)} \in \mathbb{R}^{n_\text{in} \times n_\text{in}}` and 
:math:`C^{(k)} \in \mathbb{R}^{n_\text{out} \times n_\text{in}}` are random
matrices that determine the teacher network's input-output mapping.

Having a task setup like this one can manually construct an RNN network that
can solve multiple of such tasks to perfection (assuming a task-specific output
head). For instance, consider the following Elman-type RNN with task-specific
output head.

.. math::

    h_t = \sigma (W_{hh} h_{t-1} + W_{ih} x_t + b_h) \\
    o_t = \sigma (W_{ho} h_t + b_o) \\
    y_t^{(k)} = W^{(k)} o_t + b^{(k)}

With :math:`h_t \in \mathbb{R}^{n_\text{h}}` being the hidden state (we also
assume :math:`o_t \in \mathbb{R}^{n_\text{h}}`).

We can assign this network the following weights to ensure that all :math:`K`
tasks are solved to perfection:

- :math:`b_h, b_o, b^{(k)} = 0`
- :math:`W_{ih} = \begin{pmatrix} I \\ \vdots \\ I \\ O \end{pmatrix}` where
  :math:`I \in \mathbb{R}^{n_\text{in} \times n_\text{in}}` refers to the
  identity matrix that simply copies the input into separate subspaces of the
  hidden state
- The hidden-to-hidden weights would be block diagonal:

  .. math::

    W_{hh} = \begin{pmatrix}
        A^{(1)} & & & \\
        & \ddots & & \\
        & & A^{(K)} & \\
        & & & O
    \end{pmatrix}
- The hidden-to-output weights would be block diagonal:

  .. math::

    W_{ho} = \begin{pmatrix}
        B^{(1)} & & & \\
        & \ddots & & \\
        & & B^{(K)} & \\
        & & & O
    \end{pmatrix}
- The task-specific output matrix would be

  .. math::

      W^{(k)} = \begin{pmatrix}
          O & \hdots & C^{(k)} & \hdots & O
      \end{pmatrix}
"""
import numpy as np
from warnings import warn
from scipy.stats import ortho_group
import torch

from hypnettorch.data.sequential_dataset import SequentialDataset
from hypnettorch.mnets.simple_rnn import SimpleRNN

class RndRecTeacher(SequentialDataset):
    r"""Create a dataset from a random recurrent teacher.

    Args:
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        num_val (int, optional): Number of validation samples.
        n_in (int): Dimensionality of inputs :math:`x_t`.
        n_out (int): Dimensionality of outputs :math:`y_t^{(k)}`.
        sigma (str): Name of the nonlinearity :math:`\sigma()` to be used.

            - ``'linear'``
            - ``'sigmoid'``
            - ``'tanh'``
        mat_A (numpy.ndarray, optional): A numpy array of shape
            ``[n_in, n_in]`` representing matrix :math:`A^{(k)}`. If not
            specified, a random matrix will be generated.
        mat_B (numpy.ndarray, optional): A numpy array of shape
            ``[n_in, n_in]`` representing matrix :math:`B^{(k)}`. If not
            specified, a random matrix will be generated.
        mat_C (numpy.ndarray, optional): A numpy array of shape
            ``[n_out, n_in]`` representing matrix :math:`C^{(k)}`. If not
            specified, a random matrix will be generated.
        orth_A (bool): If :math:`A^{(k)}` is randomly generated and this option
            is activated, then :math:`A^{(k)}` will be initialized as an
            orthogonal matrix.
        rank_A (int, optional): The rank of the randomly generated matrix
            :math:`A^{(k)}`. Note, this option is mutually exclusive with
            option ``orth_A``.
        max_sv_A (float, optional): The maximum singular value of the randomly
            generated matrix :math:`A^{(k)}`. Note, this option is mutually
            exclusive with option ``orth_A``.
        no_extra_fc: If ``True``, the hidden fully-connected layer using matrix
            :math:`B^{(k)}` will be omitted when computed targets from the
            teacher. Hence, the teacher computation becomes:

            .. math::

                r_t^{(k)} = \sigma (A^{(k)} r_{t-1}^{(k)} + x_t) \\
                t_t^{(k)} = C^{(k)} r_t^{(k)}
        inputs (numpy.ndarray, optional): The inputs :math:`x_t` to be used.
            Has to be an array of shape ``[n_ts_in, N, n_in]`` with
            ``N = num_train + num_test + (0 if num_val is None else num_val)``.
        input_range (tuple): Tuple of integers. Used as ranges for a uniform
            distribution from which input samples :math:`x_t` are drawn.
        n_ts_in (int): The number of input timesteps.
        n_ts_out (int, optional): The number of output timesteps. Can be greater
            than ``n_ts_in``. In this case, the inputs at time greater than
            ``n_ts_in`` will be zero.
        rseed (int, optional): If ``None``, the current random state of numpy
            is used to generate the data. Otherwise, a new random state with the
            given seed is generated.
    """
    def __init__(self, num_train=1000, num_test=100, num_val=None, n_in=7,
                 n_out=7, sigma='tanh', mat_A=None, mat_B=None, mat_C=None,
                 orth_A=False, rank_A=-1, max_sv_A=-1., no_extra_fc=False,
                 inputs=None, input_range=(-1, 1), n_ts_in=10, n_ts_out=-1,
                 rseed=None):
        super().__init__()

        if rseed is not None:
            self._rstate = np.random.RandomState(rseed)
        else:
            self._rstate = np.random

        self._n_in = n_in
        self._n_out = n_out

        if sigma == 'linear':
            self._sigma_fct = lambda x : x
        elif sigma == 'sigmoid':
            self._sigma_fct = lambda x : 1. / (1. / + np.exp(-x))
        elif sigma == 'tanh':
            self._sigma_fct = lambda x : np.tanh(x)
        else:
            raise ValueError('Unknown value "%s" for argument "sigma".' % sigma)

        ### Create matrices ###
        self._mat_A, self._mat_B, self._mat_C = \
            self._generate_teacher_matrices(n_in, n_out, mat_A, mat_B, mat_C,
                                            orth_A, rank_A, max_sv_A)
        if no_extra_fc:
            self._mat_B = None

        ### Create inputs ###
        num_samples = num_train + num_test + (0 if num_val is None else num_val)
        if inputs is None:
            inputs = self._rstate.uniform(low=input_range[0],
                high=input_range[1], size=(n_ts_in, num_samples, n_in))
        else:
            assert np.all(np.equal(inputs.shape, [n_ts_in, num_samples, n_in]))

        assert n_ts_out == -1 or n_ts_out >= n_ts_in
        if n_ts_out > n_ts_in:
            inputs = np.concatenate(inputs,
                np.zeros((n_ts_out-n_ts_in, num_samples, n_in)), axis=0)

        ### Compute targets ###
        targets = np.empty([inputs.shape[0], num_samples, n_out])
        r_t = np.zeros((num_samples, n_in)) # r_0
        for t in range(inputs.shape[0]):
            x_t = inputs[t, :, :]
            r_t = self._sigma_fct(self._mat_A @ r_t.T +  x_t.T).T
            if no_extra_fc:
                s_t = r_t
            else:
                s_t = self._sigma_fct(self._mat_B @ r_t.T).T
            t_t = (self._mat_C @ s_t.T).T

            targets[t, :,:] = t_t

        ### Setup dataset ###
        self._data['classification'] = False
        self._data['is_one_hot'] = False
        self._data['sequence'] = True
        self._data['in_shape'] = [n_in]
        self._data['out_shape'] = [n_out]
        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train, num_train + num_test)
        if num_val is not None:
            n_start = num_train + num_test
            self._data['val_inds'] = np.arange(n_start, n_start + num_val)
        self._data['in_data'] = self._flatten_array(inputs, ts_dim_first=True)
        self._data['out_data'] = self._flatten_array(targets, ts_dim_first=True)
        # Note, inputs and outputs have internally the same length, even though
        # inputs might be zero-padded.
        self._data['in_seq_lengths'] = np.ones(num_samples) * n_ts_in
        self._data['out_seq_lengths'] = np.ones(num_samples) * n_ts_out

        print('Constructed: ' + str(self))

    def _generate_teacher_matrices(self, n_in, n_out, mat_A, mat_B, mat_C,
                                   orth_A, rank_A, max_sv_A):
        """Generate matrices :math:`A^{(k)}`, :math:`B^{(k)}` and
        :math:`C^{(k)}`.

        Args:
            (....): See constructor.

        Returns:
            (tuple): Tuple containing :math:`A^{(k)}`, :math:`B^{(k)}` and
            :math:`C^{(k)}`.
        """
        def get_rand_matrix(n_r, n_c):
            """Create random matrix similar to Kaiming init."""
            std = 1. / np.sqrt(n_c)
            bound = np.sqrt(3.0) * std
            return self._rstate.uniform(low=-bound, high=bound, size=(n_r, n_c))

        if mat_A is None:
            if orth_A: # Random orthogonal matrix.
                if rank_A != -1 or max_sv_A != -1.:
                    raise ValueError('Options "rank_A" and "max_sv_A" are ' +
                                     'mutually exclusive with option "orth_A".')
                mat_A = ortho_group.rvs(n_in, random_state=self._rstate)
            elif rank_A != -1: # Random matrix with specified rank.
                assert rank_A > 0 and rank_A <= n_in
                # Consider two matrices X, Y with Y having full rank. It then
                # holds that:
                #   rank(XY) = rank(YX) = rank(X)
                # Hence, for three matrices X, Y, Z with X and Z having full
                # rank we can state that
                #   rank(XYZ) = rank(Y)
                # We make use of this fact by choosing X and Z to be two
                # random invertible (full rank) matrices while Y contains the
                # identity matrix in the first `rank_A` rows while the remaining
                # rows are zero.
                # Note, random matrices are with high probability invertible.
                # Note, if X, Y, Z are constructed as decribed above, then only
                # the first `rank_A` columns of X and the first `rank_A` rows of
                # Z enter the computation.

                #X = get_rand_matrix(n_in, rank_A)
                #Z = get_rand_matrix(rank_A, n_in)
                #Y = np.identity(rank_A)
                # The commented implementation above is more compuationally
                # efficient, but it will lead to different elements in the
                # columns/rows of X/Z everytime the constructor is called with
                # a different `rank_A` while using the same random seed.
                # This behavior might be undesired.
                X = get_rand_matrix(n_in, n_in)
                Z = get_rand_matrix(n_in, n_in)
                Y = np.identity(n_in)
                Y[rank_A:, :] = 0

                mat_A = X @ Y @ Z

                actual_rank = np.linalg.matrix_rank(mat_A)
                if actual_rank != rank_A:
                    warn('Could not create matrix with desired rank %d ' % \
                         rank_A + '(actual rank is %d).' % actual_rank)

                # The product of two uniform variables (elements of matrix X and
                # Z) is not uniformly distributed anymore. However, we at least
                # want to normalize the variance. Note, the variance of the
                # product two independent random variables with zero mean
                # is simply `Var(x) * Var(z)` (note, x and z have the same
                # variance in our case). Hence, also the std is just squared.
                # We can therefore ensure that `mat_A` has the same variance
                # as `X` or `Z` by dividing by X/Z's std.
                mat_A /= 1. / np.sqrt(n_in)

            else: # Random matrix.
                mat_A = get_rand_matrix(n_in, n_in)

            if max_sv_A != -1:
                assert max_sv_A > 0
                curr_max_sv = np.linalg.svd(mat_A, compute_uv=0)[0]
                mat_A = mat_A / curr_max_sv * max_sv_A
        else:
            assert np.all(np.equal(mat_A.shape, [n_in, n_in]))

        if mat_B is None:
            mat_B = get_rand_matrix(n_in, n_in)
        else:
            assert np.all(np.equal(mat_B.shape, [n_in, n_in]))

        if mat_C is None:
            mat_C = get_rand_matrix(n_out, n_in)
        else:
            assert np.all(np.equal(mat_C.shape, [n_out, n_in]))

        return mat_A, mat_B, mat_C

    @property
    def mat_A(self):
        r"""The teacher matrix :math:`A^{(k)}`.
        
        :type: numpy.ndarray
        """
        return self._mat_A

    @property
    def mat_B(self):
        r"""The teacher matrix :math:`B^{(k)}`.
        
        :type: numpy.ndarray
        """
        return self._mat_B

    @property
    def mat_C(self):
        r"""The teacher matrix :math:`C^{(k)}`.
        
        :type: numpy.ndarray
        """
        return self._mat_C

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Add a custom sample plot to the given Axes object.
        Note, this method is called by the :meth:`plot_samples` method.

        Note, that the number of inner subplots is configured via the method:
        :meth:`_plot_config`.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset._plot_sample`.
        """
        raise NotImplementedError()

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'Random Recurrent Teacher Dataset'

    @staticmethod
    def construct_ideal_student(net, dhandlers):
        """Set the weights of an RNN such that it perfectly solves all tasks
        represented by the teachers in ``dhandlers``.

        Note:
            This method only works properly if the RNN ``net`` is properly setup
            such that its computation resembles the target computation of the
            individual teachers. I.e., an ideal student can be constructed by
            only modifying the weights.

        Args:
            net (mnets.simple_rnn.SimpleRNN): The student RNN whose weights will
                be overwritten. Importantly, this method does not ensure that
                the teacher computation is compatible with the given student
                network.

                Note:
                    The internal weights of the network are modified in-place.
            dhandlers (list): List of datasets from teachers (i.e., instances
                of class :class:`RndRecTeacher`). The RNN ``net`` must have at
                least as many output heads as ``len(dhandlers)``.
        """
        assert isinstance(net, SimpleRNN)
        n_in = dhandlers[0].in_shape[0]
        for dh in dhandlers:
            assert isinstance(dh, RndRecTeacher)
            assert dh.in_shape[0] == n_in
            if dh.mat_B is None:
                raise NotImplementedError()

        assert not net.use_lstm and net.num_rec_layers == 1 and net.has_bias
        assert len(net.param_shapes) == 8 and len(net.internal_params) == 8
        # The 8 entries (always consecutive tuples of matrix and bias vectors)
        # of `param_shapes` are as follows:
        # - input-to-hidden
        # - hidden-to-hidden
        # - hidden-to-output
        # - output head

        for i in range(8):
            net.internal_params[i].data.zero_()

        n_in_start = 0
        n_out_start = 0
        for dh in dhandlers:
            n_in = dh.in_shape[0]
            n_out = dh.out_shape[0]

            assert net.param_shapes[0][1] == n_in
            net.internal_params[0].data[n_in_start:n_in_start+n_in,:] = \
                torch.eye(n_in)
            net.internal_params[2].data[n_in_start:n_in_start+n_in, \
                n_in_start:n_in_start+n_in] = torch.from_numpy(dh.mat_A).float()
            net.internal_params[4].data[n_in_start:n_in_start+n_in, \
                n_in_start:n_in_start+n_in] = torch.from_numpy(dh.mat_B).float()
            # Output head.
            net.internal_params[6].data[n_out_start:n_out_start+n_out, \
                n_in_start:n_in_start+n_in] = torch.from_numpy(dh.mat_C).float()

            n_in_start += n_in
            n_out_start += n_out

if __name__ == '__main__':
    pass
