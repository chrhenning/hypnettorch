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
# @title           :data/special/gmm_data.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :08/06/2019
# @version         :1.0
# @python_version  :3.6.8
r"""
Gaussian Mixture Model Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.special.gaussian_mixture_data` is stemming from a
conditional view, where every mode in the Gaussian mixture is a separate task
(single dataset). Therefore, it provides ``N`` distinct data handlers when
having ``N`` distinct modes.

Unfortunately, this configuration is not ideal for unsupervised GAN training (as
we want to be able to provide batches that contain data from a mix of modes
without having to manually assemble these batches) or for training a classifier
for a GMM toy problem.

Therefore, this module provides a wrapper that converts a sequence of data
handlers of class :class:`data.special.gaussian_mixture_data.GaussianData`
(i.e., a set of single modes) to a combined data handler.

**Model description:**

Let :math:`x` denote the input data. The class :class:`GMMData` assumes that
it's input training data is drawn from the following Gaussian Mixture Model:

.. math::

    p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)

with mixing coefficients :math:`\pi_k`, such that :math:`\sum_k \pi_k = 1`.

Note, it is up to the user of this class to provide appropriate training data
(only important to keep in mind if unequal train set sizes are provided via
constructor argument ``gaussian_datasets`` or if ``mixing_coefficients`` are
non-uniform).

Let :math:`y` denote a :math:`K`-dimensional 1-hot encoding, i.e.,
:math:`y_k \in \{0, 1\}` and :math:`\sum_k y_k = 1`. Thus, :math:`y` is the
latent variable that we want to infer (e.g., the optimal classification label)
with marginal probabilities:

.. math::

    p(y_k=1) = \pi_k

The conditional likelihood of a component is:

.. math::

    p(x \mid y_k=1) = \mathcal{N}(x; \mu_k, \Sigma_k)

Using Bayes Theorem we obtain the posterior:

.. math::

    p(y_k=1 \mid x) &= \frac{p(x \mid y_k=1) p(y_k=1)}{p(x)} \\ \
        &= \frac{\pi_k \mathcal{N}(x; \mu_k, \Sigma_k)}{ \
            \sum_{l=1}^K \pi_l \mathcal{N}(x; \mu_l, \Sigma_l)}
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
from warnings import warn

from hypnettorch.data.dataset import Dataset
from hypnettorch.utils import misc


class GMMData(Dataset):
    r"""Dataset with inputs drawn from a Gaussian mixture model.

    An instance of this class combines several instances of class
    :class:`data.special.gaussian_mixture_data.GaussianData` into one data
    handler. I.e., multiple gaussian bumps are combined to a Gaussian mixture
    dataset.

    Most importantly, the dataset can be turned into a classification task,
    where the label corresponds to the ID of the Gaussian bump from which the
    sample was drawn. Otherwise, the original outputs will remain.

    Note:
        You can use function
        :func:`data.special.gaussian_mixture_data.get_gmm_tasks` to create a set
        of tasks to be passed as constructor argument ``gaussian_datasets``.

    Args:
        gaussian_datasets (list): A list of instances of class
            :class:`data.special.gaussian_mixture_data.GaussianData`.
        classification (bool): If ``True``, the original outputs of the datasets
            will be omitted and replaced by the dataset index. Therefore, the
            original regression datasets are combined to a single classification
            dataset.
        use_one_hot (bool): Whether the class labels should be represented as a
            one-hot encoding. This option only applies if ``classification`` is
            ``True``.
        mixing_coefficients (list, optional): The mixing coefficients
            :math:`\pi_k` of the individual mixture components. If not
            specified, :math:`\pi_k` will be assumed to be
            ``1. / self.num_modes``.

            .. math::

                p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)

            Note:
                Mixing coefficients have to sum to ``1``.

            Note:
                If mixing coefficients are not uniform, then one has to
                externally ensure that the training data is distributed
                accordingly. For instance, if ``mixing_coefficients=[.1, .9]``,
                then the second dataset passed via ``gaussian_datasets`` should
                have 9 times more training samples then the first dataset.
    """
    def __init__(self, gaussian_datasets, classification=False,
                 use_one_hot=False, mixing_coefficients=None):
        super().__init__()

        if use_one_hot and not classification:
            raise ValueError('Outputs can only be 1-hot encoded for '
                             'classification datasets.')

        num_modes = len(gaussian_datasets)
        self._num_modes = num_modes
        self._means = []
        self._covs = []
        if mixing_coefficients is None:
            mixing_coefficients = [1. / self.num_modes] * num_modes
        else:
            assert len(mixing_coefficients) == num_modes and \
                np.isclose(1., np.sum(mixing_coefficients))
            
        self._mixing_coefficients = mixing_coefficients

        num_train = 0
        num_test = 0
        for i, d in enumerate(gaussian_datasets):
            self._means.append(d.mean)
            self._covs.append(d.cov)

            num_train += d.num_train_samples
            num_test += d.num_test_samples

            if i == 0:
                train_x = d.get_train_inputs()
                test_x = d.get_test_inputs()

                if classification:
                    train_y = np.ones((d.num_train_samples, 1)) * i
                    test_y = np.ones((d.num_test_samples, 1)) * i
                else:
                    train_y = d.get_train_outputs()
                    test_y = d.get_test_outputs()
            else:
                train_x = np.vstack([train_x, d.get_train_inputs()])
                test_x = np.vstack([test_x, d.get_test_inputs()])

                if classification:
                    train_y = np.vstack([train_y,
                                         np.ones((d.num_train_samples, 1)) * i])
                    test_y = np.vstack([test_y,
                                        np.ones((d.num_test_samples, 1)) * i])
                else:
                    train_y = np.vstack([train_y, d.get_train_outputs()])
                    test_y = np.vstack([test_y, d.get_test_outputs()])

        out_data = np.vstack([train_y, test_y])

        # Specify internal data structure.
        self._data['classification'] = classification
        if classification:
            self._data['num_classes'] = num_modes
            self._data['is_one_hot'] = use_one_hot
        self._data['sequence'] = False
        self._data['in_data'] = np.vstack([train_x, test_x])
        self._data['in_shape'] = gaussian_datasets[0].in_shape
        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train, num_train + num_test)

        if use_one_hot:
            self._data['out_shape'] = [num_modes]
            out_data = self._to_one_hot(out_data)
            assert np.all(np.equal(list(out_data.shape[1:]),
                                   self._data['out_shape']))
        else:
            self._data['out_shape'] = list(out_data.shape[1:])
        self._data['out_data'] = out_data

        if not classification:
            assert np.all(np.equal(list(out_data.shape[1:]),
                                   gaussian_datasets[0].out_shape))

    @property
    def num_modes(self):
        """The number of mixture components.

        :type: int
        """
        return self._num_modes

    @property
    def means(self):
        """2D array, containing the mean of each component in its rows.

        :type: np.ndarray
        """
        return np.vstack(self._means)

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'GMMData'

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Not implemented"""
        # We overwrote the plot_samples method, so there is no need to ever call
        # this method (it's just here because the baseclass requires its
        # existence).
        raise NotImplementedError('TODO implement')

    def plot_samples(self, title, inputs, outputs=None, predictions=None,
                     show=True, filename=None, interactive=False,
                     figsize=(10, 6)):
        """Plot samples belonging to this dataset.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.plot_samples`.
        """
        if inputs.shape[1] != 2:
            raise ValueError('This method is only applicable to 2D input data!')

        plt.figure(figsize=figsize)
        plt.title(title, size=20)
        if interactive:
            plt.ion()

        if self.classification:
            n = self.num_classes
            colors = np.asarray(misc.get_colorbrewer2_colors(family='Dark2'))
            if n > len(colors):
                #warn('Changing to automatic color scheme as we don\'t have ' +
                #     'as many manual colors as tasks.')
                colors = cm.rainbow(np.linspace(0, 1, n))
        else:
            norm = Normalize(vmin=self._data['out_data'].min(),
                             vmax=self._data['out_data'].max())
            cmap = cm.get_cmap(name='viridis')

            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array(np.asarray([norm.vmin, norm.vmax]))
            if outputs is not None or predictions is not None:
                plt.colorbar(sm)

        if outputs is not None and predictions is None:
            plt.scatter(inputs[:, 0], inputs[:, 1], #edgecolors='b',
                label='Targets',
                facecolor=colors[outputs.squeeze().astype(int)] \
                    if self.classification else \
                        cmap(norm(outputs.squeeze()))
                )
        elif predictions is not None and outputs is None:
            plt.scatter(inputs[:, 0], inputs[:, 1], #edgecolors='r',
                label='Predictions',
                facecolor=colors[predictions.squeeze().astype(int)] \
                    if self.classification else \
                        cmap(norm(predictions.squeeze()))
                )
        elif predictions is not None and outputs is not None:
            plt.scatter(inputs[:, 0], inputs[:, 1], label='Targets+Predictions',
                edgecolors=colors[outputs.squeeze().astype(int)] \
                    if self.classification else \
                        cmap(norm(outputs.squeeze())),
                facecolor=colors[predictions.squeeze().astype(int)] \
                    if self.classification else \
                        cmap(norm(predictions.squeeze()))
                )
        else:
            assert predictions is None and outputs is None
            plt.scatter(inputs[:, 0], inputs[:, 1], color='k', label='Inputs')

        #plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        if show:
            plt.show()

    def plot_real_fake(self, title, real, fake, show=True, filename=None,
                       interactive=False, figsize=(10, 6)):
        """Useful method when using this dataset in conjunction with GAN
        training. Plots the given real and fake input samples in a 2D plane.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.plot_samples`.
            real (numpy.ndarray): A 2D numpy array, where each row is an input
                sample. These samples correspond to actual input samples drawn
                from the dataset.
            fake (numpy.ndarray): A 2D numpy array, where each row is an input
                sample. These samples correspond to generated samples.
        """
        if real.shape[1] != 2 or fake.shape[1] != 2:
            raise ValueError('This method is only applicable to 2D input data!')

        plt.figure(figsize=figsize)
        plt.title(title, size=20)
        if interactive:
            plt.ion()

        colors = np.asarray(misc.get_colorbrewer2_colors(family='Dark2'))

        plt.scatter(real[:, 0], real[:, 1], color=colors[0], label='real')
        plt.scatter(fake[:, 0], fake[:, 1], color=colors[1], label='fake')

        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        if show:
            plt.show()

    def _compute_responsibilities(self, samples, normalize=False, eps=None):
        r"""Compute the responsibility :math:`p(y_k=1 \mid x)` of each sample
        towards each mixture component (i.e., for all :math:`k`).

        Args:
            samples: A 2D numpy array, where each row is an input sample.
            normalize: Whether responsibilities should sum to 1.
            eps (float, optional): If specified, will be used during
                normalization to avoid division by zero.

        Returns:
            See argument ``responsibilities`` of method
            :meth:`estimate_mode_coverage`.
        """
        responsibilities = np.empty([samples.shape[0], self.num_modes])
        for m in range(self.num_modes):
            responsibilities[:, m] = self._mixing_coefficients[m] * \
                multivariate_normal.pdf(samples, self._means[m], self._covs[m])
        if normalize:
            if eps is None:
                eps = 0
            responsibilities = responsibilities / \
                (responsibilities.sum(axis=1)[:, None] + eps)

        return responsibilities

    def estimate_mode_coverage(self, fake, responsibilities=None):
        """Compute the mode coverage of fake samples as suggested in

            https://arxiv.org/abs/1606.00704

        This method will compute the responsibilities for each fake sample
        towards each mixture component and assign each sample to the mixture
        component with the highest responsibility. Mixture components that
        get no fake sample assigned are considered dropped modes.

        The paper referenced above used 10,000 fake samples (on their synthetic
        dataset) to measure the mode coverage.

        Args:
            fake: A 2D numpy array, where each row is an input sample (usually
                drawn from a generator network).
            responsibilities (optional): The responsibilities of each `fake`
                data point (may be unnormalized). A 2D numpy array with each
                row corresponding to a sample in `fake` and each column
                corresponding to a mode in this dataset.

        Returns:
            (tuple): A tuple containing:

            - **num_covered**: The number of modes that have at least one fake
              sample with maximum responsibility being assigned to that mode.
            - **responsibilities**: The given or computed `responsibilities`. If
              computed by this method, the responsibilities will be
              unnormalized, i.e., correspond to the densities per component of
              this mixture model.
        """
        if responsibilities is None:
            responsibilities = self._compute_responsibilities(fake,
                                                              normalize=False)
        else:
            assert(responsibilities.shape[0] == fake.shape[0] and \
                   responsibilities.shape[1] == self.num_modes)

        max_inds = responsibilities.argmax(axis=1)

        covered_modes = np.unique(max_inds)

        return covered_modes.size, responsibilities

    def estimate_distance(self, fake, component_densities=None,
                          density_estimation='hist', eps=1e-5):
        r"""This method estimates the distance/divergence of the empirical fake
        distribution with the underlying true data disctribution.

        Therefore, we utilize the fact that we know the data distribution.

        The following distance/divergence measures are implemented:

        - `Symmetric KL divergence`: The fake samples are used to estimate the
          model density. The fake samples are used to estimate
          :math:`D_\text{KL}(\text{fake} \mid\mid \text{real})`. An additional
          set of real samples is drawn from the training data to compute a
          Monte Carlo estimate of
          :math:`D_\text{KL}(\text{real} \mid\mid \text{fake})`.

          Comment from Simone Surace about this approach: "Doing density
          estimation first and then computing the integral is known to be
          the wrong way to go (there is an entire literature about this
          problem)." This should be kept in mind when using this estimate.

        Args:
            fake (numpy.ndarray): A 2D numpy array, where each row is an input
                sample (usually drawn from a generator network).
            component_densities (numpy.ndarray, optional): A 2D numpy array with
                each row corresponding to a sample in ``fake`` and each column
                corresponding to a mode in this dataset. Each entry represents
                the density of the corresponding sample under the corresponding
                mixture component. See return value ``responsibilities`` of
                method :meth:`estimate_mode_coverage`.
            density_estimation: Which kind of method should be used to estimate
                the model distribution (i.e., density of given samples under the
                distribution estimated from those samples).
                Available methods are:

                - ``'hist'``: We estimate the fake density based on a normalized
                  2D histogram of the samples. We use the Square-root choice to
                  compute the number of bins per dimension.
                - ``'gaussian'``: Uses the kernel density method ``'gaussian'``
                  from :class:`sklearn.neighbors.kde.KernelDensity`.
                  Note, we don't change the default ```bandwidth``` value!
            eps (float): We don't allow densities to be smaller than this value
                for numerical stability reasons (when computing the log).

        Returns:
            The estimated symmetric KL divergence.
        """
        # Note, since this method is meant for generative models, we can take
        # the training data to approximate the distance between real and fake
        # distribution.
        n = fake.shape[0]
        m = min(n, self.num_train_samples)

        rand = np.random.RandomState(42)
        real_inds = rand.permutation(self.num_train_samples)[:m]
        real = self.get_train_inputs()[real_inds, :]

        ### Calculate densities under true data distribution.
        def compute_data_densities(samples, single_densities=None):
            if single_densities is None:
                single_densities = self._compute_responsibilities(samples,
                    normalize=False)
            else:
                assert(single_densities.shape[0] == n and \
                       single_densities.shape[1] == self.num_modes)

            densities = single_densities.sum(axis=1) / self.num_modes
            densities[densities < eps] = eps

            # Note, we don't benefit from the fact that `multivariate_normal`
            # has a logpdf function, as we can't translate it to the log-probs
            # of a GMM.
            return densities, np.log(densities)

        # The densities of the fake samples under the true data distribution.
        fake_data_densities, log_fake_data_densities = compute_data_densities( \
            fake, single_densities=component_densities)

        real_data_densities, log_real_data_densities = compute_data_densities( \
            real, single_densities=None)

        ### Estimate model pdf ###
        if density_estimation == 'hist':
            # Sturges' formula
            #https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
            #num_bins = np.ceil(np.log(n)) + 1
            # Square-root choice
            num_bins = np.ceil(np.sqrt(n))

            hist, x1_bins, x2_bins = np.histogram2d(fake[:, 0], fake[:, 1],
                                                    bins=num_bins)
            # Note, method `digitize`, that is used below, would otherwise
            # classify border samples as out of bin samples.
            x1_bins[0] -= 1e-2
            x1_bins[-1] += 1e-2
            x2_bins[0] -= 1e-2
            x2_bins[-1] += 1e-2

            # Compute densities of samples using the estimated model dist.
            def compute_model_densities(samples):
                # Note, we have to subtract 1 to index `hist`, as digitize
                # returns 0 for values smaller than the first bin.
                x1_bin_inds = np.digitize(samples[:, 0], x1_bins) - 1
                x2_bin_inds = np.digitize(samples[:, 1], x2_bins) - 1

                # Take care of any values that are not in our histogram (should
                # not happen for fake samples).
                x1_bin_inds_mask = np.logical_or(x1_bin_inds == -1,
                                                 x1_bin_inds == len(x1_bins)-1)
                x2_bin_inds_mask = np.logical_or(x2_bin_inds == -1,
                                                 x2_bin_inds == len(x2_bins)-1)
                # Temporary bin selection, see below for correction.
                x1_bin_inds[x1_bin_inds_mask] = 0
                x2_bin_inds[x2_bin_inds_mask] = 0

                densities = hist[x1_bin_inds, x2_bin_inds]
                # Set all samples outside histogram to min density.
                mask = np.logical_or(x1_bin_inds_mask, x2_bin_inds_mask)
                densities[mask] = eps

                densities[densities < eps] = eps

                return densities, np.log(densities)

            #fake_model_densities, log_fake_model_densities = \
            #    compute_model_densities(fake)
            _, log_fake_model_densities = compute_model_densities(fake)

            _, log_real_model_densities = compute_model_densities(real)

        elif density_estimation == 'gaussian':
            kde = KernelDensity(kernel='gaussian').fit(fake)

            log_fake_model_densities = kde.score_samples(fake)
            #fake_model_densities = np.exp(log_fake_model_densities)

            log_real_model_densities = kde.score_samples(real)
            #fake_real_densities = np.exp(log_real_model_densities)

        else:
            raise ValueError('Density estimation method %s unknown!'
                             % density_estimation)

        # DELETEME
        """
        plt.figure()
        plt.title('model density of real samples', size=20)
        norm = Normalize(vmin=real_data_densities.min(),
                         vmax=real_data_densities.max())
        cmap = cm.get_cmap(name='viridis')
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(np.asarray([norm.vmin, norm.vmax]))
        plt.colorbar(sm)
        plt.scatter(real[:, 0], real[:, 1], edgecolors='b',
            label='real', facecolor=cmap(norm(real_data_densities.squeeze())))
        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

        plt.figure()
        plt.title('estimated data density of fake samples', size=20)
        norm = Normalize(vmin=fake_model_densities.min(),
                         vmax=fake_model_densities.max())
        cmap = cm.get_cmap(name='viridis')
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(np.asarray([norm.vmin, norm.vmax]))
        plt.colorbar(sm)
        plt.scatter(fake[:, 0], fake[:, 1], edgecolors='b',
            label='fake', facecolor=cmap(norm(fake_model_densities.squeeze())))
        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
        """

        ### Compute MC estimate of symmetric KL divergence ###
        # We use real samples when we MC estimate KL( real || fake ).
        kl_pq = 1/m * (log_real_data_densities - log_real_model_densities).sum()
        # We use fake samples when we MC estimate KL( fake || real ).
        kl_qp = 1/n * (log_fake_model_densities - log_fake_data_densities).sum()

        sym_kl_div = kl_pq + kl_qp

        return sym_kl_div

    def get_input_mesh(self, x1_range=None, x2_range=None, grid_size=1000):
        """Create a 2D grid of input values.
        
        The default grid returned by this method will also be the default grid
        used by the method :meth:`plot_uncertainty_map`.

        Note:
            This method is only implemented for 2D datasets.

        Args:
            x1_range (tuple, optional): The min and max value for the first
                input dimension. If not specified, the range will be
                automatically inferred.
                
                Automatical inference is based on the underlying data (train
                and test). The range will be set, such that all data can be
                drawn inside.
            x2_range (tuple, optional): Same as ``x1_range`` for the second
                input dimension.
            grid_size (int or tuple): How many input samples per dimension.
                If an integer is passed, then the same number grid size will be
                used for both dimension. The grid is build by equally spacing
                ``grid_size`` inside the ranges ``x1_range`` and ``x2_range``.

        Returns:
            (tuple): Tuple containing:

            - **x1_grid** (numpy.ndarray): A 2D array, containing the grid
              values of the first dimension.
            - **x2_grid** (numpy.ndarray): A 2D array, containing the grid
              values of the second dimension.
            - **flattended_grid** (numpy.ndarray): A 2D array, containing all
              samples from the first dimension in the first column and all
              values corresponding to the second dimension in the second column.
              This format correspond to the input format as, for instance,
              returned by methods such as
              :meth:`data.dataset.Dataset.get_train_inputs`.
        """
        if self.in_shape[0] != 2:
            raise ValueError('This method only applies to 2D datasets.')

        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        else:
            assert len(grid_size) == 2

        if x1_range is None or x2_range is None:
            min_x1 = self._data['in_data'][:, 0].min()
            min_x2 = self._data['in_data'][:, 1].min()
            max_x1 = self._data['in_data'][:, 0].max()
            max_x2 = self._data['in_data'][:, 1].max()

            slack_1 = (max_x1 - min_x1) * 0.02
            slack_2 = (max_x2 - min_x2) * 0.02

            if x1_range is None:
                x1_range = (min_x1 - slack_1, max_x1 + slack_1)
            else:
                assert len(x1_range) == 2

            if x2_range is None:
                x2_range = (min_x2 - slack_2, max_x2 + slack_2)
            else:
                assert len(x2_range) == 2

        x1 = np.linspace(start=x1_range[0], stop=x1_range[1], num=grid_size[0])
        x2 = np.linspace(start=x2_range[0], stop=x2_range[1], num=grid_size[1])

        X1, X2 = np.meshgrid(x1, x2)
        X = np.vstack([X1.ravel(), X2.ravel()]).T

        return X1, X2, X

    def plot_uncertainty_map(self, title='Uncertainty Map', input_mesh=None,
                             uncertainties=None,
                             use_generative_uncertainty=False,
                             use_ent_joint_uncertainty=False,
                             sample_inputs=None, sample_modes=None,
                             sample_label=None, sketch_components=False,
                             norm_eps=None, show=True, filename=None,
                             figsize=(10, 6)):
        r"""Draw an uncertainty heatmap.

        Args:
            title (str): Title of plots.
            input_mesh (tuple, optional): The input mesh of the heatmap (see
                return value of method :meth:`get_input_mesh`). If not
                specified, the default return value of method
                :meth:`get_input_mesh` is used.
            uncertainties (numpy.ndarray, optional): The uncertainties
                corresponding to ``input_mesh``. If not specified, then the
                uncertainties will be computed based the entropy across
                :math:`k=1..K` for
                
                .. math::

                    p(y_k = 1 \mid x) = \frac{ \
                     \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)}{\
                     \sum_{l=1}^K \pi_l \mathcal{N}(x; \mu_l, \Sigma_l)}

                Note:
                    The entropies will be normalized by the maximum uncertainty
                    ``-np.log(1.0 / self.num_modes)``.
            use_generative_uncertainty (bool): If ``True``, the uncertainties
                plotted by default (if ``uncertainties`` is left unspecified)
                are not based on the entropy of the responsibilities
                :math:`p(y_k = 1 \mid x)`, but are the densities of the
                underlying GMM :math:`p(x)`.
            use_ent_joint_uncertainty (bool): If ``True``, the uncertainties
                plotted by default (if ``uncertainties`` is left unspecified)
                are based on the entropy of :math:`p(y, x)` at location
                :math:`x`:

                .. math::

                    & - \sum_k p(x) p(y_k=1 \mid x) \log p(x) p(y_k=1 \mid x)\\\
                    =& -p(x) \sum_k p(y_k=1 \mid x) \log p(y_k=1 \mid x) - \
                        p(x) \log p(x)

                Note, we normalize :math:`p(x)` by its maximum inside the chosen
                grid. Hence, the plot depends on the chosen ``input_mesh``. In
                this way, :math:`p(x) \in [0, 1]` and the second term
                :math:`-p(x) \log p(x) \in [0, \exp(-1)]` (note,
                :math:`-p(x) \log p(x)` would be negative for :math:`p(x) > 1`).

                The first term is simply the entropy of :math:`p(y \mid x)`
                scaled by :math:`p(x)`. Hence, it shows where in the input space
                are the regions where Gaussian bumps are overlapping (regions
                in which data exists but multiple labels :math:`y` are
                possible).

                The second term shows the boundaries of the data manifold. Note,
                :math:`-1 \log 1 = 0` and
                :math:`-\lim_{p(x) \rightarrow 0} p(x) \log p(x) = 0`.

                Note:
                    This option is mutually exclusive with option
                    ``use_generative_uncertainty``.

                Note:
                    Entropies of :math:`p(y \mid x)` won't be normalized in this
                    case.
            sample_inputs (numpy.ndarray, optional): Sample inputs. Can be
                specified if a scatter plot of samples (e.g., train samples)
                should be laid above the heatmap.
            sample_modes (numpy.ndarray, optional): To which mode do the samples
                in ``sample_inputs`` belong to? If provided, then for each
                sample in ``sample_inputs`` a number smaller than
                :attr:`num_modes` is expected. All samples with the same mode
                identifier are colored with the same color.
            sample_label (str, optional): If a label should be shown in the
                legend for inputs ``sample_inputs``.
            sketch_components (bool): Sketch the mean and variance of each
                component.
            norm_eps (float, optional): If uncertainties are computed by this
                method, then (normalized) densities for each x-value in the
                input mesh have to be computed. To avoid division by zero,
                a positive number ``norm_eps`` can be specified.
            (....): See docstring of method
                :meth:`data.dataset.Dataset.plot_samples`.
        """
        assert not use_generative_uncertainty or not use_ent_joint_uncertainty
        if input_mesh is None:
            input_mesh = self.get_input_mesh()
        else:
            assert len(input_mesh) == 3
        X1, X2, X = input_mesh

        if uncertainties is None:
            responsibilities = self._compute_responsibilities(X,
                normalize=not use_generative_uncertainty, eps=norm_eps)
            if use_generative_uncertainty:
                uncertainties = responsibilities.sum(axis=1)
            else:
                # Compute entropy.
                uncertainties = - np.sum(responsibilities * \
                    np.log(np.maximum(responsibilities, 1e-5)), axis=1)

                if use_ent_joint_uncertainty:
                    cond_entropies = np.copy(uncertainties)

                    # FIXME Instead of computing responsibilities again, we
                    # should let `_compute_responsibilities` return both.
                    unnormed_resps = self._compute_responsibilities(X,
                        normalize=False)
                    loc_densities = unnormed_resps.sum(axis=1)
                    # Make sure that p(x) is between 0 and 1.
                    loc_densities /= loc_densities.max()

                    uncertainties = loc_densities * cond_entropies - \
                        loc_densities * np.log(np.maximum(loc_densities, 1e-5))

                    # Look at individual terms instead (by uncommeting).
                    # Areas where data is still likely but uncertainty is high
                    # (e.g., overlapping Gaussian bumps)
                    #uncertainties = loc_densities * cond_entropies
                    # Areas where data is still somewhat likely (not totally
                    # OOD) but also not very common -> boundary of the data
                    # manifold.
                    #uncertainties = -loc_densities * \
                    #    np.log(np.maximum(loc_densities, 1e-5))
                else:
                    # Normalize conditional entropies.
                    max_entropy = -np.log(1.0 / self.num_modes)
                    uncertainties /= max_entropy
        else:
            assert np.all(np.equal(uncertainties.shape,
                                   [X.shape[0], 1]))

        if np.any(np.isnan(uncertainties)):
            warn('NaN detected in uncertainties to be drawn. Set to 0 instead!')
        uncertainties[np.isnan(uncertainties)] = 0.

        uncertainties = uncertainties.reshape(X1.shape)

        fig, ax = plt.subplots(figsize=figsize)
        plt.title(title, size=20)

        f = plt.contourf(X1, X2, uncertainties)
        plt.colorbar(f)

        if sample_inputs is not None:
            n = self.num_modes
            colors = np.asarray(misc.get_colorbrewer2_colors(family='Dark2'))
            if n > len(colors):
                colors = cm.rainbow(np.linspace(0, 1, n))

            plt.scatter(sample_inputs[:, 0], sample_inputs[:, 1],
                        color='b' if sample_modes is None else None,
                        label=sample_label,
                        facecolor=colors[sample_modes.squeeze().astype(int)] \
                        if sample_modes is not None else None)

        if sketch_components:
            self._draw_components('Means')

        if sample_inputs is not None and sample_label is not None or \
                sketch_components:
            plt.legend()

        plt.xlabel('x1')
        plt.ylabel('x2')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        if show:
            plt.show()

    def plot_optimal_classification(self, title='Classification Map',
                                    input_mesh=None, mesh_modes=None,
                                    sample_inputs=None, sample_modes=None,
                                    sample_label=None, sketch_components=False,
                                    show=True, filename=None, figsize=(10, 6)):
        r"""Plot a color-coded grid on how to optimally classify for each input
        value.

        Note:
            Since the training data is drawn randomly, it might be that some
            training samples have a label that doesn't correpond to the optimal
            label.

        Args:
            (....): See arguments of method :meth:`plot_uncertainty_map`.
            mesh_modes (numpy.ndarray, optional): If not provided, then the
                color of each grid position :math:`x` is determined based on
                :math:`\arg\max_k \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)`.
                Otherwise, the labeling provided here will determine the
                coloring.
        """
        if input_mesh is None:
            input_mesh = self.get_input_mesh()
        else:
            assert len(input_mesh) == 3
        _, _, X = input_mesh

        if mesh_modes is None:
            responsibilities = self._compute_responsibilities(X)
            optimal_labels = responsibilities.argmax(axis=1)
        else:
            assert np.all(np.equal(mesh_modes.shape,
                                   [X.shape[0], 1]))
            optimal_labels = mesh_modes

        n = self.num_modes
        colors = np.asarray(misc.get_colorbrewer2_colors(family='Dark2'))
        if n > len(colors):
            colors = cm.rainbow(np.linspace(0, 1, n))

        fig, ax = plt.subplots(figsize=figsize)
        plt.title(title, size=20)

        plt.scatter(X[:, 0], X[:, 1], s=1,
                    facecolor=colors[optimal_labels.squeeze().astype(int)])

        if sample_inputs is not None:
            plt.scatter(sample_inputs[:, 0], sample_inputs[:, 1],
                        color='b' if sample_modes is None else None,
                        label=sample_label,
                        edgecolor='k' if sample_modes is not None else None,
                        facecolor=colors[sample_modes.squeeze().astype(int)] \
                        if sample_modes is not None else None)

        if sketch_components:
            self._draw_components('Means')

        if sample_inputs is not None and sample_label is not None or \
                sketch_components:
            plt.legend()

        plt.xlabel('x1')
        plt.ylabel('x2')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        if show:
            plt.show()

    def _draw_components(self, label=None):
        """Sketh the individual Gaussian components in the current plot.

        Method can be called while building a plot. It will sketch the mean
        of each component as well as the covariance matrix (as an ellipse).

        Args:
            label (str): Legend label associated with the added scatter plot
                of means.
        """
        m1 = [m[0] for m in self._means]
        m2 = [m[1] for m in self._means]

        plt.scatter(m1, m2, color='k', label=label)

        for i, m in enumerate(self._means):
            # https://cookierobotics.com/007/
            a = self._covs[i][0,0]
            b = self._covs[i][0,1]
            assert np.isclose(b, self._covs[i][1,0])
            c = self._covs[i][1,1]
            
            l1 = (a+c) / 2 + np.sqrt(((a-c)/2)**2 + b**2)
            l2 = (a+c) / 2 - np.sqrt(((a-c)/2)**2 + b**2)
            if b == 0 and a >= c:
                theta = 0
            elif b == 0 and a < c:
                theta = np.pi / 2
            else:
                theta = np.arctan2(l1-a, b)
            theta = theta * 180 / np.pi # radiant to degree
            ellipse = Ellipse(xy=m, width=l1, height=l2, angle=theta,
                              edgecolor='k', facecolor='none')
            plt.gca().add_artist(ellipse)

if __name__ == '__main__':
    pass
