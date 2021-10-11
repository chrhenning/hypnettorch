#!/usr/bin/env python3
# Copyright 2021 Francesco D'Angelo
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
# @title          :data/special/donuts.py
# @author         :fd, ch
# @contact        :fdangelo@student.ethz.ch
# @created        :07/08/2021
# @version        :1.0
# @python_version :3.8.10
"""
2D Donut Dataset
^^^^^^^^^^^^^^^^

This data handler creates a synthetic toy problem comprising 2D annuli.
"""
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from hypnettorch.data.dataset import Dataset

class Donuts(Dataset):
    """Donut dataset handler.

    Note, each donut prescribes a different class.

    Args:
        centers (tuple or list): List of tuples, each determining the center
            of a donut.
        radii (tuple or list): List of tuples, each tuple defines the inner and
            outer radius of a donut.
        num_train (int): Number of training samples per donut.
        num_test (int): Number of test samples per donut.
        use_one_hot (bool): Whether the class labels should be represented as a
            one-hot encoding.
        rseed (int): If ``None``, the current random state of numpy is used
            to generate the data. Otherwise, a new random state with the
            given seed is generated.
    """
    def __init__(self, centers=((0, 0), (0, 0)), radii=((3,4), (9,10)),
                 num_train=100, num_test=100, use_one_hot=True, rseed=42):
        super().__init__()

        if rseed is None:
            rand = np.random
        else:
            rand = np.random.RandomState(rseed)

        assert len(centers) == len(radii)

        for i in range(len(centers)):
            c = centers[i]
            r = radii[i]

            donut_train = Donuts.sample_annulus(c[0], c[1], r[0], r[1],
                                                num=num_train, rand=rand)
            donut_test = Donuts.sample_annulus(c[0], c[1], r[0], r[1],
                                               num=num_test, rand=rand)

            if i == 0:
                train_x = donut_train
                train_y = np.ones((num_train, 1)) * i

                test_x = donut_test
                test_y = np.ones((num_test, 1)) * i
            else:
                train_x = np.vstack([train_x, donut_train])
                train_y = np.vstack([train_y, np.ones((num_train, 1)) * i])

                test_x = np.vstack([test_x, donut_test])
                test_y = np.vstack([test_y, np.ones((num_test, 1)) * i])

        in_data = np.vstack([train_x, test_x])
        out_data = np.vstack([train_y, test_y])

        # Specify internal data structure.
        self._data['classification'] = True
        self._data['sequence'] = False
        self._data['in_data'] = in_data
        self._data['in_shape'] = [2]
        self._data['num_classes'] = len(centers)
        if use_one_hot:
            out_data = self._to_one_hot(out_data)
        self._data['out_data'] = out_data
        self._data['out_shape'] = [2]
        self._data['train_inds'] = np.arange(train_x.shape[0])
        self._data['test_inds'] = np.arange(train_x.shape[0], 
                                            train_x.shape[0] + test_x.shape[0])

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'DonutsDataset'

    @staticmethod
    def sample_annulus(x_c, y_c, r_inner, r_outer, num=1, rand=None):
        r"""Sample uniformly from an annulus.

        Sample uniformly :math:`(x, y)` satisfiying:

        .. math::

           (x-x_\text{c})^2 + (y-y_\text{c})^2 \leq r_\text{outer}^2

        and

        .. math::

           (x-x_\text{c})^2 + (y-y_\text{c})^2 > r_\text{inner}^2

        Args:
            x_c (float): x-position of the center.
            y_c (float): y-position of the center.
            r_inner (float): Inner radius.
            r_outer (float): Outer radius.
            num (int): Number of samples to return.
            rand (numpy.random.RandomState, optional): Random state object
                used for sampling.

        Returns:
            (numpy.ndarray): Array of shape ``[num, 2]``.
        """
        # The code is inspired by this thread:
        # https://stackoverflow.com/questions/47005884/random-point-inside-annulus-with-a-shifted-hole
        assert r_inner <= r_outer

        if rand is None:
            rand = np.random

        # Sample from a normal annulus with radii r_inner and r_outer.
        rad = np.sqrt(rand.uniform(low=r_inner ** 2, high=r_outer ** 2,
                                     size=num))
        angle = rand.uniform(low=-np.pi, high=np.pi, size=num)
        x, y = rad * np.cos(angle) + x_c, rad * np.sin(angle) + y_c

        return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Not implemented"""
        raise NotImplementedError('TODO implement')

    def plot_dataset(self, title, show=True, filename=None, interactive=False,
                     figsize=(10, 6)):
        """Plot samples belonging to this dataset.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.plot_samples`.
        """

        colors = ListedColormap(['#FF0000', '#0000FF'])

        # Create plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        x_train_0 = self.get_train_inputs()
        y_train_0 = self.get_train_outputs()
        x_test_0 = self.get_test_inputs()
        y_test_0 = self.get_test_outputs()

        ax.scatter(x_train_0[:, 0], x_train_0[:, 1], alpha=1, marker='o',
                   c=np.argmax(y_train_0, 1), cmap=colors,
                   edgecolors='k', s=50, label='train')
        ax.scatter(x_test_0[:, 0], x_test_0[:, 1], alpha=0.6, marker='s',
                   c=np.argmax(y_test_0, 1), cmap=colors,
                   edgecolors='k', s=50, label='test')
        plt.title(title, fontsize=30)
        plt.legend(loc=2, fontsize=30)
        plt.show()

if __name__ == '__main__':
    pass


