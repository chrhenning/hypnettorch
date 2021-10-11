#!/usr/bin/env python3
# Copyright 2020 Rafael Daetwyler
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
# @title          :data/special/regression1d_bimodal_data.py
# @author         :rd
# @contact        :rafael.daetwyler@uzh.ch
# @created        :11/06/2020
# @version        :1.0
# @python_version :3.7.4
"""
1D Regression Dataset with bimodal error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.special.regression1d_bimodal_data` contains a data handler
for a CL toy regression problem. The user can construct individual datasets with
this data handler and use each of these datasets to train a model in a continual
learning setting.
"""
import numpy as np
from hypnettorch.data.special.regression1d_data import ToyRegression

class BimodalToyRegression(ToyRegression):
    """An instance of this class shall represent a simple regression task, but
    with a bimodal Gaussian mixture error distribution.
    """
    def __init__(self, train_inter=[-10, 10], num_train=20,
                 test_inter=[-10, 10], num_test=80, val_inter=None,
                 num_val=None, map_function=lambda x : x, alpha1=0.5, dist1=5,
                 dist2=None, std1=1, std2=None, rseed=None,
                 perturb_test_val=False):
        """Generate a new dataset.

        The input data x will be uniformly drawn for train samples and
        equidistant for test samples. The user has to specify a function that
        will map this random input data onto output samples y.

        Args:
            (....): See docstring of class
                :class:`data.special.regression_1d_data.ToyRegression`.
            alpha1: Mixture coefficient of the first Gaussian mode of the error.
            dist1: The distance from zero of mean of the first Gaussian
                component of the error.
            dist2 (optional): The distance from zero of mean of the first
                Gaussian component of the error.  If ``None``, the value of
                `dist1` will be taken.
            std1: The standard deviation of the first Gaussian component of the
                error.
            std2 (optional): The standard deviation of the first Gaussian
                component of the error. If ``None``, the value of `std1` will be
                taken.
        """
        super().__init__()

        assert(val_inter is None and num_val is None or \
               val_inter is not None and num_val is not None)
        assert(0 <= alpha1 <= 1)

        if rseed is None:
            rand = np.random
        else:
            rand = np.random.RandomState(rseed)

        train_x = rand.uniform(low=train_inter[0], high=train_inter[1],
                               size=(num_train, 1))
        test_x = np.linspace(start=test_inter[0], stop=test_inter[1],
                             num=num_test).reshape((num_test, 1))

        train_y = map_function(train_x)
        test_y = map_function(test_x)

        # Perturb training outputs.
        if dist2 is None:
            dist2 = dist1
        if std2 is None:
            std2 = std1

        dist = np.array([-dist1, dist2])
        std = np.array([std1, std2])
        train_mode = rand.binomial(1, alpha1, (num_train, 1))
        train_eps = rand.normal(loc=dist[train_mode], scale=std[train_mode])
        train_y += train_eps

        if perturb_test_val:
            test_mode = rand.binomial(1, alpha1, (num_test, 1))
            test_eps = rand.normal(loc=dist[test_mode], scale=std[test_mode])
            test_y += test_eps

        # Create validation data if requested.
        if num_val is not None:
            val_x = np.linspace(start=val_inter[0], stop=val_inter[1],
                                num=num_val).reshape((num_val, 1))
            val_y = map_function(val_x)

            if perturb_test_val:
                val_mode = rand.binomial(1, alpha1, (num_val, 1))
                val_eps = rand.normal(loc=dist[val_mode],
                                       scale=std[val_mode])
                val_y += val_eps

            in_data = np.vstack([train_x, test_x, val_x])
            out_data = np.vstack([train_y, test_y, val_y])
        else:
            in_data = np.vstack([train_x, test_x])
            out_data = np.vstack([train_y, test_y])

        # Specify internal data structure.
        self._data['classification'] = False
        self._data['sequence'] = False
        self._data['in_data'] = in_data
        self._data['in_shape'] = [1]
        self._data['out_data'] = out_data
        self._data['out_shape'] = [1]
        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train, num_train + num_test)

        if num_val is not None:
            n_start = num_train + num_test
            self._data['val_inds'] = np.arange(n_start, n_start + num_val)

        self._alpha1 = alpha1
        self._dist1 = dist1
        self._dist2 = dist2
        self._std1 = std1
        self._std2 = std2
        self._map = map_function
        self._train_inter = train_inter
        self._test_inter = test_inter
        self._val_inter = val_inter

    def get_identifier(self):
        """Returns the name of the dataset."""
        return '1D Bimodal Regression'


if __name__ == '__main__':
    pass


