#!/usr/bin/env python3
# Copyright 2020 Benjamin Ehret
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
# title           :data/timeseries/audioset_data.py
# author          :be
# contact         :behret@ethz.ch
# created         :02/04/2020
# version         :1.0
# python_version  :3.7
"""
Dataset for the Audioset task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A data handler for the audioset dataset taken from:

    https://research.google.com/audioset/download.html

Data were preprocessed with the script
:mod:`data.timeseries.structure_audioset` and then uploaded to
`dropbox <https://www.dropbox.com/s/07dfeeuf5aq4w1h/\
audioset_data_balanced?dl=1>`__. If this link becomes invalid, the data has to
be preprocessed from scratch.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import urllib.request

from hypnettorch.data.sequential_dataset import SequentialDataset

class AudiosetData(SequentialDataset):
    """Datahandler for the audioset task.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be
            represented in a one-hot encoding.
        validation_size (int): The number of validation samples.
        target_per_timestep (bool, optional): If activated, the one-hot
            encoding of the current image will be copied across the entire
            sequence. Else, there is a single target for the entire
            sequence (rather than one per timestep.
        rseed (int, optional): If ``None``, the current random state of numpy 
            is used to select a validation set from the training data.
            Otherwise, a new random state with the given seed is generated.
    """
    def __init__(self, data_path, use_one_hot=True, validation_size=0,
                 target_per_timestep=True, rseed=None):
        super().__init__()

        self.target_per_timestep = target_per_timestep

        if rseed is not None:
            rstate = np.random.RandomState(rseed)
        else:
            rstate = np.random

        # If dataset does not exist in dataset folder, download it from dropbox.
        # FIXME Dropbox link might become invalid in the near future.
        data_path = os.path.join(data_path,
            'sequential/audioset/audioset_data_balanced.pickle')
        if not os.path.exists(data_path):
            data_dir = os.path.dirname(data_path)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            url = 'https://www.dropbox.com/s/07dfeeuf5aq4w1h/' +\
                'audioset_data_balanced?dl=1'
            try:
                u = urllib.request.urlopen(url)
                data = u.read()
                u.close()
            except:
                raise RuntimeError('Audioset data cannot be downloaded. '+
                    'If you are working on the cluster, please manually '+
                    'copy the pickled dataset into the following location: '
                    '%s. ' % (data_path) + 'If the dropbox link (%s) ' % url +
                    'is invalid, please rebuild the dataset using the script ' +
                    '"preprocess_audioset.py".')

            with open(data_path, "wb") as f:
                f.write(data)

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Load data.
        [x_data, y_data, train_ind, test_ind] = data

        # Specify internal data structure.
        self._data['sequence'] = True
        self._data['classification'] = True
        self._data['num_classes'] = 100
        self._data['in_shape'] = [128]
        self._data['out_shape'] = [self._data['num_classes'] \
            if use_one_hot else 1]
        self._data['num_time_steps'] = 10
        self._data['is_one_hot'] = use_one_hot
        self._data['in_data'] = self._flatten_array(x_data)
        self._data['out_data'] = \
            self._structure_output_data(y_data.reshape(-1, 1))
        if use_one_hot:
            self._data['out_data'] = self._to_one_hot(self._data['out_data'])

        if not (validation_size >= 0 and validation_size < len(train_ind)):
            raise ValueError('Invalid validation set size.')
        if validation_size > 0:
            # Note, the data is not shuffled! I.e., consecutive indices belong
            # to the same class.
            train_ind, val_ind = train_test_split(train_ind,
                test_size=validation_size, shuffle=True, random_state=rstate,
                stratify=y_data[train_ind])
            self._data['val_inds'] = val_ind
        self._data['train_inds'] = train_ind
        self._data['test_inds'] = test_ind

        # Note, all sequences in this dataset have the same length.
        num_samples = self._data['in_data'].shape[0]
        self._data['in_seq_lengths'] = np.ones(num_samples) * \
            self._data['num_time_steps']
        if target_per_timestep:
            self._data['out_seq_lengths'] = self._data['in_seq_lengths']
        else:
            self._data['out_seq_lengths'] = \
                np.ones_like(self._data['in_seq_lengths'])

    def _structure_output_data(self, out_data):
        """Restructures the sample output data to the format expected by the 
        dataset class. 

        The task has one global target (for all timesteps), given as a one
        hot encoding. However this can be changed using the option 
        `target_per_timestep`.

        Args:
            out_data (list): List of length ``n_samples`` (total number of
                samples in the dataset). Each sample is a 1D array of
                size ``[10]``.
        Returns:
            (numpy.ndarray): Numpy array of shape 
                ``[n_samples, self._data['out_shape']]``.
        """
        out_mat = out_data

        if self.target_per_timestep:
            out_mat = np.matlib.repmat(np.asarray(out_mat), 1,
                                       self._data['num_time_steps'])

        return out_mat

    def _plot_config(self, inputs, outputs=None, predictions=None):
        """Defines properties, used by the method :meth:`plot_samples`.

        This method can be overwritten, if these configs need to be different
        for a certain dataset.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset._plot_config`.

        Returns:
            (dict): A dictionary with the plot configs.
        """
        plot_configs = super()._plot_config(inputs, outputs=outputs,
                                            predictions=predictions)
        if outputs is not None:
            plot_configs['num_inner_rows'] += 1
        if predictions is not None:
            plot_configs['num_inner_rows'] += 1
        plot_configs['num_inner_plots'] = plot_configs['num_inner_rows']

        return plot_configs

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None, is_one_hot=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset._plot_sample`.
            is_one_hot (bool, optional): Whether ``outputs`` and ``predictions``
                are provided as 1-hot encodings. If not specified, we will
                assume the value specified by attribute
                :attr:`data.dataset.Dataset.is_one_hot`.
        """
        if is_one_hot is None:
            is_one_hot = self.is_one_hot

        # Bring the data into a proper form.
        x = self._flatten_array(inputs, ts_dim_first=True, reverse=True,
                                feature_shape=self.in_shape)
        pdata = [x]
        plabel = ['inputs']

        if outputs is not None:
            # We want to display outputs as one-hot encoding.
            raise NotImplementedError
            # t = ...
            pdata.append(t)
            plabel.append('outputs')

        if predictions is not None:
            fs = [self.num_classes] if is_one_hot else [1]
            y = self._flatten_array(predictions, ts_dim_first=True,
                                    reverse=True, feature_shape=fs)
            pdata.append(y)
            plabel.append('predictions')

        for i, d in enumerate(pdata):
            ax = plt.Subplot(fig, inner_grid[i])
            # Note, we can't use `set_axis_off`, if we wanna keep the y-label.
            ax.set_ylabel(plabel[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(d.squeeze().transpose())
            fig.add_subplot(ax)

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'audioset'

    def __str__(self):
        """Print major characteristics of the current dataset."""
        return 'Data handler for the audioset dataset.'

if __name__ == '__main__':
    pass
