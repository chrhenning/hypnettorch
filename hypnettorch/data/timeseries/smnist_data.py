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
# title           :data/timeseries/smnist_data.py
# author          :be
# contact         :behret@ethz.ch
# created         :23/03/2020
# version         :1.0
# python_version  :3.7
"""
Stroke MNIST (SMNIST) Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A data handler for the stroke mnist data as discribed here:

    https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/

The data was preprocessed with the script
:mod:`data.timeseries.preprocess_smnist` and then uploaded to
`dropbox <https://www.dropbox.com/s/sadzc8qvjvexdtx/ss_mnist_data?dl=1>`__. If
this link becomes invalid, the data has to be preprocessed from scratch.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os 
import urllib.request

from hypnettorch.data.sequential_dataset import SequentialDataset

class SMNISTData(SequentialDataset):
    """Datahandler for stroke MNIST.

    Note:
        That the outputs are always provided as one-hot encodings of duration
        equal to one. One can decide to make these targets span the entirety of
        the sequence (by repeating it over timesteps) by setting
        ``target_per_timestep`` to ``True``.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be
            represented in a one-hot encoding.
        validation_size (int): The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
        target_per_timestep (bool): If activated, the one-hot
            encoding of the current image will be copied across the entire
            sequence. Else, there is a single target for the entire
            sequence (rather than one per timestep.
    """
    def __init__(self, data_path, use_one_hot=False, validation_size=0,
                 target_per_timestep=True):
        super().__init__()

        self.target_per_timestep = target_per_timestep

        # If dataset does not exist in dataset folder, download it from dropbox.
        # FIXME Dropbox link might become invalid in the near future.
        data_path = os.path.join(data_path,
                                 'sequential/smnist/ss_mnist_data.pickle')
        if not os.path.exists(data_path):
            data_dir = os.path.dirname(data_path)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            url = "https://www.dropbox.com/s/sadzc8qvjvexdtx/ss_mnist_data?dl=1"
            try:
                u = urllib.request.urlopen(url)
                data = u.read()
                u.close()
            except:
                raise RuntimeError('SMNIST data cannot be downloaded. '+
                    'If you are working on the cluster, please manually '+
                    'copy the pickled dataset into the following location: '
                    '%s. ' % (data_path) + 'If the dropbox link (%s) ' % url +
                    'is invalid, please rebuild the dataset using the script ' +
                    '"preprocess_smnist.py".')

            with open(data_path, "wb") as f:
                f.write(data)

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Concatenate train and test samples from original data set.
        # Partitioning will be redone below.
        x_data = data[0] + data[2]
        y_data = data[1] + data[3]

        if not (validation_size >= 0 and validation_size < 60000):
            raise ValueError('Invalid validation set size.')
        num_train = 60000 - validation_size
        num_test = 10000
        num_val = validation_size

        # Specify internal data structure.
        self._data['classification'] = True
        self._data['sequence'] = True
        self._data['num_classes'] = 10
        # Quadruple per timestep: (dx, dy, eos, eod).
        self._data['in_shape'] = [4] # Quatruple 
        self._data['out_shape'] = [10 if use_one_hot else 1]
        # Maximum number of timesteps, sequences will be padded to this length.
        self._data['num_time_steps'] = 117
        self._data['is_one_hot'] = use_one_hot
        self._data['in_data'], seq_lengths = self._structure_input_data(x_data)
        self._data['out_data'] = self._structure_output_data(y_data)
        if not use_one_hot:
            self._data['out_data'] = self._to_one_hot(self._data['out_data'],
                                                      reverse=True)
        if validation_size > 0:
            self._data['val_inds'] = np.arange(num_val)
        self._data['train_inds'] = np.arange(num_val, num_val+num_train)
        self._data['test_inds'] = np.arange(num_val+num_train,
                                            num_val+num_train + num_test)

        self._data['in_seq_lengths'] = seq_lengths
        if target_per_timestep:
            self._data['out_seq_lengths'] = seq_lengths
        else:
            self._data['out_seq_lengths'] = np.ones_like(seq_lengths)

    def _structure_input_data(self, in_data):
        """Restructures the sample input data to the format expected by the
        dataset class.

        Args:
            in_data (list): List of length ``n_samples`` (total number of
                samples in the dataset). Each sample is a 2D array of size
                ``[seq_len, 4]``, where ``seq_len`` is different for every
                sample.
                To have a common data structure, from here on every sample has 
                the same length and time steps that are not used are set to 0
                (padded)

        Returns:
            (tuple): Tuple containing:

            - **seq_data** (numpy.ndarray): Numpy array of shape
              ``[n_samples, max_num_time_steps * 4]``.
            - **seq_lengths** (numpy.ndarray): The original unpadded sequence
              lengths.
        """
        n_samples = len(in_data)
        out_mat = np.zeros((n_samples, 4 * self._data['num_time_steps']))
        seq_lengths = np.zeros(n_samples)

        for i in range(n_samples):
            assert in_data[i].shape[1] == 4
            sample_len = in_data[i].shape[0] * in_data[i].shape[1]
            seq_lengths[i] = in_data[i].shape[0]
            out_mat[i, :sample_len] = in_data[i].flatten(order='C')

            eod = np.argwhere(in_data[i][:,3])
            assert eod.size == 1 and eod.squeeze() == seq_lengths[i]-1

        assert seq_lengths.max() <= self._data['num_time_steps']

        return out_mat, seq_lengths

    def _structure_output_data(self, out_data):
        """Restructures the sample output data to the format expected by the
        dataset class.

        The task has one global target (for all timesteps), given as a 1-hot
        encoding. However this can be changed using the constructor option
        ``target_per_timestep``.

        Args:
            out_data (list): List of length ``n_samples``. Each sample is a
                1D array of size ``[10]``.

        Returns:
            (numpy.ndarray): Numpy array of shape
            ``[n_samples, max_num_time_steps]``.
        """
        out_mat = np.asarray(out_data)

        if self.target_per_timestep:
            out_mat = np.matlib.repmat(np.asarray(out_mat), 1,
                                       self._data['num_time_steps'])

        return out_mat

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'SMNIST'

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None, sample_ids=None,
                     is_one_hot=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset._plot_sample`.
            sample_ids (numpy.ndarray): See option ``sample_ids`` of method
                :meth:`get_out_pattern_bounds`. Only required if
                ``predictions`` is not ``None`` but provided as a sequence of
                labels (note, this method will consider the label at the end
                of the input sequence as predicted label).
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

        # Sanity check.
        if sample_ids is not None:
            eod = np.argwhere(x[:,0,3])
            assert eod.size == 1 and eod.squeeze() == \
                self.get_in_seq_lengths(sample_ids[[ind]]).squeeze() - 1

        if outputs is not None:
            # Note, the base class already removed 1-hot encoding from ground-
            # truth data.
            t = self._flatten_array(outputs, ts_dim_first=True, reverse=True,
                                    feature_shape=[1])
            if t.shape[0] > 1: # Multiple timesteps.
                # Note, the label should be the same across all timesteps,
                # as this is a ground-truth output.
                t = t[0,:,:]
        if predictions is not None:
            fs = [self.num_classes] if is_one_hot else [1]
            y = self._flatten_array(predictions, ts_dim_first=True,
                                    reverse=True, feature_shape=fs)
            if y.shape[0] > 1: # Multiple timesteps.
                # Note, we consider the correct label the one that is predicted
                # at the end of the input sequence.
                if sample_ids is None:
                    raise ValueError('Option "sample_ids" must be specified ' +
                                     'when providing timeseries predictions.')
                sl = self.get_in_seq_lengths(sample_ids[[ind]])
                y = y[int(sl[0])-1,:,:]

        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("SMNIST Sample")
        else:
            assert(np.size(t) == 1)
            label = np.asscalar(t)

            if predictions is None:
                ax.set_title('SMNIST sample with\nlabel: %d' % label)
            else:
                if np.size(y) == self.num_classes:
                    pred_label = np.argmax(y)
                else:
                    pred_label = np.asscalar(y)

                ax.set_title('SMNIST sample with\nlabel: %d (prediction: %d)' %
                             (label, pred_label))

        # Build image from stroke data.
        image = np.zeros((28, 28))
        eos = True
        for i in range(x.shape[0]):
            if x[i, 0, 3] == 1: # end-of-digit
                break

            if eos:
                eos = False
                x_idx = int(x[i, 0, 0]) - 1
                y_idx = int(x[i, 0, 1]) - 1
            else:
                x_idx += int(x[i, 0, 0])
                y_idx += int(x[i, 0, 1])

            # This doesn't seem to matter. Seems only the first position is
            # absolute.
            #if x[i, 0, 2] == 1: # end-of-stroke
            #    eos = True

            x_idx = 0 if x_idx < 0 else x_idx
            y_idx = 0 if y_idx < 0 else y_idx
            x_idx = 27 if x_idx > 27 else x_idx
            y_idx = 27 if y_idx > 27 else y_idx
            image[x_idx, y_idx] = 255

        ax.set_axis_off()
        ax.imshow(image.transpose())
        fig.add_subplot(ax)

        if num_inner_plots == 2:
            ax = plt.Subplot(fig, inner_grid[1])
            ax.set_title('Predictions')
            bars = ax.bar(range(self.num_classes), np.squeeze(y))
            ax.set_xticks(range(self.num_classes))
            if outputs is not None:
                bars[int(label)].set_color('r')
            fig.add_subplot(ax)

    def _plot_config(self, inputs, outputs=None, predictions=None):
        """Re-Implementation of method
        :meth:`data.dataset.Dataset._plot_config`.

        This method has been overriden to ensure, that there are 2 subplots,
        in case the predictions are given.
        """
        plot_configs = super()._plot_config(inputs, outputs=outputs,
                                            predictions=predictions)

        if predictions is not None and self.is_one_hot and \
                np.shape(predictions)[1] == self._data['out_data'].shape[1]:
            plot_configs['outer_hspace'] = 0.6
            plot_configs['inner_hspace'] = 0.4
            plot_configs['num_inner_rows'] = 2
            #plot_configs['num_inner_cols'] = 1
            plot_configs['num_inner_plots'] = 2

        return plot_configs

    def __str__(self):
        """Print major characteristics of the current dataset."""
        return 'Data handler for stroke MNIST'

if __name__ == '__main__':
    pass


