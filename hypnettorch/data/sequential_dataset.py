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
# @title          :data/sequential_dataset.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :03/31/2020
# @version        :1.0
# @python_version :3.6.10
"""
Wrapper for sequential datasets
-------------------------------

The module :mod:`data.sequential_dataset` contains an abstract wrapper for
datasets containing sequential data.

Even though the dataset interface :class:`data.dataset.Dataset` contains basic
support for sequential datasets, this wrapper was considered necessary to
increase the convinience when working with sequential datasets (especially,
if those datasets contain sequences of varying lengths).
"""
import numpy as np
import torch

from hypnettorch.data.dataset import Dataset

class SequentialDataset(Dataset):
    """A general wrapper for datasets with sequential inputs and outpus."""
    def __init__(self):
        super().__init__()

        ### IMPLEMENTATION HINTS FOR DEVELOPERS                              ###
        # A major limitation of the dataset interface `Dataset` is that it     #
        # doesn't provide a convinient way of accessing the unpadded sequence  #
        # lengths (note, all samples are internally stored with the same       #
        # length, i.e., zero-padded whenever necessary).                       #
        # Therefore, this wrapper introduces additional attributes to the      #
        # internal dict `_data` that have to be filled.                        #
        #                                                                      #
        # Note, `self._data['in_shape']` is the per timestep shape. For        #
        # instance, consider a sample of shape `[T, C]`, where `T` is the      #
        # sequence length (number of timesteps) and `C` is a feature dimension #
        # (e.g., number of classes). Then, we set                              #
        # `self._data['in_shape'] = [C]`.                                      #
        # Similar considerations account for the output shape                  #
        # `self._data['out_shape']`.                                           #
        ########################################################################

        # This wrapper is only for sequential datasets.
        self._data['sequence'] = True

        # Numpy array: The length of each dataset input sample.
        # Note, all samples are expected to be stored internally all with the
        # same length (i.e., zero-padded). This attribute allows us to extract
        # the unpadded input sequences again.
        # Note, the length of this array must be equal to
        # `self._data['in_shape'].shape[0]`.
        self._data.setdefault('in_seq_lengths', None)
        # Numpy array: The length of each dataset output sample.
        self._data.setdefault('out_seq_lengths', None)

    @property
    def max_num_ts_in(self):
        """The maximum number of timesteps input sequences may have.

        Note:
            Internally, all input sequences are stored according to this
            length using zero-padding.

        :type: int
        """
        return self._data['in_data'].shape[1] // np.prod(self.in_shape)

    @property
    def max_num_ts_out(self):
        """The maximum number of timesteps output sequences may have.

        Note:
            Internally, all input sequences are stored according to this
            length using zero-padding.

        :type: int
        """
        return self._data['out_data'].shape[1] // np.prod(self.out_shape)

    def get_in_seq_lengths(self, sample_ids):
        """Get the unpadded input sequence lengths for given samples.

        Args:
            sample_ids (numpy.ndarray): A 1D numpy array of unique sample
                identifiers. Please see documentation of option ``return_ids``
                of method :meth:`data.dataset.Dataset.next_train_batch` as well
                as method :meth:`data.dataset.Dataset.get_train_ids` for more
                information of sample identifiers.

        Returns:
            (numpy.ndarray): A 1D array of the same length as ``sample_ids``
            containing the unpadded input sequence lengths of these samples.
        """
        return self._data['in_seq_lengths'][sample_ids]

    def get_out_seq_lengths(self, sample_ids):
        """Get the unpadded output sequence lengths for given samples.

        See documentation of method :meth:`get_in_seq_lengths`.

        Args:
            (....): See docstring of method :meth:`get_in_seq_lengths`.

        Returns:
            (numpy.ndarray): A 1D numpy array.
        """
        return self._data['out_seq_lengths'][sample_ids]

    @staticmethod
    def _flatten_array(arr, ts_dim_first=False, reverse=False,
                       feature_shape=None):
        """Helper function to flatten arrays.

        Flattens a given numpy array such that it is prepared for internal
        storage in attributes such as ``self._data['in_data']`` and
        ``self._data['out_data']``.

        Args:
            arr (numpy.ndarray): Numpy array of shape
                ``[batch_size, time_steps, *arr.shape[2:]]``.
            ts_dim_first (bool): If set to ``True``, then the input array
                ``arr`` is expected to be of shape
                ``[time_steps, batch_size, *arr.shape[2:]]``.
            reverse (bool): If ``True``, then the given ``arr`` is actually
                expected to be in 2D shape
                ``[batch_size, time_steps * np.prod(feature_shape)]``.
                In this case, the array will be unflattened.
            feature_shape (list, optional): Only required if ``reverse`` is
                ``True``.

        Returns:
            (numpy.ndarray): Numpy array of shape
            ``[batch_size, time_steps * np.prod(arr.shape[2:])]``.
        """
        if reverse:
            if feature_shape is None:
                raise ValueError('Option "feature_shape" must be specified '+
                                 'if "reverse" is True.')

            assert len(arr.shape) == 2
            num_samples = arr.shape[0]
            num_ts = arr.shape[1] // np.prod(feature_shape)

            arr = np.reshape(arr, [num_samples, num_ts, *feature_shape])

            if ts_dim_first:
                arr = np.swapaxes(arr, 0, 1)

        else:
            if ts_dim_first:
                arr = np.swapaxes(arr, 0, 1)

            num_samples = arr.shape[0]
            num_ts = arr.shape[1]
            num_features = np.prod(arr.shape[2:])

            arr = np.reshape(arr, [num_samples, num_ts * num_features])

        return arr

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor. It has
            dimensions ``[T, B, *in_shape]``, where ``T`` is the number of time
            steps (see attribute :attr:`max_num_ts_in`), ``B`` is the batch size
            and ``in_shape`` refers to the input feature shape, see
            :attr:`data.dataset.Dataset.in_shape`.
        """
        # FIXME Reduce padding within mini-batch if `sample_ids` is given?
        # Could be problematic if input and output seq have different seq
        # lengths but are padded to the same length.
        out_tensor = self._flatten_array(x, ts_dim_first=True, reverse=True,
                                         feature_shape=self.in_shape)
        return torch.from_numpy(out_tensor).float().to(device)

    def output_to_torch_tensor(self, y, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):
        """Similar to method :meth:`input_to_torch_tensor`, just for dataset
        outputs.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.output_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor. It has
            dimensions ``[T, B, *out_shape]``, where ``T`` is the number of time
            steps (see attribute :attr:`max_num_ts_out`), ``B`` is the batch
            size and ``out_shape`` refers to the output feature shape, see
            :attr:`data.dataset.Dataset.out_shape`.
        """
        # FIXME Reduce padding within mini-batch if `sample_ids` is given?
        # Could be problematic if input and output seq have different seq
        # lengths but are padded to the same length.
        out_tensor = self._flatten_array(y, ts_dim_first=True, reverse=True,
                                         feature_shape=self.out_shape)
        # FIXME Delete commented code.
        #if isinstance(out_tensor, torch.Tensor):
        #    return out_tensor.float().to(device)
        #else:
        return torch.from_numpy(out_tensor).float().to(device)

if __name__ == '__main__':
    pass


