#!/usr/bin/env python3
# Copyright 2019 Benjamin Ehret
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
# title           :data/timeseries/cognitive_tasks/cognitive_data.py
# author          :be
# contact         :behret@ethz.ch
# created         :29/10/2019
# version         :1.0
# python_version  :3.7
"""
Set of cognitive tasks
^^^^^^^^^^^^^^^^^^^^^^

A data handler for cognitive tasks as implemented in Masse et al (PNAS). The 
user can construct individual datasets with this data handler and use each of 
these datasets to train a model in a continual leraning setting.
"""
import numpy as np
from  torch import from_numpy

# from Masse et al. code base, needed for task generation
import hypnettorch.data.timeseries.cognitive_tasks.stimulus as stim_masse
import hypnettorch.data.timeseries.cognitive_tasks.parameters as params_masse

from hypnettorch.data.dataset import Dataset

# TODO Use `SequentialDataset` as baseclass.
class CognitiveTasks(Dataset):
    """An instance of this class shall represent a one of the 20 cognitive
    tasks.
    """
    def __init__(self, task_id=0, num_train=80, num_test=20, num_val=None,
                 rstate=None):
        """Generate a new dataset.

        We use the MultiStimulus class from Masse el al. to genereate
        the inputs and outputs of different cognitive tasks in accordance with
        the data handling structures of the hnet code base.
        
        Note that masks (part of the Masse et al. trial generator) will be
        handled independently of this data handler.

        Args:
            num_train (int): Number of training samples.
            num_test (int): Number of test samples.
            num_val (optional): Number of validation samples.
            rstate: If ``None``, the current random state of numpy is used to
                generate the data.
        """
        super().__init__()

        # set random state
        if rstate is not None:
            self._rstate = rstate
        else:
            self._rstate = np.random

        # TODO: generate task library and load train / test data instead of 
        # generating them for every call. Keeping this version as a quick fix
        # for now.

        # get train and test data
        train_x, train_y = self._generate_trial_samples(num_train,task_id)
        test_x, test_y = self._generate_trial_samples(num_test,task_id)

        # Create validation data if requested.
        if num_val is not None:
            val_x, val_y = self._generate_trial_samples(num_val,task_id)

            in_data = np.vstack([train_x, test_x, val_x])
            out_data = np.vstack([train_y, test_y, val_y])
        else:
            in_data = np.vstack([train_x, test_x])
            out_data = np.vstack([train_y, test_y])

        # Specify internal data structure.
        self._data['classification'] = True
        self._data['sequence'] = True
        self._data['in_shape'] = [68]
        self._data['out_shape'] = [9]
        self._data['is_one_hot'] = True
        self._data['num_classes'] = 9
        self._data['task_id'] = task_id
        self._data['in_data'] = in_data
        self._data['out_data'] = out_data
        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train, num_train + num_test)

        if num_val is not None:
            n_start = num_train + num_test
            self._data['val_inds'] = np.arange(n_start, n_start + num_val)

    def _generate_trial_samples(self,n_samples,task_id):
        """Generate a certain number of trials

        Args:
            n_samples
            task_id

        Returns:
            (tuple): Tuple containing:

            - **x**: Matrix of trial inputs of shape
              ``[batch_size, in_size*time_steps]``.
            - **y**: Matrix of trial targets of shape
              ``[batch_size, in_size*time_steps]``.
        """
        # update batch_size in their parameter dict to get desired number of
        # trials for training, then create stim object
        params_masse.update_parameters({'batch_size': n_samples})
        # create new stim object with the updated parameters
        stim = stim_masse.MultiStimulus(self._rstate)
        # generate trials and reshape
        _, x, y, _, _ = stim.generate_trial(task_id)
        x = self._flatten_tensor(x)
        y = self._flatten_tensor(y)

        return x, y

    def _flatten_tensor(self,in_tensor):
        """Flattens the trial data tensors to the format expected by the 
        dataset class.

        Args:
            in_tensor: Numpy array of shape
                ``[time_steps, batch_size, in_size]``.

        Returns:
            out_mat: Numpy array of shape ``[batch_size, in_size*time_steps]``.
        """
        (time_steps, batch_size, in_size) = in_tensor.shape
        in_tensor = np.moveaxis(in_tensor,[0,1,2],[2,0,1])
        out_mat = np.reshape(in_tensor,[batch_size, in_size*time_steps])
        
        return out_mat

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as 3D PyTorch tensor. It has
            dimensions ``[T, B, N]``, where ``T`` is the number of time steps
            per stimulus, ``B`` is the batch size and ``N`` the number of input
            units.
        """
        assert(self._data['in_data'].shape[1] % np.prod(self.in_shape) == 0)
        num_time_steps = self._data['in_data'].shape[1] // \
            np.prod(self.in_shape)

        out_tensor = np.reshape(x,[x.shape[0],self.in_shape[0],num_time_steps])
        out_tensor = np.moveaxis(out_tensor,[0,1,2],[1,2,0])

        return from_numpy(out_tensor).float().to(device)

    def output_to_torch_tensor(self, y, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):
        """Similar to method :meth:`input_to_torch_tensor`, just for dataset
        outputs.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.output_to_torch_tensor`.

        Returns:
            (torch.Tensor): A tensor of shape ``[T, B, C]``, where ``T`` is the
            number of time steps per stimulus, ``B`` is the batch size and ``C``
            the number of classes.
        """
        assert(self._data['out_data'].shape[1] % np.prod(self.out_shape) == 0)
        num_time_steps = self._data['out_data'].shape[1] // \
            np.prod(self.out_shape)

        out_tensor = np.reshape(y,[y.shape[0],self.out_shape[0],num_time_steps])
        out_tensor = np.moveaxis(out_tensor,[0,1,2],[1,2,0])

        return from_numpy(out_tensor).float().to(device)

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'Cognitive'

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Not implemented"""
        raise NotImplementedError('TODO implement')

if __name__ == '__main__':
    pass


