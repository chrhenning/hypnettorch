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
# @title           :split_mnist.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :04/11/2019
# @version         :1.0
# @python_version  :3.6.7
"""
Split MNIST Dataset
^^^^^^^^^^^^^^^^^^^

The module :mod:`data.special.split_mnist` contains a wrapper for data
handlers for the SplitMNIST task.
"""
import numpy as np

from hypnettorch.data.mnist_data import MNISTData
from hypnettorch.data.special.split_cifar import _transform_split_outputs

def get_split_mnist_handlers(data_path, use_one_hot=True, validation_size=0,
                             use_torch_augmentation=False,
                             num_classes_per_task=2, num_tasks=None,
                             trgt_padding=None):
    """This function instantiates 5 objects of the class :class:`SplitMNIST`
    which will contain a disjoint set of labels.

    The SplitMNIST task consists of 5 tasks corresponding to the images with
    labels [0,1], [2,3], [4,5], [6,7], [8,9].

    Args:
        data_path: Where should the MNIST dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot: Whether the class labels should be represented in a one-hot
            encoding.
        validation_size: The size of the validation set of each individual
            data handler.
        use_torch_augmentation (bool): See docstring of class
            :class:`data.mnist_data.MNISTData`.
        num_classes_per_task (int): Number of classes to put into one data
            handler. If ``2``, then every data handler will include 2 digits.
        num_tasks (int, optional): The number of data handlers that should be
            returned by this function.
        trgt_padding (int, optional): See docstring of class
            :class:`SplitMNIST`.

    Returns:
        (list): A list of data handlers, each corresponding to a
        :class:`SplitMNIST` object.
    """
    assert num_tasks is None or num_tasks > 0
    if num_tasks is None:
        num_tasks = 10 // num_classes_per_task

    if not (num_tasks >= 1 and (num_tasks * num_classes_per_task) <= 10):
        raise ValueError('Cannot create SplitMNIST datasets for %d tasks ' \
                         % (num_tasks) + 'with %d classes per task.' \
                         % (num_classes_per_task))

    print('Creating %d data handlers for SplitMNIST tasks ...' % num_tasks)

    handlers = []
    steps = num_classes_per_task
    for i in range(0, 10, steps):
        handlers.append(SplitMNIST(data_path, use_one_hot=use_one_hot,
            use_torch_augmentation=use_torch_augmentation,
            validation_size=validation_size, labels=range(i, i+steps),
            trgt_padding=trgt_padding))

        if len(handlers) == num_tasks:
            break

    print('Creating data handlers for SplitMNIST tasks ... Done')

    return handlers

class SplitMNIST(MNISTData):
    """An instance of the class shall represent a SplitMNIST task.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be represented in a
            one-hot encoding.
        validation_size (int): The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
        use_torch_augmentation (bool): See docstring of class
            :class:`data.mnist_data.MNISTData`.
        labels (list): The labels that should be part of this task.
        full_out_dim (bool): Choose the original MNIST instead of the new
            task output dimension. This option will affect the attributes
            :attr:`data.dataset.Dataset.num_classes` and
            :attr:`data.dataset.Dataset.out_shape`.
        trgt_padding (int, optional): If provided, ``trgt_padding`` fake classes
            will be added, such that in total the returned dataset has
            ``len(labels) + trgt_padding`` classes. However, all padded classes
            have no input instances. Note, that 1-hot encodings are padded to
            fit the new number of classes.
    """
    def __init__(self, data_path, use_one_hot=False, validation_size=1000,
                 use_torch_augmentation=False, labels=[0, 1],
                 full_out_dim=False, trgt_padding=None):
        # Note, we build the validation set below!
        super().__init__(data_path, use_one_hot=use_one_hot,
             use_torch_augmentation=use_torch_augmentation, validation_size=0)

        self._full_out_dim = full_out_dim

        if isinstance(labels, range):
            labels = list(labels)
        assert np.all(np.array(labels) >= 0) and \
               np.all(np.array(labels) < self.num_classes) and \
               len(labels) == len(np.unique(labels))
        K = len(labels)

        self._labels = labels

        train_ins = self.get_train_inputs()
        test_ins = self.get_test_inputs()

        train_outs = self.get_train_outputs()
        test_outs = self.get_test_outputs()

        # Get labels.
        if self.is_one_hot:
            train_labels = self._to_one_hot(train_outs, reverse=True)
            test_labels = self._to_one_hot(test_outs, reverse=True)
        else:
            train_labels = train_outs
            test_labels = test_outs

        train_labels = train_labels.squeeze()
        test_labels = test_labels.squeeze()

        train_mask = train_labels == labels[0]
        test_mask = test_labels == labels[0]
        for k in range(1, K):
            train_mask = np.logical_or(train_mask, train_labels == labels[k])
            test_mask = np.logical_or(test_mask, test_labels == labels[k])

        train_ins = train_ins[train_mask, :]
        test_ins = test_ins[test_mask, :]

        train_outs = train_outs[train_mask, :]
        test_outs = test_outs[test_mask, :]

        if validation_size > 0:
            if validation_size >= train_outs.shape[0]:
                raise ValueError('Validation set size must be smaller than ' +
                                 '%d.' % train_outs.shape[0])
            val_inds = np.arange(validation_size)
            train_inds = np.arange(validation_size, train_outs.shape[0])

        else:
            train_inds = np.arange(train_outs.shape[0])

        test_inds = np.arange(train_outs.shape[0],
                              train_outs.shape[0] + test_outs.shape[0])

        outputs = np.concatenate([train_outs, test_outs], axis=0)

        if not full_out_dim:
            # Transform outputs, e.g., if 1-hot [0,0,0,1,0,0,0,0,0,0] -> [0,1]

            # Note, the method assumes `full_out_dim` when later called by a
            # user. We just misuse the function to call it inside the
            # constructor.
            self._full_out_dim = True
            outputs = self.transform_outputs(outputs)
            self._full_out_dim = full_out_dim

            # Note, we may also have to adapt the output shape appropriately.
            if self.is_one_hot:
                self._data['out_shape'] = [len(labels)]

        images = np.concatenate([train_ins, test_ins], axis=0)

        ### Overwrite internal data structure. Only keep desired labels.

        # Note, we continue to pretend to be a 10 class problem, such that
        # the user has easy access to the correct labels and has the original
        # 1-hot encodings.
        if not full_out_dim:
            self._data['num_classes'] = len(labels)
        else:
            self._data['num_classes'] = 10
        self._data['in_data'] = images
        self._data['out_data'] = outputs
        self._data['train_inds'] = train_inds
        self._data['test_inds'] = test_inds
        if validation_size > 0:
            self._data['val_inds'] = val_inds

        n_val = 0
        if validation_size > 0:
            n_val = val_inds.size

        if trgt_padding is not None and trgt_padding > 0:
            print('SplitMNIST targets will be padded with %d zeroes.' \
                  % trgt_padding)
            self._data['num_classes'] += trgt_padding

            if self.is_one_hot:
                self._data['out_shape'] = [self._data['out_shape'][0] + \
                                           trgt_padding]
                out_data = self._data['out_data']
                self._data['out_data'] = np.concatenate((out_data,
                    np.zeros((out_data.shape[0], trgt_padding))), axis=1)

        print('Created SplitMNIST task with labels %s and %d train, %d test '
              % (str(labels), train_inds.size, test_inds.size) +
              'and %d val samples.' % (n_val))

    def transform_outputs(self, outputs):
        """Transform the outputs from the 10D MNIST dataset into proper labels
        based on the constructor argument ``labels``.

        I.e., the output will have ``len(labels)`` classes.

        Example:
            Split with labels [2,3]

            1-hot encodings: [0,0,0,1,0,0,0,0,0,0] -> [0,1]

            labels: 3 -> 1

        Args:
            outputs: 2D numpy array of outputs.

        Returns:
            2D numpy array of transformed outputs.
        """
        return _transform_split_outputs(self, outputs)

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'SplitMNIST'

if __name__ == '__main__':
    pass
