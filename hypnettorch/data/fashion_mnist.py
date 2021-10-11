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
# @title          :data/fashion_mnist.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :04/08/2020
# @version        :1.0
# @python_version :3.6.10
"""
Fashion-MNIST Dataset
---------------------

The module :mod:`data.fashion_mnist` contains a handler for the
`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`__ dataset.

The dataset was introduced in:

    Xiao et al., `Fashion-MNIST: a Novel Image Dataset for Benchmarking \
Machine Learning Algorithms <https://arxiv.org/abs/1708.07747>`__, 2017.

This module contains a simple wrapper from the corresponding
`torchvision dataset <https://pytorch.org/docs/master/torchvision/datasets.\
html#fashion-mnist>`__ to our dataset interface :class:`data.dataset.Dataset`.
"""
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import FashionMNIST

from hypnettorch.data.cifar10_data import CIFAR10Data
from hypnettorch.data.dataset import Dataset
from hypnettorch.data.mnist_data import MNISTData

class FashionMNISTData(Dataset):
    """An instance of the class shall represent the Fashion-MNIST dataset.

    Note:
        By default, input samples are provided in a range of ``[0, 1]``.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be
            represented in a one-hot encoding.
        validation_size (int): The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
        use_torch_augmentation (bool): Apply data augmentation to inputs when
            calling method :meth:`data.dataset.Dataset.input_to_torch_tensor`.

            The augmentation will be identical to the one provided by class
            :class:`data.mnist_data.MNISTData`, **except** that during training
            also random horizontal flips are applied.

            Note:
                If activated, the statistics of test samples are changed as
                a normalization is applied.
    """
    def __init__(self, data_path, use_one_hot=False, validation_size=0,
                 use_torch_augmentation=False):
        super().__init__()

        fmnist_train = FashionMNIST(data_path, train=True, download=True)
        fmnist_test = FashionMNIST(data_path, train=False, download=True)
        assert np.all(np.equal(fmnist_train.data.shape, [60000, 28, 28]))
        assert np.all(np.equal(fmnist_test.data.shape, [10000, 28, 28]))

        train_inputs = fmnist_train.data.numpy().reshape(60000, -1)
        test_inputs = fmnist_test.data.numpy().reshape(10000, -1)
        train_labels = fmnist_train.targets.numpy().reshape(60000, 1)
        test_labels = fmnist_test.targets.numpy().reshape(10000, 1)

        images = np.concatenate([train_inputs, test_inputs], axis=0)
        labels = np.concatenate([train_labels, test_labels], axis=0)

        # Scale images into a range between 0 and 1. Such that it is identical
        # to the default MNIST scale in `data.dataset.mnist_data`.
        images = images / 255

        val_inds = None
        train_inds = np.arange(train_labels.size)
        test_inds = np.arange(train_labels.size,
                              train_labels.size + test_labels.size)

        if validation_size > 0:
            if validation_size >= train_inds.size:
                raise ValueError('Validation set must contain less than %d ' \
                                 % (train_inds.size) + 'samples!')

            val_inds = np.arange(validation_size)
            train_inds = np.arange(validation_size, train_inds.size)

        # Bring everything into the internal structure of the Dataset class.
        self._data['classification'] = True
        self._data['sequence'] = False
        self._data['num_classes'] = 10
        self._data['is_one_hot'] = use_one_hot
        self._data['in_data'] = images
        self._data['in_shape'] = [28, 28, 1]
        self._data['out_shape'] = [10 if use_one_hot else 1]
        self._data['val_inds'] = val_inds
        self._data['train_inds'] = train_inds
        self._data['test_inds'] = test_inds

        if use_one_hot:
            labels = self._to_one_hot(labels)

        self._data['out_data'] = labels

        # Information specific to this dataset.
        assert np.all([fmnist_train.classes[i] == c for i, c in \
                       enumerate(fmnist_test.classes)])
        self._data['fmnist'] = dict()
        self._data['fmnist']['classes'] = fmnist_train.classes

        # Initialize PyTorch data augmentation.
        self._augment_inputs = False
        if use_torch_augmentation:
            self._augment_inputs = True
            self._train_transform, self._test_transform = \
                MNISTData.torch_input_transforms(use_random_hflips=True)

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'Fashion-MNIST'

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        Note, this method has been overwritten from the base class.

        If enabled via constructor option ``use_torch_augmentation``, input
        images are preprocessed.
        Preprocessing involves normalization and (for training mode) random
        perturbations.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor.
        """
        # FIXME Method is identical to the one used by the MNIST dataset.
        if self._augment_inputs and not force_no_preprocessing:
            if mode == 'inference':
                transform = self._test_transform
            elif mode == 'train':
                transform = self._train_transform
            else:
                raise ValueError('"%s" not a valid value for argument "mode".'
                                 % mode)

            return CIFAR10Data.torch_augment_images(x, device, transform,
                                                    img_shape=self.in_shape)

        else:
            return Dataset.input_to_torch_tensor(self, x, device,
                mode=mode, force_no_preprocessing=force_no_preprocessing,
                sample_ids=sample_ids)

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("Fashion-MNIST Sample")
        else:
            assert(np.size(outputs) == 1)
            label = np.asscalar(outputs)
            label_name = self._data['fmnist']['classes'][label]

            if predictions is None:
                ax.set_title('Sample with label:\n%s (%d)' % \
                             (label_name, label))
            else:
                if np.size(predictions) == self.num_classes:
                    pred_label = np.argmax(predictions)
                else:
                    pred_label = np.asscalar(predictions)
                pred_label_name = self._data['fmnist']['classes'][pred_label]

                ax.set_title('Label: %s (%d)\n' % (label_name, label) + \
                             'Prediction: %s (%d)' % (pred_label_name,
                                                      pred_label))

        #plt.subplots_adjust(wspace=0.5, hspace=0.4)

        ax.set_axis_off()
        ax.imshow(np.squeeze(np.reshape(inputs, self.in_shape)))
        fig.add_subplot(ax)

        if num_inner_plots == 2:
            ax = plt.Subplot(fig, inner_grid[1])
            ax.set_title('Predictions')
            bars = ax.bar(range(self.num_classes), np.squeeze(predictions))
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
        # FIXME code copied from MNISTData.
        plot_configs = super()._plot_config(inputs, outputs=outputs,
                                            predictions=predictions)
        
        if predictions is not None and \
                np.shape(predictions)[1] == self.num_classes:
            plot_configs['outer_hspace'] = 0.6
            plot_configs['inner_hspace'] = 0.4
            plot_configs['num_inner_rows'] = 2
            #plot_configs['num_inner_cols'] = 1
            plot_configs['num_inner_plots'] = 2

        return plot_configs

if __name__ == '__main__':
    pass


