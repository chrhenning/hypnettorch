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
# @title          :data/svhn_data.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :04/08/2020
# @version        :1.0
# @python_version :3.6.10
"""
Street View House Numbers (SVHN) Dataset
----------------------------------------

The module :mod:`data.svhn_data` contains a handler for the
`SVHN <http://ufldl.stanford.edu/housenumbers>`__ dataset.

The dataset was introduced in:

    Netzer et al., `Reading Digits in Natural Images with Unsupervised Feature \
Learning <http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf>`__,
    2011.

This module contains a simple wrapper from the corresponding
`torchvision <https://pytorch.org/docs/master/torchvision/datasets.html#svhn>`__
class :class:`torchvision.datasets.SVHN` to our dataset interface
:class:`data.dataset.Dataset`.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.datasets import SVHN

from hypnettorch.data.cifar10_data import CIFAR10Data
from hypnettorch.data.dataset import Dataset

class SVHNData(Dataset):
    """An instance of the class shall represent the SVHN dataset.

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
        use_torch_augmentation (bool): Note, this option currently only applies
            to input batches that are transformed using the class member
            :meth:`input_to_torch_tensor` (hence, **only available for
            PyTorch**, so far).

            The augmentation will be identical to the one provided by class
            :class:`data.cifar10_data.CIFAR10Data`, **except** that during
            training no random horizontal flips are applied.

            Note:
                If activated, the statistics of test samples are changed as
                a normalization is applied (identical to the of class
                :class:`data.cifar10_data.CIFAR10Data`).
        use_cutout (bool): Whether option ``apply_cutout`` should be set of
            method :meth:`torch_input_transforms`. We use cutouts of size
            ``20 x 20`` as recommended
            `here <https://arxiv.org/pdf/1708.04552.pdf>`__.

            Note:
                Only applies if ``use_data_augmentation`` is set.
        include_train_extra (bool): The training dataset can be extended by
            "531,131 additional, somewhat less difficult samples" (see
            `here <http://ufldl.stanford.edu/housenumbers>`__).

            Note, as long as the validation set size is smaller than the
            original training set size, all validation samples would be taken
            from the original training set (and thus not contain those "less
            difficult" samples).
    """
    # In which subfolder of the datapath should the data be stored.
    _SUBFOLDER = 'SVHN'

    def __init__(self, data_path, use_one_hot=False, validation_size=0,
                 use_torch_augmentation=False, use_cutout=False,
                 include_train_extra=False):
        super().__init__()

        # Actual data path
        data_path = os.path.join(data_path, SVHNData._SUBFOLDER)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        svhn_train = SVHN(data_path, split='train', download=True)
        svhn_test = SVHN(data_path, split='test', download=True)
        svhn_extra = None
        if include_train_extra:
            svhn_extra = SVHN(data_path, split='extra', download=True)

        assert np.all(np.equal(svhn_train.data.shape, [73257, 3, 32, 32]))
        assert np.all(np.equal(svhn_test.data.shape, [26032, 3, 32, 32]))
        assert not include_train_extra or \
               np.all(np.equal(svhn_extra.data.shape, [531131, 3, 32, 32]))

        train_inputs = svhn_train.data
        test_inputs = svhn_test.data
        train_labels = svhn_train.labels
        test_labels = svhn_test.labels
        if include_train_extra:
            train_inputs = np.concatenate([train_inputs, svhn_extra.data],
                                          axis=0)
            train_labels = np.concatenate([train_labels, svhn_extra.labels],
                                          axis=0)

        images = np.concatenate([train_inputs, test_inputs], axis=0)
        labels = np.concatenate([train_labels, test_labels], axis=0)

        # Note, images are currently encoded in a way, that their shape
        # corresponds to (3, 32, 32). For consistency reasons, we would like to
        # change that to (32, 32, 3).
        images = np.rollaxis(images, 1, 4)
        # Scale images into a range between 0 and 1.
        images = images / 255.

        images = images.reshape(-1, 32 * 32 * 3)
        labels = labels.reshape(-1, 1)

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
        self._data['in_shape'] = [32, 32, 3]
        self._data['out_shape'] = [10 if use_one_hot else 1]
        self._data['val_inds'] = val_inds
        self._data['train_inds'] = train_inds
        self._data['test_inds'] = test_inds

        if use_one_hot:
            labels = self._to_one_hot(labels)

        self._data['out_data'] = labels

        # Dataset specific attributes.
        self._data['svhn'] = dict()
        # 0 - original train, 1 - extra train, 2 - test
        # Note, independent of whether samples are now in the validation set.
        self._data['svhn']['type'] = np.zeros(self._data['in_data'].shape[0])
        if include_train_extra:
            self._data['svhn']['type'][svhn_train.labels.size:] = 1
        self._data['svhn']['type'][test_inds] = 2

        # Initialize PyTorch data augmentation.
        self._augment_inputs = False
        if use_torch_augmentation:
            self._augment_inputs = True
            # Note, horizontal flips change the meaning of digits!
            self._train_transform, self._test_transform = \
                CIFAR10Data.torch_input_transforms(apply_rand_hflips=False,
                    apply_cutout=use_cutout, cutout_length=20)

        print('Created %s.' % (str(self)))

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'SVHN'

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        Note, this method has been overwritten from the base class.

        The input images are preprocessed if data augmentation is enabled.
        Preprocessing involves normalization and (for training mode) random
        perturbations.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor.
        """
        # FIXME Method copied from `CIFAR100Data`.
        if self._augment_inputs and not force_no_preprocessing:
            if mode == 'inference':
                transform = self._test_transform
            elif mode == 'train':
                transform = self._train_transform
            else:
                raise ValueError('"%s" not a valid value for argument "mode".'
                                 % mode)

            return CIFAR10Data.torch_augment_images(x, device, transform)

        else:
            return Dataset.input_to_torch_tensor(self, x, device,
                mode=mode, force_no_preprocessing=force_no_preprocessing,
                sample_ids=sample_ids)

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None, batch_ids=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.

        Args:
            batch_ids (numpy.ndarray, optional): If provided, then samples
                stemming from the "extra" training set will be marked in the
                caption.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        lbl = 'SVHN sample'
        if batch_ids is not None:
            stype = self._data['svhn']['type'][batch_ids[ind]]
            if stype == 1:
                lbl = 'SVHN (extra) sample'

        if outputs is None:
            ax.set_title(lbl)
        else:
            assert(np.size(outputs) == 1)
            label = np.asscalar(outputs)

            if predictions is None:
                ax.set_title('%s\nLabel: %d' % (lbl, label))
            else:
                if np.size(predictions) == self.num_classes:
                    pred_label = np.argmax(predictions)
                else:
                    pred_label = np.asscalar(predictions)

                ax.set_title('%s\nLabel: %d, Prediction: %d' % \
                             (lbl, label, pred_label))

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
        plot_configs = Dataset._plot_config(self, inputs, outputs=outputs,
                                            predictions=predictions)

        if predictions is not None and \
                np.shape(predictions)[1] == self.num_classes:
            plot_configs['outer_hspace'] = 0.6
            plot_configs['inner_hspace'] = 0.4
            plot_configs['num_inner_rows'] = 2
            #plot_configs['num_inner_cols'] = 1
            plot_configs['num_inner_plots'] = 2

        return plot_configs

    def __str__(self):
        return 'SVHN Dataset with %d training, %d validation and %d test ' % \
            (self.num_train_samples, self.num_val_samples,
             self.num_test_samples) + 'samples'

if __name__ == '__main__':
    pass


