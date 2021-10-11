#!/usr/bin/env python3
# Copyright 2021 Christian Henning
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
# @title          :data/udacity_ch2.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :08/30/2021
# @version        :1.0
# @python_version :3.8.10
"""
Udacity Self-Driving Car Challenge 2 - Steering Angle Prediction
----------------------------------------------------------------

The module :mod:`.udacity_ch2` contains a handler for the
`Udacity Self-Driving Car Challenge 2 <https://medium.com/@maccallister.h/\
challenge-2-submission-guidelines-284ce6641c41#.az85snjmh>`__, which contains
imagery from a car's frontal center camera in combination with CAN recorded
steering angles (the actual dataset contains more information, but those
ingredients are enough for the steering angle prediction task).

.. note::
    In the current implementation, this handler will not download and extract
    the dataset for you. You have to do this manually by following the
    instructions of the README file (which is located in the same folder as this
    file).

When using PyTorch, this class will create dataset classes
(:class:`torch.utils.data.Dataset`) for you for the training, testing and
validation set. Afterwards, you can use these dataset instances to create data
loaders:

.. code-block:: python

    train_loader = torch.utils.data.DataLoader(
        udacity_ch2.torch_train, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True)

You should then use these Pytorch data loaders rather than class internal
methods to work with the dataset.

PyTorch data augmentation is applied as defined by the method
:meth:`UdacityCH2Data.torch_input_transforms`.
"""
from copy import deepcopy
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from warnings import warn

from hypnettorch.data.large_img_dataset import LargeImgDataset

class UdacityCh2Data(LargeImgDataset):
    """An instance of the class is representing the Udacity Ch2 dataset.

    The input data of the dataset will be strings to image files. The output
    data corresponds to steering angles.

    Note:
        The dataset has to be already downloaded and extracted before
        this method can be called. See the local README file for details.

    Args:
        data_path (str): Where should the dataset be read from? The dataset
            folder is expected to contain the subfolders ``Ch2_001`` (test set)
            and ``Ch2_002`` (train and validation set). See README for details.
        num_val (int): The number of validation samples. The validation set
            will be random subset of the training set. Validation samples are
            excluded from the training set!

            .. note::
                Validation samples use the same data augmentation pipeline
                as test samples.
    """
    def __init__(self, data_path, num_val=0):
        # 33808 is the number of training samples.
        assert num_val < 33808
        # We keep the full path to each image in memory, so we don't need to
        # tell the super class the root path to each image (i.e., samples
        # contain absolute not relative paths).
        super().__init__('')

        start = time.time()

        print('Reading Udacity Ch2 dataset ...')

        test_img_dir = os.path.join(data_path, 'Ch2_001')
        test_meta_fn = os.path.join(data_path, 'Ch2_001', 'interpolated.csv')
        train_img_dir = os.path.join(data_path, 'Ch2_002')
        train_meta_fn = os.path.join(data_path, 'Ch2_002', 'interpolated.csv')

        err_msg = 'Please follow the steps described in the file ' + \
            'data/README.md to download and extract the data.'
        if not os.path.exists(train_img_dir):
            raise FileNotFoundError('Training images not found in directory ' +
                train_img_dir + '.\n' + err_msg)
        elif not os.path.exists(test_img_dir):
            raise FileNotFoundError('Test images not found in ' +
                'directory ' + test_img_dir + '.\n' + err_msg)
        elif not os.path.exists(train_meta_fn):
            raise FileNotFoundError('Training meta data not found in ' +
                train_meta_fn + '.\n' + err_msg)
        elif not os.path.exists(test_meta_fn):
            raise FileNotFoundError('Test meta data not found in ' +
                test_meta_fn + '.\n' + err_msg)

        # Read dataset.
        self._process_dataset(test_img_dir, test_meta_fn, train_img_dir,
                              train_meta_fn, num_val)

        # Translate everything into the internal structure of this class.
        num_train = len(self.torch_train)
        num_test = len(self.torch_test)
        if num_val > 0:
            assert num_val == len(self.torch_val)
        num_samples = num_train + num_test + num_val
        # Just a sanity check, as these numbers should be fixed whenever the
        # full dataset is loaded.
        if num_test != 5614:
            warn('Udacity Ch2 should contain 5,614 test samples, ' +
                 'but %d samples were found!' % num_test)
        if num_train + num_val != 33808:
            warn('Udacity Ch2 should contain 33,808 training ' +
                 'samples, but %d samples were found!'
                 % (num_train + num_val))

        # Maximum string length of an image path.
        max_path_len = len(max(self.torch_train.samples +
            ([] if num_val == 0 else self.torch_val.samples) +
            self.torch_test.samples, key=lambda t : len(t[0]))[0])

        self._data['classification'] = False
        self._data['sequence'] = False

        self._data['in_shape'] = [192, 256, 3] # Height / Width / Channels
        self._data['out_shape'] = [1]

        self._data['in_data'] = np.chararray([num_samples, 1],
            itemsize=max_path_len, unicode=True)
        for i, (img_path, _) in enumerate(self.torch_train.samples +
                ([] if num_val == 0 else self.torch_val.samples) +
                self.torch_test.samples):
            self._data['in_data'][i, :] = img_path

        targets = np.array(self.torch_train.targets +
                           ([] if num_val == 0 else self.torch_val.targets) +
                           self.torch_test.targets).reshape(-1, 1)

        self._data['out_data'] = targets

        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train + num_val, num_samples)
        if num_val == 0:
            self._data['val_inds'] = None
        else:
            self._data['val_inds'] = np.arange(num_train, num_train + num_val)

        print('Dataset consists of %d training, %d validation and %d test '
              % (num_train, num_val, num_test) + 'samples.')

        end = time.time()
        print('Elapsed time to read dataset: %f sec' % (end-start))

    @property
    def test_angles_available(self):
        """Whether the test angles are available.

        Note:
            If not available, test angles will all be set to zero!

        The original dataset comes only with test images. However, the test set
        was later released too, which contains both images and angles. See the
        README for details.

        :type: bool
        """
        return self._test_angles_available

    def tf_input_map(self, mode='inference'):
        """Not impemented."""
        # Confirm, whether you wanna process data as in the baseclass or
        # implement a new image loader.
        raise NotImplementedError('Not implemented yet!')

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'UdacityCh2'

    def _process_dataset(self, test_img_dir, test_meta_fn, train_img_dir,
                         train_meta_fn, num_val):
        """Read and process the datasets using PyTorch its ImageFolder class.

        Additionally, this method splits the training set into train
        and validation set.

        The following attributes are added to the class:
            _torch_ds_train: A PyTorch Dataset class representing the training
                set.
            _torch_ds_test: A PyTorch Dataset class representing the validation
                set (corresponds to the dataset in "val_dir").
            _torch_ds_val: A PyTorch Dataset class representing the validation
                set (A subset of the training set).

        Args:
            (....): See docstring of constructor.
            test_img_dir: Path to test images.
            test_meta_fn: Path to test ``interpolated.csv`` file.
            train_img_dir: Path to training images.
            train_meta_fn: Path to training ``interpolated.csv`` file.
        """
        # Read raw dataset using the PyTorch ImageFolder class.
        train_transform, test_transform = \
            UdacityCh2Data.torch_input_transforms()
        ds_train = datasets.ImageFolder(train_img_dir, train_transform)
        ds_test = datasets.ImageFolder(test_img_dir, test_transform)
        ds_val = None

        ### Read interpolated steering angles ###
        test_meta = pd.read_csv(test_meta_fn)
        train_meta = pd.read_csv(train_meta_fn)

        # In case only the HMB_3_release bag is available, and not the actual
        # test set.
        self._test_angles_available = True
        if np.all(np.isclose(test_meta['angle'].to_numpy(), 0)):
            self._test_angles_available = False
            warn('No steering angles are available for the test data! Zeroes ' +
                 'will be filled in instead.')

        for df, ds, dd in zip([test_meta, train_meta],
                              [ds_test, ds_train],
                              [test_img_dir, train_img_dir]):
            lbl_dict = {}
            for index, row in df.iterrows():
                if row['frame_id'] != 'center_camera':
                    continue

                fn = os.path.join(dd, row['filename'])
                lbl_dict[fn] = row['angle']

            for ii, ss in enumerate(ds.samples):
                ds.samples[ii] = (ss[0], lbl_dict[ss[0]])
                ds.targets[ii] = lbl_dict[ss[0]]

        ### Split training set into train/val set.
        if num_val > 0:
            orig_samples = ds_train.samples
            ds_train.samples = None
            ds_train.imgs = None
            ds_train.targets = None

            ds_val = deepcopy(ds_train)
            ds_val.transform = test_transform
            assert ds_val.target_transform is None

            ds_train.samples = []
            ds_train.imgs = ds_train.samples
            ds_val.samples = []
            ds_val.imgs = ds_val.samples

            # Always same validation set!
            rstate = np.random.RandomState(42)
            train_ind, val_ind = train_test_split( \
                np.arange(len(orig_samples)),
                test_size=num_val, shuffle=True, random_state=rstate,
                stratify=None)

            ds_train.samples.extend([orig_samples[ii] for ii in train_ind])
            ds_val.samples.extend([orig_samples[ii] for ii in val_ind])

            ds_train.targets = [s[1] for s in ds_train.samples]
            ds_val.targets = [s[1] for s in ds_val.samples]

            for ds_obj in [ds_train, ds_val]:
                assert len(ds_obj.samples) == len(ds_obj.imgs) and \
                       len(ds_obj.samples) == len(ds_obj.targets)

        self._torch_ds_train = ds_train
        self._torch_ds_test = ds_test
        self._torch_ds_val = ds_val

    @staticmethod
    def torch_input_transforms():
        """Get data augmentation pipelines for Udacity Ch2 inputs.

        Returns:
            (tuple): Tuple containing:

                - **train_transform**: A transforms pipeline that resizes
                  images to 256 x 192 pixels and normalizes them.
                - **test_transform**: Similar to ``train_transform``.
        """
        # This normalization is taken from the imagenet datahandler.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.Resize((256, 192)),
            transforms.ToTensor(),
            #normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize((256, 192)),
            transforms.ToTensor(),
            #normalize,
        ])

        return train_transform, test_transform

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("UdacityCh2 Sample")
        else:
            circle = plt.Circle((50, 50), 30, color='b', fill=False)
            ax.add_patch(circle)

            assert np.size(outputs) == 1
            target = np.asscalar(outputs)

            # Note, the steering targets are 1/r (with r being the turning
            # radius), and not actual angles!
            trgt_angle = target
            ax.add_artist(lines.Line2D( \
                [50, 50 + 30*np.cos(-np.pi/2 - trgt_angle)],
                [50, 50 + 30*np.sin(-np.pi/2 - trgt_angle)],
                linewidth=2, color='b', linestyle='solid'))

            if predictions is None:
                ax.set_title('Target steering angle:\n%f' % (target))
            else:
                pred_trgt = np.asscalar(predictions)

                ax.set_title('Target steering angle:\n%f' % (target) + \
                             '\nPrediction: %f' % (pred_trgt))

                pred_angle = pred_trgt
                ax.add_artist(lines.Line2D( \
                    [50, 50 + 30*np.cos(-np.pi/2 - pred_angle)],
                    [50, 50 + 30*np.sin(-np.pi/2 - pred_angle)],
                    linewidth=2, color='g', linestyle='solid'))

        if inputs.size == 1:
            img = self.read_images(inputs)
        else:
            img = inputs

        ax.set_axis_off()
        ax.imshow(np.squeeze(np.reshape(img, self.in_shape)))
        fig.add_subplot(ax)

if __name__ == '__main__':
    pass


