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
# title           :data/timeseries/seq_smnist.py
# author          :be
# contact         :behret@ethz.ch
# created         :14/04/2020
# version         :1.0
# python_version  :3.7
"""
Sequence of Stroke MNIST Samples (SeqSMNIST) Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A data handler to generate a set of sequential stroke MNIST tasks for continual
learning. The used stroke MNIST data was already preprocessed with the script
:mod:`data.timeseries.preprocess_smnist` (see also the corresponding data
handler in :mod:`data.timeseries.smnist_data`).

**The task**

Given a sequence of two smnist digits of length ``n`` (e.g. ``2,5,5,2,2`` with
``n=5``), classify which of the ``2**n`` possible binary sequences (classes) the
presented sequence belongs to. E.g., for ``n=3`` the number of classes would be
8 (corresponding to all possible sequences with two digits (``0`` and ``1``
here): ``000, 001, 010, 100, 011, 110, 101, 111``.

The individual tasks of the task family differ in which digits are used to 
generate the binary sequences. Considering all possible pairs of digits we 
can generate (10**2-10) / 2 = 45 tasks.
"""
import numpy as np
import numpy.matlib as npm
import pickle
import os
import urllib.request
import itertools
from sklearn.preprocessing import OneHotEncoder

from hypnettorch.data.sequential_dataset import SequentialDataset

# generate sequences and classes
# generate dataset by sampling numbers from smnist
# write class that implements dataset interface
# write train file
# write hpsearch file

#def generate_tasks(self):


class SeqSMNIST(SequentialDataset):
    """Datahandler for one sequential stroke MNIST task (as described above).

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
        num_train (int) : Number of training samples to be generated.
        num_test (int) : Number of test samples to be generated.
        num_val (int) : Number of validation samples to be generated.
        target_per_timestep (bool): If activated, the one-hot
            encoding of the current image will be copied across the entire
            sequence. Else, there is a single target for the entire
            sequence (rather than one per timestep.
        sequence_length (int): The length of the binary sequence to be
            classified. This also affects the number of classes which is
            ``2**n``.
        digits (tuple): The two digits that shall be used for generating the
            binary sequence.
        two_class (bool): When true, instead of classifying each possible 
            sequence individually, sequences are randomly grouped into two 
            classes. This makes the number of classes (and therefore the chance 
            level) independent of the sequence length.
        upsample_control (bool): If ``True``, instead of building sequences
                of digits, we upsample single digits by a factor given by
                ``seq_len``.
        fix_class_partition (bool): TODO
        rseed (int): Seed for numpy random state.
    """
    def __init__(self, data_path, use_one_hot=True, num_train=1600,
                 num_test=400, num_val=0, target_per_timestep=True,
                 sequence_length=4, digits=(0,1), two_class=False,
                 upsample_control=False, fix_class_partition=False, rseed=None):
        super().__init__()

        # set random state
        if rseed is not None:
            self._rstate = np.random.RandomState(rseed)
        else:
            self._rstate = np.random


        self.target_per_timestep = target_per_timestep
        self.two_class = two_class

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

        smnist_n_train = 50000
        x_data_train = data[0][:smnist_n_train]
        y_data_train = data[1][:smnist_n_train]
        x_data_test = data[2]
        y_data_test = data[3]
        x_data_val = data[0][smnist_n_train:]
        y_data_val = data[1][smnist_n_train:]

        # generate data
        if two_class:
            num_classes = 2**sequence_length
            if sequence_length == 1:
                class_partition = np.asarray([1])
            else:
                # randomly group sequences into 2 classes
                class_partition = self._rstate.choice(num_classes,
                    int(num_classes/2), replace=False)
            if fix_class_partition:
                # use the same random partition for all tasks
                rstate_partition = np.random.RandomState(42)
                class_partition = rstate_partition.choice(num_classes,
                    int(num_classes/2), replace=False)
        else:
            class_partition = None

        #print(class_partition)

        max_seq_len = 117 * sequence_length
        x_train, y_train, sample_lengths_train = \
            self._generate_data(x_data_train, y_data_train, max_seq_len, digits,
                                sequence_length, num_train, use_one_hot,
                                class_partition, upsample_control)
        x_test, y_test, sample_lengths_test = \
            self._generate_data(x_data_test, y_data_test, max_seq_len, digits,
                                sequence_length, num_test, use_one_hot,
                                class_partition, upsample_control)
        x_val, y_val, sample_lengths_val = \
            self._generate_data(x_data_val, y_data_val, max_seq_len, digits,
                                sequence_length, num_val, use_one_hot,
                                class_partition, upsample_control)

        in_data = np.vstack((x_val,x_train,x_test))
        out_data = np.vstack((y_val,y_train,y_test))
        sample_lengths = np.hstack((sample_lengths_val, sample_lengths_train,
                                    sample_lengths_test))

        # Specify internal data structure.
        self._data['classification'] = True
        self._data['sequence'] = True
        if two_class:
            self._data['num_classes'] = 2
        else:
            self._data['num_classes'] = 2**sequence_length
        # Quadruple per timestep: (dx, dy, eos, eod).
        self._data['in_shape'] = [4] # Quatruple 
        self._data['out_shape'] = [self._data['num_classes'] if use_one_hot else 1]
        # Maximum number of timesteps, sequences will be padded to this length.
        self._data['num_time_steps'] = max_seq_len
        self._data['is_one_hot'] = use_one_hot
        self._data['in_data'] = in_data
        self._data['out_data'] = out_data

        if num_val > 0:
            self._data['val_inds'] = np.arange(num_val)
        self._data['train_inds'] = np.arange(num_val, num_val+num_train)
        self._data['test_inds'] = np.arange(num_val+num_train,
                                            num_val+num_train + num_test)

        self._data['in_seq_lengths'] = sample_lengths
        if target_per_timestep:
            self._data['out_seq_lengths'] = sample_lengths
        else:
            self._data['out_seq_lengths'] = np.ones_like(sample_lengths)

    def _generate_data(self, x_data, y_data, max_seq_len, digits, seq_len,
                       n_samples, use_one_hot, class_partition,
                       upsample_control):
        
        """Generates data for a single sequence stroke MNIST task with
        specified length and digits to use. 

        Args:
            x_data (list): Original stroke mnist input data, with every list
                entry being a numpy.ndarray of shape ``[4, stroke_seq_len]``
            y_data (list): Original stroke mnist labels, with every list 
                entry being a numpy.ndarray of shape ``[10]``
            max_seq_len (int): The maximum length of a sequence (i.e. number of
                        timesteps of a sample)
            digits (tuple): The two digits used to build the sequence
            seq_len (int): The length of the sequence of digits to build
            n_samples (int): The number of samples that should be generated
            use_one_hot (bool): Whether or not to use one hot encodings
            class_partition (list): If sequences should be grouped into 2
                different classes this list specifies the class partition.
            upsample_control (bool): See constructor docstring.

        Returns:
            (tuple): Tuple containing:

            - **in_data** (numpy.ndarray): Numpy array of shape
              ``[n_samples, max_num_time_steps * 4]``.
            - **out_data** (numpy.ndarray): Numpy array of shape
            ``[n_samples, max_num_time_steps]``.
            - **sample_lengths** (numpy.ndarray): The original unpadded sequence
              lengths.
        """
        # modify seq_len in case we do upsampling control
        if upsample_control:
            upsample_factor = seq_len
            seq_len = 1
            if not self.two_class:
                raise NotImplementedError()

        # construct all possible classes
        classes = ["".join(seq) for seq in \
                   itertools.product("01", repeat=seq_len)]

        # get the right number of samples per class to get a balanced data set
        # with the desired n_samples.
        num = n_samples
        div = len(classes)
        n_samples_per_class = [num // div + (1 if x < num % div else 0) \
                               for x in range (div)]

        # find indices of samples with the wanted digit class
        y_data = [np.argmax(y) for y in y_data]
        digit_idx = []
        digit_idx.append(np.where(np.asarray(y_data) == digits[0])[0])
        digit_idx.append(np.where(np.asarray(y_data) == digits[1])[0])

        # generate samples for every class
        samples = []
        labels = []
        for i,c in enumerate(classes):
            this_label = i
            digits_to_sample = [int(c[i]) for i in range(len(c))]
            for s in range(n_samples_per_class[i]):
                this_sample = None
                for d in digits_to_sample:
                    rand_idx = self._rstate.randint(len(digit_idx[d]))
                    sample_idx = digit_idx[d][rand_idx]
                    digit_sample = x_data[sample_idx]
                    if this_sample is None:
                        this_sample = digit_sample
                    else:
                        this_sample = np.vstack((this_sample,digit_sample)) 
                samples.append(this_sample)
                labels.append(this_label)

        # if configured sort labels into 2 classes
        labels = np.asarray(labels)
        if self.two_class and not upsample_control:
            lbl_mask = np.isin(labels, class_partition)
            labels[~lbl_mask] = 0
            labels[lbl_mask] = 1

        if upsample_control:
            for i,s in enumerate(samples):
                # Initial timestep is absolute start position of digit. To
                # translate to a higher resolution image, we can just multiply
                # the abolute position vby the scaling factor.
                upsample = s[0,:]*upsample_factor
                for t in np.arange(1,s.shape[0]):
                    # don't do upsampling at end of strokes or end of digits
                    if all((s[t,2] == 0, s[t,3] == 0)):
                        # Repeat original stroke "upsample_factor" times, such
                        # that the relative stroke length is identical if
                        # images are normalized to same resolution.
                        for k in range(upsample_factor):
                            upsample = np.vstack((upsample, s[t,:]))
                    else:
                        upsample = np.vstack((upsample, s[t,:]))
                samples[i] = upsample

        # structure output data
        out_data = labels.reshape(-1, 1)
        if use_one_hot:
            n_classes = 2**seq_len
            if self.two_class:
                n_classes = 2

            # FIXME We shouldn't call this method if the validation set size is
            # zero.
            if out_data.size == 0:
                out_data = np.matlib.repmat(out_data, 1, n_classes)
            else:
                # FIXME use internal method `_to_one_hot` and set required class
                # attributes beforehand.
                one_hot_encoder = OneHotEncoder(categories=[range(n_classes)])
                one_hot_encoder.fit(npm.repmat(np.arange(n_classes), 1, 1).T)
                out_data = one_hot_encoder.transform(out_data).toarray()

        if self.target_per_timestep:
            out_data = np.matlib.repmat(np.asarray(out_data), 1, max_seq_len)

        # structure input data
        in_data = np.zeros((n_samples,max_seq_len,4))
        sample_lengths = np.zeros(n_samples)
        for i,s in enumerate(samples):
            in_data[i,:s.shape[0],:] = s
            sample_lengths[i] = s.shape[0]

        in_data = self._flatten_array(in_data)

        return in_data, out_data, sample_lengths

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'Sequence SMNIST'

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None, sample_ids=None,
                     is_one_hot=None):
        raise NotImplementedError()

    def _plot_config(self, inputs, outputs=None, predictions=None):
        raise NotImplementedError()

    def __str__(self):
        """Print major characteristics of the current dataset."""
        return 'Data handler for sequential SMNIST'

if __name__ == '__main__':
    pass
