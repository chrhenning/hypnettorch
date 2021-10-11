#!/usr/bin/env python3
# Copyright 2020 Maria Cervera
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
# title           :data/timeseries/mud_data.py
# author          :mc
# contact         :mariacer@ethz.ch
# created         :09/08/2020
# version         :1.0
# python_version  :3.7
"""
Multilingual universal Dependencies Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A data handler for the multilingual universal dependencies dataset:

    https://universaldependencies.org/

This dataset is a Part-of-Speech tagging dataset that assigns to each token in
a sentence one of a set of universal syntactic tags. We adapt this dataset
to a Continual Learning scenario by considering Part-of-Speech tagging in
different languages as different tasks.
"""
import numpy as np
import os
import pickle
import urllib.request
import torch

from hypnettorch.data.sequential_dataset import SequentialDataset

def get_mud_handlers(data_path, num_tasks=5):
    """This function instantiates ``num_tasks`` objects of the class
    :class:`MUDData` each of which will contain a PoS dataset for a different
    language.

    Args:
        data_path (str): See argument ``data_path`` of class
            :class:`data.timeseries.smnist_data.SMNISTData`. If not existing,
            the dataset will be downloaded into this folder.
        num_tasks (int, optional): The number of data handlers that should be
            returned by this function.

    Returns:
        (list): A list of data handlers, each corresponding to an object of
        class :class:`MUDData` object.
    """

    print('Creating %d data handlers for PoS tagging tasks ...' % num_tasks)

    # LOAD DATA
    # If dataset does not exist in dataset folder, download it.
    # FIXME Dropbox link might become invalid in the near future.
    data_path_ud = os.path.join(data_path, 'sequential/mud/mud_data_2_6.pickle')
    if not os.path.exists(data_path_ud):
        data_dir = os.path.dirname(data_path_ud)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        url = \
            "https://www.dropbox.com/s/9xjrtprc2mfxcla/mud_data_2_6.pickle?dl=1"

        try:
            u = urllib.request.urlopen(url)
            data = u.read()
            u.close()
        except:
            raise RuntimeError('Multilingual universal dependencies data ' +
                'cannot be downloaded. ' +
                'If you are working on the cluster, please manually '+
                'copy the pickled dataset into the following location: '
                '%s. ' % (data_path_ud) + 'If the dropbox link (%s) ' % url +
                'is invalid, please rebuild the dataset using the script ' +
                '"preprocess_mud.py".')

        with open(data_path_ud, "wb") as f:
            f.write(data)

    # load embedding data
    data_path_emb = os.path.join(data_path, 'sequential/mud/embeddings.pickle')
    if not os.path.exists(data_path_emb):
        url = "https://www.dropbox.com/s/e8itpz7uez5n3hc/embeddings.pickle?dl=1"
        try:
            u = urllib.request.urlopen(url)
            data = u.read()
            u.close()
        except:
            raise RuntimeError('Word embeddings cannot be downloaded. '+
                'If you are working on the cluster, please manually '+
                'copy the pickled dataset into the dataset folder.')

        with open(data_path_emb, "wb") as f:
            f.write(data)

    with open(data_path_ud, 'rb') as f:
        data = pickle.load(f)


    assert len(data) == 3
    data, vocab, tagset = data
    # `data[i]` is a tuple consisting of train, test and val set. Each of those
    # is a tuple consisting of an array encoding the sentences (contaning
    # vocabulary indices), an array containing the target outputs (containing
    # tagset indices) and an array containing the actual sentence lengths
    # (note, they are already all padded to the same length).
    # `vocab[i]` is a list representing the vocabulary of the language (task).
    # `tagset` is the PoS tagset shared among all tasks.

    if num_tasks > len(data):
        raise RuntimeError('Requested %d PoS tag dataset, but only %d ' \
                           % (num_tasks, len(data)) +'languages are ' +
                           'available!')

    handlers = []
    for task_id in range(num_tasks):
        dhandler = MUDData(data[task_id], vocabulary=vocab[task_id],
                           tagset=tagset)
        handlers.append(dhandler)

    print('Creating data handlers for PoS tasks ... Done')

    return handlers

class MUDData(SequentialDataset):
    """Datahandler for the multilingual universal dependencies dataset.

    Args:
        task_data: A preprocessed dataset structure. Please use function
            :func:`get_mud_handlers` to create instances of this class.
        vocabulary (list or tuple, optional): The vocabular, i.e., a list of
            words that allows us to decode input sentences.
        tagset (list or tuple, optional): The PoS tagset.
    """
    def __init__(self, task_data, vocabulary=None, tagset=None):
        super().__init__()

        self.target_per_timestep = True

        self._vocab = vocabulary
        self._tagset = tagset

        # select the task and structure train/test/val data
        data = task_data
        num_time_steps = data[0][0].shape[1]
        num_samples = np.sum([x[0].shape[0] for x in data])
        in_shape = 1
        out_shape = 17

        in_data = np.zeros((num_time_steps,num_samples))
        out_data = np.zeros((num_time_steps,num_samples))
        seq_lengths = np.zeros(num_samples)


        # concat train test and val and save indices
        inds = []
        sample_count = 0

        for d in data:
            # d contains words, tags, lens
            words, tags, lens = d
            idx =  np.arange(sample_count, sample_count+words.shape[0])
            sample_count += words.shape[0]
            inds.append(idx)
            in_data[:,idx] = words.T
            out_data[:,idx] = tags.T
            seq_lengths[idx] = lens

        # Set attributes
        self._data['classification'] = True
        self._data['sequence'] = True
        self._data['num_classes'] = 17
        self._data['in_shape'] = [in_shape]
        self._data['out_shape'] = [out_shape]
        # Maximum number of timesteps, sequences will be padded to this length.
        self._data['num_time_steps'] = num_time_steps
        self._data['is_one_hot'] = True
        self._data['in_data'] = in_data.transpose()
        self._data['out_data'] = self._to_one_hot(out_data.T)
        self._data['train_inds'] = inds[0]
        self._data['test_inds'] = inds[1]
        self._data['val_inds'] = inds[2]
        self._data['in_seq_lengths'] = seq_lengths
        self._data['out_seq_lengths'] = seq_lengths

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None, **kwargs):
        raise NotImplementedError()

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'Multilingual Universal Dependencies dataset.'

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.
        
        Note:
            If ``sample_ids`` are provided, then padding will be reduced
            according to the sample within the minibatch with the longest
            sequence length.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.LongTensor): See docstring of method
            :meth:`data.sequential_dataset.SequentialDataset.\
input_to_torch_tensor`.
        """
        out_tensor = self._flatten_array(x, ts_dim_first=True, reverse=True,
                                         feature_shape=self.in_shape)
        y = torch.from_numpy(out_tensor).long().to(device)

        if sample_ids is not None:
            max_sl = int(self.get_in_seq_lengths(sample_ids).max())
            y = y[:max_sl, :, :]
            assert len(y.shape) == 3

        return y

    def output_to_torch_tensor(self, y, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):
        """Identical to method :meth:`data.sequential_dataset.\
SequentialDataset.output_to_torch_tensor`.

        However, if ``sample_ids`` are provided, then the same padding behavior
        as elicited by method :meth:`input_to_torch_tensor` is performed.
        """
        y = SequentialDataset.output_to_torch_tensor(self, y, device,
            mode=mode, force_no_preprocessing=force_no_preprocessing,
            sample_ids=sample_ids)

        if sample_ids is not None:
            max_sl = int(self.get_out_seq_lengths(sample_ids).max())
            y = y[:max_sl, :, :]
            assert len(y.shape) == 3

        return y

    def decode_batch(self, inputs, outputs, sample_ids=None):
        """Decode a batch of input and output samples into strings.

        This method translates a batch of input and output sequences (consisting
        of vocabulary and tagset indices) into actual sentences consisting of
        strings.
        
        Note:
            This method is only applicable if ``vocabulary`` and ``tagset``
            were provided to the constructor.

        Args:
            inputs (numpy.ndarray or torch.Tensor): Input samples as provided to
                or returned from method :meth:`input_to_torch_tensor`.
            outputs (numpy.ndarray or torch.Tensor): Output samples as provided
                to or returned from method :meth:`output_to_torch_tensor`.
            sample_ids (numpy.ndarray): See method
                :meth:`train_ids_to_indices`. If provided, the returned
                sentences are cropped to the actual sequence length.

        Returns:
            (tuple): Tuple containing:

            - **in_words** (list): List of list of strings, where each string
              corresponds to a word in the corresponding input sentence of
              ``inputs``.
            - **out_tags** (list): List of list of strings, where each string
              corresponds to the output tag corresponding to the tag ID read
              from ``outputs``.
        """
        if self._tagset is None or self._vocab is None:
            raise RuntimeError('Method only callable if "tagset" and "vocab" ' +
                               'were passed to the constructor.')

        if sample_ids is not None:
            # Note, input and output sequences have the same length.
            seq_lengths = self.get_out_seq_lengths(sample_ids)

        if len(inputs.shape) == 2:
            assert outputs.shape == 2

            inputs = self.input_to_torch_tensor(inputs, 'cpu', mode='inference')
            outputs = self.output_to_torch_tensor(outputs, 'cpu',
                                                  mode='inference')

        assert len(inputs.shape) == len(outputs.shape) == 3 and \
            inputs.shape[1] == outputs.shape[1]

        assert inputs.shape[2] == 1
        if self.is_one_hot:
            assert outputs.shape[2] == len(self._tagset)
            if isinstance(outputs, np.ndarray):
                outputs = outputs.argmax(axis=2)
            else:
                _, outputs = outputs.max(dim=2)

        ret_ins = []
        ret_outs = []

        for bid in range(inputs.shape[1]):
            l = inputs.shape[0]
            if sample_ids is not None:
                l = int(seq_lengths[bid])

            ret_ins.append( \
                [self._vocab[int(i)] for i in inputs[:l, bid, 0].tolist()])
            ret_outs.append( \
                [self._tagset[int(i)] for i in outputs[:l, bid].tolist()])

        return ret_ins, ret_outs

if __name__=='__main__':
    pass
