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
# title           :data/timeseries/permuted_copy.py
# author          :mc
# contact         :mariacer@ethz.ch
# created         :12/05/2020
# version         :1.0
# python_version  :3.7
"""
Permuted Copy Dataset
^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.timeseries.permuted_copy` contains a data handler for the
permuted Copy Task dataset.
"""
import copy
import numpy as np

from hypnettorch.data.timeseries.copy_data import CopyTask

class PermutedCopyList():
    """A list of permuted Copy tasks that only uses a single instance of class
    :class:`PermutedCopy`.

    This class is inspired by 
    :class:`data.special.permuted_mnist.PermutedMNISTList`. For explanations
    and instructions on how to use, please refer to its documentation.
    Always keep in mind that this class emulates a Python list that holds
    objects of class :class:`PermutedCopy`. However, it doesn't actually hold
    several objects, but only one with just the permutation matrix being
    exchanged everytime a different element of this list is retrieved.
    Therefore, **use this class with care**!

    Caution:
        **You may never use more than one entry of this class at the same
        time**, as all entries share the same underlying data object and
        therewith the same permutation.

    Args:
        (....): See docstring of constructor of class :class:`PermutedMNIST`.
        permutations: A list of permutations (see parameter ``permutation``
            of class :class:`PermutedCopy` to have a description of valid list
            entries). The length of this list denotes the number of tasks.
                                              
            May also be a list of lists of permutations in case option
            ``permute_xor_separate`` of class
            :class:`data.timeseries.copy_data.CopyTask`.
        scatter_steps_list: A list of scatter steps to be used for the
            reconstruction of the output if ``scatter_pattern`` is active.
        show_perm_change_msg: Whether to print a notification everytime the
            data permutation has been exchanged. This should be enabled
            during developement such that a proper use of this list is
            ensured. **Note** You may never work with two elements of this
            list at a time.
    """
    def __init__(self, permutations, input_len, scatter_steps_list=None,
                show_perm_change_msg=True, **kwargs):

        print('Loading CopyTask into memory, that is shared among ' + \
            '%d permutation tasks.' % (len(permutations)))

        self._data = PermutedCopy(input_len, **kwargs)

        self._permutations = permutations
        self._scatter_steps = scatter_steps_list

        if scatter_steps_list is not None:
            assert len(permutations) == len(scatter_steps_list)

        self._show_perm_change_msg = show_perm_change_msg

        # To ensure that we do not disturb the randomness inside each Dataset
        # object, we store the corresponding batch generators internally.
        # In this way, we don't break the randomness used to generate batches
        # (or the order for deterministically retrieved minibatches, such as
        # test batches).
        self._batch_gens_train = [None] * len(permutations)
        self._batch_gens_test = [None] * len(permutations)
        self._batch_gens_val = [None] * len(permutations)

        # Sanity check! Assert that the implementation inside the `Dataset`
        # class hasn't changed.
        assert hasattr(self._data, '_batch_gen_train') and \
            self._data._batch_gen_train is None
        assert hasattr(self._data, '_batch_gen_test') and \
            self._data._batch_gen_test is None
        assert hasattr(self._data, '_batch_gen_val') and \
            self._data._batch_gen_val is None

        # Index of the currently active permutation.
        self._active_perm = -1

    def __len__(self):
        """Number of tasks."""
        return len(self._permutations)

    def __getitem__(self, index):
        """Return the underlying data object with the index'th permutation.

        Args:
            index: Index of task for which data should be returned.

        Return:
            The data loader for task ``index``.
        """
        ### User Warning ###
        color_start = '\033[93m'
        color_end = '\033[0m'
        help_msg = 'To disable this message, disable the flag ' + \
            '"show_perm_change_msg" when calling the constructor of class ' + \
            'data.timeseries.permuted_copy.PermutedCopyList.'
        ####################

        if isinstance(index, slice):
            new_list = copy.copy(self)
            new_list._permutations = self._permutations[index]
            new_list._batch_gens_train = self._batch_gens_train[index]
            new_list._batch_gens_test = self._batch_gens_test[index]
            new_list._batch_gens_val = self._batch_gens_val[index]
            if self._scatter_steps is not None:
                new_list._data['scatter_steps'] = self._scatter_steps[index]

            ### User Warning ###
            if self._show_perm_change_msg:
                indices = list(range(*index.indices(len(self))))
                print(color_start + 'data.timeseries.permuted_copy.' +
                      'PermutedCopyList: A slice of permutations with ' +
                      'indices %s has been created. ' % indices +
                      'The applied permutation has not changed! ' + color_end +
                      help_msg)
            ####################

            return new_list

        assert(isinstance(index, int) or isinstance(index, np.int64))

        # Backup batch generator to preserve random behavior.
        if self._active_perm != -1:
            self._batch_gens_train[self._active_perm] = \
                self._data._batch_gen_train
            self._batch_gens_test[self._active_perm] = \
                self._data._batch_gen_test
            self._batch_gens_val[self._active_perm] = self._data._batch_gen_val

        self._data._permutation = self._permutations[index]
        self._data._batch_gen_train = self._batch_gens_train[index]
        self._data._batch_gen_test = self._batch_gens_test[index]
        self._data._batch_gen_val = self._batch_gens_val[index]
        self._data._data['task_id']  = index
        self._active_perm = index
        if self._scatter_steps is not None:
            self._data._data['scatter_steps'] = self._scatter_steps[index]

        ### User Warning ###
        if self._show_perm_change_msg:
            color_start = '\033[93m'
            color_end = '\033[0m'

            print(color_start + \
                  'data.timeseries.permuted_copy.PermutedCopyList:' + \
                  ' Data permutation has been changed to %d. ' % index + \
                  color_end + help_msg)
        ####################

        return self._data

    def __setitem__(self, key, value):
        """Not implemented."""
        raise NotImplementedError('Not yet implemented!')

    def __delitem__(self, key):
        """Not implemented."""
        raise NotImplementedError('Not yet implemented!')


class PermutedCopy(CopyTask):
    """An instance of this class shall represent the permuted Copy Task dataset,
    which corresponds to the CopyTask dataset, where all sequences have
    identical lengths within and across tasks, and where tasks differ according
    to a given permutation of the outputs. Note that the inputs are always 
    comparable, so it does not make sense to consider CL3 scenarios with this
    task.

    Note:
        Image transformations are computed on the fly when transforming batches
        to torch tensors. Hence, this class is only applicable to PyTorch
        applications. Internally, the class stores the unpermuted sequences.

    Attributes:
        permutation: The permuation matrix that is applied to input sequences.
            before they are transformed to Torch tensors.

    Args:
        input_len (int): The length of the inputs.
        task_id (int): The id of the current task.
        permutation: The permutation that should be applied to the dataset.
            If ``None``, no permutation will be applied. We expect a numpy
            permutation of the form
            :code:`np.random.permutation(input_len*seq_width)`, where
            `input_len` and `seq_width` correspond to the sequence lenght and 
            width.
        scatter_steps (list): The input timesteps to be used for reconstructing
            the output.
        (....): See docstring of class :class:`CopyTask`.
    """
    def __init__(self, input_len, task_id=None, permutation=None,
                 scatter_steps=None, permute_xor=False, permute_xor_iter=None,
                 permute_xor_separate=False, **kwargs):
        # FIXME If `scatter_pattern` is True, the super class will now create
        # `scatter_steps` that are gonna be overwritten below.
        # We should pass `scatter_steps=False` instead and set
        # `self._data['scatter_steps']` to True below.
        super().__init__(input_len, input_len, permute_time=False,
                         permute_width=False, permute_xor=False, **kwargs)
        print('Creating data handler for PermutedCopy ...')

        self._permutation = permutation # See setter below.
        self._permute_xor = permute_xor
        if permute_xor_iter is None:
            self._permute_xor_iter = 1
        else:
            self._permute_xor_iter = permute_xor_iter
        self._permute_xor_separate = permute_xor_separate
        if scatter_steps is not None:
            assert 'scatter_pattern' in kwargs.keys() and \
                kwargs['scatter_pattern']
        self._data['scatter_steps'] = scatter_steps

    @property
    def permutation(self):
        """Getter for attribute :attr:`permutation`"""
        return self._permutation

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'PermutedCopy'


if __name__ == '__main__':
    pass


