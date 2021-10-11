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
# title           :data/timeseries/preprocess_smnist.py
# author          :be
# contact         :behret@ethz.ch
# created         :23/03/2020
# version         :1.0
# python_version  :3.7
"""
Script to preprocess and structure the stroke mnist data set, which can then be
used via :class:`data.timeseries.smnist_data.SMNISTData`.

The result of this script is available at

    https://www.dropbox.com/s/sadzc8qvjvexdtx/ss_mnist_data?dl=0

If you want to recreate or modify this dataset, download the SMNIST data from

    https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/

and extract the tar.gz into the following folder:

    ``datasets/sequential/smnist/smnist_download``.

Subsequently executing this script will create a pickle file containing the
SMNIST dataset used in this study.

After extracting the `tar.gz` file, we have 3 files for every image:
- `trainimg-xxxxx-inputdata`
- `trainimg-xxxxx-targetdata`
- `trainimg-xxxxx-points`

We only use some parts of the data (all of input data and some parts of
target data).
"""
import numpy as np
import os
import pickle
from warnings import warn

warn('The script was created for one time usage and has to be adapted when ' +
     'reusing it.')

download_dir = '../../datasets/sequential/smnist/smnist_download'
target_path = '../../datasets/sequential/smnist/ss_mnist_data.pickle'

if not os.path.exists(download_dir):
    raise RuntimeError('Pathnames have to be adapted manually before using ' +
                       'the script.')

### Load and structure data from train files.
x_train = []
y_train = []
data_dir = os.path.join(download_dir,'sequences')
for i in range(60000):
    fpath = os.path.join(data_dir,'trainimg-' + str(i) + '-inputdata.txt')
    with open(fpath, 'r') as file:
        data = file.read()

    data = data.replace('\n',' ')
    data = np.asarray([int(d) for d in data.split()])
    data = np.reshape(data,(int(len(data)/4),4))
    x_train.append(data)

    # Targets
    fpath = os.path.join(data_dir,'trainimg-' + str(i) + '-targetdata.txt')
    with open(fpath, 'r') as file:
        data = file.read()
    data = data[:20]
    data = np.asarray([int(d) for d in data.split()])
    y_train.append(data)

### Load and structure data from test files.
x_test = []
y_test = []
for i in range(10000):
    fpath = os.path.join(data_dir,'testimg-' + str(i) + '-inputdata.txt')
    with open(fpath, 'r') as file:
        data = file.read()
        
    data = data.replace('\n',' ')
    data = np.asarray([int(d) for d in data.split()])
    data = np.reshape(data,(int(len(data)/4),4))
    x_test.append(data)

    # targets
    fpath = os.path.join(data_dir,'testimg-' + str(i) + '-targetdata.txt')
    with open(fpath, 'r') as file:
        data = file.read()
    data = data[:20]
    data = np.asarray([int(d) for d in data.split()])
    y_test.append(data)

### Save everything.
with open(target_path, 'wb') as f:
        pickle.dump([x_train, y_train, x_test, y_test], f)
