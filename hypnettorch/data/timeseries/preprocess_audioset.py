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
# title           :data/timeseries/preprocess_audioset.py
# author          :be
# contact         :behret@ethz.ch
# created         :31/03/2020
# version         :1.0
# python_version  :3.7
"""
Script to structure the audioset dataset, which can then be used via
:class:`data.timeseries.audioset_data.AudiosetData`.

The result of this script is available at

    https://www.dropbox.com/s/07dfeeuf5aq4w1h/audioset_data_balanced?dl=0

If you want to recreate or modify this dataset, download the audioset data from

    https://research.google.com/audioset/download.html

and extract the tar.gz into the following folder:

    ``datasets/sequential/audioset/audioset_download``.

Subsequently executing this script will create a pickle file containing the 100
class subset of audioset used in this study. 

The dataset is stored in tensorflow files. Since we work with pytorch and there 
is no utility to read tensorflow files, we extract the data and safe them as
numpy arrays in a pickle file.

Furthermore the data are preprocessed to fit our continual learning experiments.
The original dataset provides three subsets with different compositions of
samples and classes. Since we only work with a subset of classes and samples, 
we load all available data and then filter and structure them according to our 
criteria.

We use the same criteria as Kemker et al. Classes and samples are restricted in
the following way:
    Classes:
        - no restriction according to ontology file (parsed from ontology.json)
        - no parent / child relationship (parsed from ontology.json)
        - confidence level > 70% (data was copied from website into txt file)
        - number of samples: we only take classes that have more samples than
        a certain threshold
    Samples:
        - since samples can have multiple labels, we only use samples which 
        only belong to one of the classes we use
        - we exclude samples that don't have the full length of 10 seconds

The chosen classes and samples are then split into train and test data and 
saved to a pickle file.
"""
import numpy as np
import pickle
import tensorflow as tf
import os
import json
from warnings import warn

warn('The script was created for one time usage and has to be adapted when ' +
     'reusing it. All paths specified here are absolute.')

# Tensorflow eager mode needs to be enabled for dataset mapping to work!
tf.enable_eager_execution()

# Set paths and parameters
data_dir = '../../datasets/sequential/audioset/'
download_dir = os.path.join(data_dir,'audioset_download')
fpath_conf_data = os.path.join(data_dir, 'confidence_data.csv')
fpath_label_inds = os.path.join(data_dir, 'class_labels_indices.csv')
fpath_ontology = os.path.join(data_dir, 'ontology.json')
target_path = os.path.join(data_dir, 'audioset_data_balanced.pickle')

n_classes = 100
n_sample = 1000
test_frac = 0.20


### Load data by serializing files and applying decode function.
def decode(serialized_example):
    """Decode data from TFRecord files.

    Args:
        serialized_example: serialized_example as created by 
        tf.data.TFRecordDataset

    Returns:
        (tuple): Tuple containing:

        - **audio** (numpy.ndarray): Array of shape (10,128) representing one 
        sample with 10 timesteps and 128 features
        - **label** (numpy.ndarray): Array of shape (1,) containing the class 
        of the corresponding sample
    """

    sequence_features = {
          'audio_embedding': tf.FixedLenSequenceFeature([], tf.string),
          }

    context_features = {
          'start_time_seconds': tf.FixedLenFeature([], tf.float32),
          'labels': tf.VarLenFeature(dtype=tf.int64),
        }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
      serialized_example,
      sequence_features=sequence_features,
      context_features=context_features
      )
    audio = tf.decode_raw(sequence_parsed['audio_embedding'], tf.uint8)
    label = tf.cast(context_parsed['labels'], tf.int64)

    return audio, label

# Apply decode function to all dataset entries using map function.
# Take files from all three data sets since we repartition anyway.
fpaths = []
for path, subdirs, files in os.walk(download_dir):
    for name in files:
        if 'tfrecord' in name:
            fpaths.append(os.path.join(path, name))

# Create dataset and decode
dataset = tf.data.TFRecordDataset(fpaths)
dataset = dataset.map(decode)

# Extract data to lists
x = []
y = []
for d in dataset:
    x.append(d[0].numpy())
    y.append(tf.sparse.to_dense(tf.sparse.reorder(d[1])).numpy())


### Filter classes as described above.  
# Parse confidence values
conf_data = {}
with open(fpath_conf_data) as f:
    for line in f:
        tokens = line.split() 
        # parse confidence
        c = 0
        for t in tokens:
            if t.find('%') is not -1:
                c = int(t[:-1])
        # parse class name
        n = ''
        for t in tokens:
            if t.find('%') == -1 and t != '-':
                if n == '':
                    n = t
                else:
                    n = n+' '+t
            else:
                break
        conf_data.update({n:c})
    
# Parse class numbers from label csv file
l = -1
csv_data = {}
with open(fpath_label_inds) as f:
    for line in f:
        if l == -1:
            l += 1
            continue
        tokens = line.split('"')
        n = tokens[1]
        csv_data.update({n:l})
        l +=1

# Parse ontology info from json file
with open(fpath_ontology, 'r') as f:
    json_data = json.load(f)
    
# Put all data into a single list.
all_data = []    
for j in json_data:
    if j['name'] in conf_data.keys():
        class_info = {
                'name' : j['name'],
                'restricted' : j['restrictions'] != [],
                'has_child' : j['child_ids'] != [],
                'conf' : conf_data[j['name']],
                'id' : csv_data[j['name']]
        }
        all_data.append(class_info)
        
# Filter classes
classes = []
for c in all_data:
    if not c['restricted'] and not c['has_child'] and c['conf'] >= 70:
        classes.append(c['id'])


### Filter the samples.
# Find samples that belong to only one of the potential classes.
# We also exclude some samples that don't have data for the full 10 seconds.
# First discard labels that are not in the set of potential classes
y_fil = []
for i in range(len(y)):
    y_fil.append( np.intersect1d(y[i],classes))

# Find samples with one label    
n_labels = np.asarray([len(y) for y in y_fil])
single_label_idx = np.where(n_labels == 1)[0]

# Find samples that are shorter than 10 seconds (to be excluded)
too_short = np.where(np.asarray([x.shape[0] for x in x]) != 10)[0]

# Construct the set of valid samples
valid_idx = np.setdiff1d(single_label_idx,too_short)

# Count number of valid samples for potential classes
y_single = np.asarray([y_fil[i][0] for i in valid_idx])
num_samples = [len(np.where(y_single == i)[0])  for i in classes]

# Take the n classes with the highest number of samples
n_sample_cutoff = np.sort(num_samples)[-n_classes]
class_idx = np.where(np.asarray(num_samples) >= n_sample_cutoff)[0]
our_classes = [classes[i] for i in class_idx]


### Filter the data again according the the chosen classes
y_fil = []
for i in range(len(y)):
    y_fil.append( np.intersect1d(y[i],our_classes))

# Find samples that belong to only one of the potential classes
n_labels = np.asarray([len(y) for y in y_fil])
single_label_idx = np.where(n_labels == 1)[0]

# Find samples that dont are shorter than 10 seconds 
too_short = np.where(np.asarray([x.shape[0] for x in x]) != 10)[0]

# Construct the set of valid samples
valid_idx = np.setdiff1d(single_label_idx,too_short)

# Restructure data and relabel the classes to be between 0 and n_classes
y_data = [y_fil[i][0] for i in valid_idx]
y_data = [np.where(np.asarray(our_classes) == i)[0][0] for i in y_data]
y_data = np.asarray(y_data)

x_data = [x[i] for i in valid_idx]
x_data = np.stack(x_data)


### Split into test and train and restrict the number of samples per class
np.random.seed(42)
n_train = int(n_sample * (1-test_frac))
n_test = int(n_sample * test_frac)

train_ind = []
test_ind = []

for i in range(n_classes):
    sample_idx = np.where(y_data == i)[0]
    n_sample_class = len(sample_idx)
    rand_idx = np.arange(n_sample_class)
    np.random.shuffle(rand_idx)
    train_ind.extend(sample_idx[rand_idx[0:n_train]])
    test_ind.extend(sample_idx[rand_idx[n_train:n_sample]])

train_ind = np.asarray(train_ind)
test_ind = np.asarray(test_ind)

sub_sample_idx = np.hstack((train_ind,test_ind))
x_data_sub = x_data[sub_sample_idx,:,:]
y_data_sub = y_data[sub_sample_idx]
train_ind = np.arange(0,len(train_ind))
test_ind = np.arange(len(train_ind),len(train_ind)+len(test_ind))


### Save data
with open(target_path, 'wb') as f:
        pickle.dump([x_data_sub, y_data_sub, train_ind, test_ind], f)
