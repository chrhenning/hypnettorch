#!/usr/bin/env python3
# Copyright 2022 Christian Henning
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
# @title          :examples/hypercl/run.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :01/05/2022
# @version        :1.0
# @python_version :3.8.12
"""
Script to run CL experiments with hypernetworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script showcases the usage of ``hypnettorch`` by demonstrating how to use
the pacakge for writing a continual learning simulation that utilizes
hypernetworks. See :ref:`here <exmp-hypercl-reference-label>` for details on the
approach and usage instructions.
"""
import argparse
from datetime import datetime
from hypnettorch.data.special.permuted_mnist import PermutedMNISTList
from hypnettorch.data.special.split_cifar import get_split_cifar_handlers
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers
import hypnettorch.utils.cli_args as cli
import hypnettorch.utils.hnet_regularizer as hreg
import hypnettorch.utils.sim_utils as sutils
import hypnettorch.utils.torch_utils as tutils
import numpy as np
from time import time
import torch
import torch.nn.functional as F

def load_datasets(config, logger, writer):
    """Load the datasets corresponding to individual tasks.

    Args:
        config (argparse.Namespace): Command-line arguments.
        logger (logging.Logger): Logger object.
        writer (tensorboardX.SummaryWriter): Tensorboard logger.

    Returns:
        (list): A list of data handlers
        :class:`hypnettorch.data.dataset.Dataset`.
    """
    data_dir = './datasets'

    if config.cl_exp == 'splitmnist':
        logger.info('Running SplitMNIST experiment.')

        dhandlers = get_split_mnist_handlers(data_dir, use_one_hot=True,
            num_tasks=config.num_tasks,
            num_classes_per_task=config.num_classes_per_task,
            validation_size=config.val_set_size)

    elif config.cl_exp == 'permmnist':
        logger.info('Running PermutedMNIST experiment.')
        
        pd = 2 # Apply padding as in original paper.
        in_shape = [28 + 2*pd, 28 + 2*pd, 1]
        input_dim = np.prod(in_shape)

        # Ensure reproducibility for every call of this function!
        rand = np.random.RandomState(42)
        permutations = [None] + [rand.permutation(input_dim)
                                 for _ in range(config.num_tasks - 1)]
        # NOTE When using `PermutedMNISTList` rather than `PermutedMNIST`,
        # we have to ensure a proper use of the data handlers ourselves. See
        # the corresponding documentation.
        dhandlers = PermutedMNISTList(permutations, data_dir,
            padding=pd, trgt_padding=None, show_perm_change_msg=False,
            validation_size=config.val_set_size)

    elif config.cl_exp == 'splitcifar':
        logger.info('Running CIFAR-10/100 experiment.')

        dhandlers = get_split_cifar_handlers(data_dir, use_one_hot=True,
            use_data_augmentation=True, num_tasks=config.num_tasks,
            num_classes_per_task=config.num_classes_per_task,
            validation_size=config.val_set_size)

    return dhandlers

def test(dhandlers, mnet, hnet, device, config, logger, writer):
    """Evaluate the network.

    Evaluate the performance of the network on a single task on the validation
    set during training.

    Args:
         (....): See docstring of function :func:`train`. 
         dhandlers (list): Datasets of tasks that should be tested. We assume
             that the index of the dataset corresponds to the index of the task
             embedding used as input to the hypernet.
    """
    n = len(dhandlers)
    logger.info('# Testing network on %d tasks ...' % n)

    mnet.eval()
    hnet.eval()

    if config.test_with_val_set:
        logger.warning('Testing will be performed with the validation set!')

    with torch.no_grad():
        accs = []
        for t in range(n):
            logger.info('Testing on task %d ...' % (t+1))
            data = dhandlers[t]

            # Get main network weights of current task.
            W = hnet.forward(cond_id=t)

            num_correct = 0

            if config.test_with_val_set:
                iter_fct = data.val_iterator
                ident = 'validation'
                num_samples= data.num_val_samples
            else:
                iter_fct = data.test_iterator
                ident = 'test'
                num_samples = data.num_test_samples

            # The dataset interface provides easy ways to iterate dataset splits.
            for batch_size, x, y, ids in iter_fct(config.val_batch_size,
                    return_ids=True):
                # The current batch is given as numpy data and has to first be
                # converted to numpy.
                X = data.input_to_torch_tensor(x, device, mode='inference')
                Y = data.output_to_torch_tensor(y, device, mode='inference')

                P = mnet.forward(X, weights=W)

                num_correct += int(torch.sum(Y.argmax(dim=1) == \
                    P.argmax(dim=1)).detach().cpu())

            acc = num_correct / num_samples * 100.
            accs.append(acc)

            logger.info('Test - Accuracy on %s set: %f%%.' % (ident, acc))
            writer.add_scalar('test/task_%d/accuracy' % t, acc, n)

        logger.info('Average accuracy on all trained tasks: %f%%' \
                    % np.mean(accs))

        logger.info('# Testing ... Done')

def evaluate(task_id, data, mnet, hnet, device, config, logger, writer,
             train_iter):
    """Evaluate the network.

    Evaluate the performance of the network on a single task on the validation
    set during training.

    Args:
         (....): See docstring of function :func:`train`.
        train_iter (int): The current training iteration.
    """
    logger.info('# Evaluating network on task %d ' % (task_id+1) +
                'before running training step %d ...' % (train_iter))

    mnet.eval()
    hnet.eval()

    if data.num_val_samples == 0:
        logger.warning('Cannot evaluate training without validation set!')
        return

    with torch.no_grad():
        # Get main network weights of current task.
        W = hnet.forward(cond_id=task_id)

        num_correct = 0

        # The dataset interface provides easy ways to iterate dataset splits.
        for batch_size, x, y, ids in data.val_iterator(config.val_batch_size,
                    return_ids=True):
            # The current batch is given as numpy data and has to first be
            # converted to numpy.
            X = data.input_to_torch_tensor(x, device, mode='inference')
            Y = data.output_to_torch_tensor(y, device, mode='inference')

            P = mnet.forward(X, weights=W)

            num_correct += int(torch.sum(Y.argmax(dim=1) == P.argmax(dim=1)).\
                detach().cpu())

        acc = num_correct / data.num_val_samples * 100.

        logger.info('Eval - Accuracy on validation set: %f%%.' % (acc))
        writer.add_scalar('eval/task_%d/accuracy' % task_id, acc,
                          train_iter)

        logger.info('# Evaluating training ... Done')

def train(task_id, data, mnet, hnet, device, config, logger, writer):
    r"""Train the network using the task-specific loss plus a regularizer that
    should mitigate catastrophic forgetting.

    .. math::
        \text{loss} = \text{task\_loss} + \beta * \text{regularizer}

    Args:
        task_id (int): The index of the task on which we train.
        data (hypnettorch.data.dataset.Dataset): The dataset handler for the
            current task, corresponding to ``task_id``.
        mnet (hypnettorch.mnets.mnet_interface.MainNetInterface): The model of
            the main network, which is needed to make predictions.
        hnet (hypnettorch.hnets.hnet_interface.HyperNetInterface): The model of
            the hyper network, which contains the parameters to be learned.
        device: (torch.device) Torch device (cpu or gpu).
        config (argparse.Namespace): Command-line arguments.
        logger (logging.Logger): Logger object.
        writer (tensorboardX.SummaryWriter): Tensorboard logger.
    """
    logger.info('Training network on task %d ...' % (task_id+1))

    mnet.train()
    hnet.train()

    ############################
    ### Setup CL regularizer ###
    ############################
    # The helper functions in module `hypnettorch.utiils.hnet_interface` provide
    # an easy interface for applying the desired continual learning
    # regularization to hypernets.

    # Whether the regularizer will be computed during training?
    calc_reg = task_id > 0 and config.beta > 0

    # Regularizer targets.
    if calc_reg:
        if config.calc_hnet_reg_targets_online:
            # Compute targets for the regularizer whenever they are needed.
            # -> Computationally expensive.
            reg_targets = None
            # Checkpoint hypernetwork before training.
            # Instead of checkpointing the parameters, it might be cleaner to
            # checkpoint the whole model.
            prev_hnet_theta = [p.detach().clone() \
                               for p in hnet.unconditional_params]
            prev_task_embs = [p.detach().clone() \
                              for p in hnet.conditional_params]
        else:
            # Compute targets for the regularizer once before training, as they
            # don't change during training. However, this requires storing the
            # main network's parameters for each previous task.
            # -> Computationally efficient, memory expensive.
            reg_targets = hreg.get_current_targets(task_id, hnet)
            prev_hnet_theta = None
            prev_task_embs = None

    ########################
    ### Create optimizer ###
    ########################
    # Only the hypernetwork has parameters to be trained!
    params = hnet.parameters()
    # Just a helper function, you can create the optimizer using PyTorch
    # directly.
    optimizer = tutils.get_optimizer(params, config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay,
        use_adam=True, adam_beta1=config.adam_beta1)

    ######################
    ### Start training ###
    ######################
    # Note, the user can decide to either train for a certain number of
    # iterations or specify a number of epochs per task.
    num_train_iter, iter_per_epoch = sutils.calc_train_iter( \
        data.num_train_samples, config.batch_size, num_iter=config.n_iter,
        epochs=config.epochs)

    for i in range(num_train_iter):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            evaluate(task_id, data, mnet, hnet, device, config, logger,
                     writer, i)
            mnet.train()
            hnet.train()

        if i % 100 == 0:
            logger.debug('Training iteration: %d.' % i)

        ### Train theta and task embeddings.
        optimizer.zero_grad()

        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')

        # Get weights of current task.
        weights = hnet.forward(cond_id=task_id)
        # Compute predictions on training batch via these weights.
        Y = mnet.forward(X, weights=weights)

        # Evaluate task-specific loss.
        labels = T.argmax(dim=1)
        loss_task = F.cross_entropy(Y, labels, reduction='mean')

        ### Compute hypernet regularizer ###
        loss_reg = 0
        if calc_reg:
            # We use the corresponding helper function to compute the
            # regularizer.
            loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                targets=reg_targets, mnet=mnet, prev_theta=prev_hnet_theta,
                prev_task_embs=prev_task_embs, inds_of_out_heads=None,
                batch_size=config.hnet_reg_batch_size)

        loss = loss_task + config.beta * loss_reg
        loss.backward()
        optimizer.step()

        ### Tensorboard summary ###
        if i % 50 == 0:
            writer.add_scalar('train/task_%d/loss' % task_id, loss, i)
            writer.add_scalar('train/task_%d/loss_task' % task_id, loss_task, i)
            writer.add_scalar('train/task_%d/regularizer' % task_id, loss_reg,
                              i)

            train_acc = (labels == Y.argmax(dim=1)).sum().detach().cpu()
            writer.add_scalar('train/task_%d/accuracy' % task_id, train_acc, i)

            weights_flattened = torch.cat([d.clone().view(-1) for d in weights])
            writer.add_histogram('train/task_%d/predicted_weights' % task_id,
                                 weights_flattened, i)

    logger.info('Training network on task %d ... Done' % (task_id+1))

def run():
    """Run the script.

    #. Define and parse command-line arguments
    #. Setup environment
    #. Load data
    #. Instantiate models
    #. Run training for each task
    """
    script_start = time()

    ####################################
    ### Parse Command-Line Arguments ###
    ####################################
    # One may use the helper functions provided in `hypnettorch.utils.cli_args`
    # to efficiently generate useful command-line arguments such as those useful
    # for selecting a hypernetwork architecture.
    # Note, the amount of command-line arguments specified here might at first
    # seem overwhelming, but most of them are not necessary for you to
    # know (see README for relevant ones) and they are all attached with a
    # description when running the script with `--help`.
    parser = argparse.ArgumentParser(description='CL with hypernetworks')
    # Default output directory:
    dout_dir = './out/run_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    ### Continual learning arguments ###
    cl_agroup = cli.cl_args(parser, show_beta=True, dbeta=0.01,
        show_from_scratch=False, show_multi_head=False, show_cl_scenario=False,
        show_split_head_cl3=False, show_num_tasks=True, dnum_tasks=5,
        show_num_classes_per_task=True, dnum_classes_per_task=2,
        show_calc_hnet_reg_targets_online=True, show_hnet_reg_batch_size=True)
    # You can easily add your own arguments.
    cl_agroup.add_argument('--cl_exp', type=str, default='splitmnist',
                            help='Which continual learning experiment should ' +
                                 'be performed: SplitMNIST, PermutedMNIST or ' +
                                 'SplitCIFAR. Default: %(default)s.',
                            choices=['splitmnist', 'permmnist', 'splitcifar'])
    ### Training arguments ###
    # We only explore Adam as optimizer.
    cli.train_args(parser, show_lr=True, dlr=1e-4, show_epochs=True, depochs=-1,
        dbatch_size=32, dn_iter=1001, show_adam_beta1=True, dadam_beta1=0.9,
        show_momentum=True)
    ### Main network arguments ###
    cli.main_net_args(parser, allowed_nets=['mlp', 'lenet', 'resnet', 'wrn'],
        dmlp_arch='100,100', dlenet_type='mnist_small', dresnet_block_depth=5,
        dresnet_channel_sizes='16,16,32,64', dwrn_block_depth=4,
        dwrn_widening_factor=10, show_net_act=True, dnet_act='relu',
        show_no_bias=True, show_dropout_rate=True, ddropout_rate=-1,
        show_specnorm=True, show_batchnorm=True, show_bn_no_running_stats=True,
        show_bn_distill_stats=False, show_bn_no_stats_checkpointing=True)
    ### Hypernetwork arguments ###
    cli.hnet_args(parser, allowed_nets=['chunked_hmlp', 'hmlp'],
        dhmlp_arch='100,100', show_cond_emb_size=True, dcond_emb_size='8',
        dchmlp_chunk_size=750, dchunk_emb_size=8,
        show_use_cond_chunk_embs=True, show_net_act=True, dnet_act='relu',
        show_no_bias=True, show_dropout_rate=True, ddropout_rate=-1,
        show_specnorm=False, show_batchnorm=False, show_no_batchnorm=False)
    ### Validation arguments ###
    cli.eval_args(parser, dval_iter=500, show_val_batch_size=True,
        dval_batch_size=256, show_val_set_size=True, dval_set_size=100,
        show_test_with_val_set=True)
    ### Miscellaneous arguments ###
    cli.miscellaneous_args(parser, big_data=False, synthetic_data=False,
        show_plots=False, no_cuda=False, dout_dir=dout_dir)

    config = parser.parse_args()

    cli.check_invalid_argument_usage(config)

    if config.cl_exp == 'permmnist':
        if config.num_classes_per_task != 10:
            raise ValueError('Argument "num_classes_per_task" must be 10 in ' +
                             'PermutedMNIST!')

    if config.batchnorm:
        # It's not complicated to realize, but one has to handle the batchnorm
        # statistics properly, e.g., by telling the main net at each forward
        # pass which ones to use: https://git.io/J9LRv
        # and by checkpointing the stats when training of a task is finished:
        # https://git.io/J9LRm
        raise NotImplementedError('Use of batchnorm not implemented yet!')

    ##############################################
    ### Load datasets and instantiate networks ###
    ##############################################

    device, writer, logger = sutils.setup_environment(config,
        logger_name=config.cl_exp + '_logger')

    ### Generate datasets ###
    dhandlers = load_datasets(config, logger, writer)

    # Note, when using the helper functions `main_net_args` and  `hnet_args` in
    # `hypnettorch.utils.cli_args`, main networks and hypernetworks can easily
    # be instantiated by using the corresponding helper functions in
    # `hypnettorch.utils.sim_utils`.
    ### Instantiate main network ###
    in_shape = dhandlers[0].in_shape
    if hasattr(dhandlers[0], 'torch_in_shape'):
        # This is only relevant for PermutedMNIST if padding is used (see the
        # corresponding data handler).
        in_shape = dhandlers[0].torch_in_shape
    if config.net_type == 'mlp':
        in_shape = [int(np.prod(in_shape))] # flattened images
    # As we learn task-specific models via the hypernetwork, the main network
    # only needs as many output units as required for a given task.
    out_shape = [config.num_classes_per_task]
    # The network is instantiated with no internal weights (they need to be
    # provided via a hypernet).
    mnet = sutils.get_mnet_model(config, config.net_type, in_shape, out_shape,
        device, no_weights=True)

    ### Instantiate hypernetwork ###
    hnet = sutils.get_hypernet(config, device, config.hnet_type,
        mnet.param_shapes, config.num_tasks)
    # If desired, one could now perform some specialized initialization of
    # the hypernet's parameters.

    ###################################
    ### Train on tasks sequentially ###
    ###################################
    for t in range(config.num_tasks):
        logger.info('### Training on task %d ###' % (t+1))
        data = dhandlers[t]

        # Only data of current task will be available for training.
        train(t, data, mnet, hnet, device, config, logger, writer)

        # Test on all tasks trained so far.
        test(dhandlers[:(t+1)], mnet, hnet, device, config, logger, writer)

    writer.close()

    logger.info('Program finished successfully in %f sec.'
                % (time() - script_start))

if __name__ == '__main__':
    run()


