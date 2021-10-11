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
# @title          :utils/sim_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :12/12/2019
# @version        :1.0
# @python_version :3.6.8
"""
General helper functions for simulations
----------------------------------------

The module :mod:`utils.sim_utils` comprises a bunch of functions that are in
general useful for writing simulations in this repository.
"""
import torch
import tensorboardX
from tensorboardX import SummaryWriter
import numpy as np
import random
import os
import select
import shutil
import sys
import pickle
import logging
from time import time
from warnings import warn
import json

from hypnettorch.mnets.bio_conv_net import BioConvNet
from hypnettorch.mnets.chunk_squeezer import ChunkSqueezer
from hypnettorch.mnets.lenet import LeNet
from hypnettorch.mnets.mlp import MLP
from hypnettorch.mnets.resnet_imgnet import ResNetIN
from hypnettorch.mnets.resnet import ResNet
from hypnettorch.mnets.simple_rnn import SimpleRNN
from hypnettorch.mnets.wide_resnet import WRN
from hypnettorch.mnets.zenkenet import ZenkeNet
from hypnettorch.hnets.mlp_hnet import HMLP
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
from hypnettorch.hnets.structured_mlp_hnet import StructuredHMLP
from hypnettorch.hnets.deconv_hnet import HDeconv
from hypnettorch.hnets.chunked_deconv_hnet import ChunkedHDeconv
from hypnettorch.utils import logger_config
from hypnettorch.utils import misc

def setup_environment(config, logger_name='hnet_sim_logger'):
    """Setup the general environment for training.

    This function should be called at the beginning of a simulation script
    (right after the command-line arguments have been parsed). The setup will
    incorporate:

        - creating the output folder
        - initializing logger
        - making computation deterministic (depending on config)
        - selecting the torch device
        - creating the Tensorboard writer

    Args:
        config (argparse.Namespace): Command-line arguments.

            .. note::
                The function expects command-line arguments available according
                to the function :func:`utils.cli_args.miscellaneous_args`.
        logger_name (str): Name of the logger to be created (time stamp will be
            appended to this name).

    Returns:
        (tuple): Tuple containing:

        - **device**: Torch device to be used.
        - **writer**: Tensorboard writer. Note, you still have to close the
          writer manually!
        - **logger**: Console (and file) logger.
    """
    ### Output folder.
    if os.path.exists(config.out_dir):
        # TODO allow continuing from an old checkpoint.
        # FIXME We do not want to use python its `input` function, as it blocks
        # the program completely. Therefore, we use `select`, but this might
        # not work on all platforms!
        #response = input('The output folder %s already exists. ' % \
        #                 (config.out_dir) + \
        #                 'Do you want us to delete it? [y/n]')
        print('The output folder %s already exists. ' % (config.out_dir) + \
              'Do you want us to delete it? [y/n]')
        inps, _, _ = select.select([sys.stdin], [], [], 30)
        if len(inps) == 0:
            warn('Timeout occurred. No user input received!')
            response = 'n'
        else:
            response = sys.stdin.readline().strip()
        if response != 'y':
            raise IOError('Could not delete output folder!')
        shutil.rmtree(config.out_dir)

        os.makedirs(config.out_dir)
        print("Created output folder %s." % (config.out_dir))

    else:
        os.makedirs(config.out_dir)
        print("Created output folder %s." % (config.out_dir))

    # Save user configs to ensure reproducibility of this experiment.
    with open(os.path.join(config.out_dir, 'config.pickle'), 'wb') as f:
        pickle.dump(config, f)
    # A JSON file is easier to read for a human.
    with open(os.path.join(config.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f)

    ### Initialize logger.
    logger_name = '%s_%d' % (logger_name, int(time() * 1000))
    logger = logger_config.config_logger(logger_name,
        os.path.join(config.out_dir, 'logfile.txt'),
        logging.DEBUG, logging.INFO if config.loglevel_info else logging.DEBUG)
    # FIXME If we don't disable this, then the multiprocessing from the data
    # loader causes all messages to be logged twice. I could not find the cause
    # of this problem, but this simple switch fixes it.
    logger.propagate = False

    ### Deterministic computation.
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    # Ensure that runs are reproducible. Note, this slows down training!
    # https://pytorch.org/docs/stable/notes/randomness.html
    if config.deterministic_run:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if hasattr(config, 'num_workers') and config.num_workers > 1:
            logger.warning('Deterministic run desired but not possible with ' +
                           'more than 1 worker (see "num_workers").')

    ### Select torch device.
    assert(hasattr(config, 'no_cuda') or hasattr(config, 'use_cuda'))
    assert(not hasattr(config, 'no_cuda') or not hasattr(config, 'use_cuda'))

    if hasattr(config, 'no_cuda'):
        use_cuda = not config.no_cuda and torch.cuda.is_available()
    else:
        use_cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info('Using cuda: ' + str(use_cuda))

    ### Initialize summary writer.
    # Flushes every 120 secs by default.
    # DELETEME Ensure downwards compatibility.
    if not hasattr(tensorboardX, '__version__'):
        writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'summary'))
    else:
        writer = SummaryWriter(logdir=os.path.join(config.out_dir, 'summary'))

    return device, writer, logger

def get_mnet_model(config, net_type, in_shape, out_shape, device, cprefix=None,
                   no_weights=False, **mnet_kwargs):
    """Generate a main network instance.

    A helper to generate a main network according to the given the user
    configurations.

    .. note::
        Generation of networks with context-modulation is not yet supported,
        since there is no global argument set in :mod:`utils.cli_args` yet.

    Args:
        config (argparse.Namespace): Command-line arguments.

            .. note::
                The function expects command-line arguments available according
                to the function :func:`utils.cli_args.main_net_args`.
        net_type (str): The type of network. The following options are
            available:

            - ``mlp``: :class:`mnets.mlp.MLP`
            - ``resnet``: :class:`mnets.resnet.ResNet`
            - ``wrn``: :class:`mnets.wide_resnet.WRN`
            - ``iresnet``: :class:`mnets.resnet_imgnet.ResNetIN`
            - ``zenke``: :class:`mnets.zenkenet.ZenkeNet`
            - ``bio_conv_net``: :class:`mnets.bio_conv_net.BioConvNet`
            - ``chunked_mlp``: :class:`mnets.chunk_squeezer.ChunkSqueezer`
            - ``simple_rnn``: :class:`mnets.simple_rnn.SimpleRNN`
        in_shape (list): Shape of network inputs. Can be ``None`` if not
            required by network type.

            For instance: For an MLP network :class:`mnets.mlp.MLP` with 100
            input neurons it should be :code:`in_shape=[100]`.
        out_shape (list): Shape of network outputs. See ``in_shape`` for more
            details.
        device: PyTorch device.
        cprefix (str, optional): A prefix of the config names. It might be, that
            the config names used in this method are prefixed, since several
            main networks should be generated (e.g., :code:`cprefix='gen_'` or
            ``'dis_'`` when training a GAN).

            Also see docstring of parameter ``prefix`` in function
            :func:`utils.cli_args.main_net_args`.
        no_weights (bool): Whether the main network should be generated without
            weights.
        **mnet_kwargs: Additional keyword arguments that will be passed to the
            main network constructor.

    Returns:
        The created main network model.
    """
    assert(net_type in ['mlp', 'lenet', 'resnet', 'zenke', 'bio_conv_net',
                        'chunked_mlp', 'simple_rnn', 'wrn', 'iresnet'])

    if cprefix is None:
        cprefix = ''

    def gc(name):
        """Get config value with that name."""
        return getattr(config, '%s%s' % (cprefix, name))
    def hc(name):
        """Check whether config exists."""
        return hasattr(config, '%s%s' % (cprefix, name))

    mnet = None

    if hc('net_act'):
        net_act = gc('net_act')
        net_act = misc.str_to_act(net_act)
    else:
        net_act = None

    def get_val(name):
        ret = None
        if hc(name):
            ret = gc(name)
        return ret

    no_bias = get_val('no_bias')
    dropout_rate = get_val('dropout_rate')
    specnorm = get_val('specnorm')
    batchnorm = get_val('batchnorm')
    no_batchnorm = get_val('no_batchnorm')
    bn_no_running_stats = get_val('bn_no_running_stats')
    bn_distill_stats = get_val('bn_distill_stats')
    # This argument has to be handled during usage of the network and not during
    # construction.
    #bn_no_stats_checkpointing = get_val('bn_no_stats_checkpointing')

    use_bn = None
    if batchnorm is not None:
        use_bn = batchnorm
    elif no_batchnorm is not None:
        use_bn = not no_batchnorm

    # If an argument wasn't specified, then we use the default value that
    # is currently in the constructor.
    assign = lambda x, y : y if x is None else x

    if net_type == 'mlp':
        assert(hc('mlp_arch'))
        assert(len(in_shape) == 1 and len(out_shape) == 1)

        # Default keyword arguments of class MLP.
        dkws = misc.get_default_args(MLP.__init__)

        mnet = MLP(n_in=in_shape[0], n_out=out_shape[0],
            hidden_layers=misc.str_to_ints(gc('mlp_arch')),
            activation_fn=assign(net_act, dkws['activation_fn']),
            use_bias=assign(not no_bias, dkws['use_bias']),
            no_weights=no_weights,
            #init_weights=None,
            dropout_rate=assign(dropout_rate, dkws['dropout_rate']),
            use_spectral_norm=assign(specnorm, dkws['use_spectral_norm']),
            use_batch_norm=assign(use_bn, dkws['use_batch_norm']),
            bn_track_stats=assign(not bn_no_running_stats,
                                  dkws['bn_track_stats']),
            distill_bn_stats=assign(bn_distill_stats, dkws['distill_bn_stats']),
            #use_context_mod=False,
            #context_mod_inputs=False,
            #no_last_layer_context_mod=False,
            #context_mod_no_weights=False,
            #context_mod_post_activation=False,
            #context_mod_gain_offset=False,
            #out_fn=None,
            verbose=True,
            **mnet_kwargs
        ).to(device)

    elif net_type == 'resnet':
        assert(len(out_shape) == 1)
        assert hc('resnet_block_depth') and hc('resnet_channel_sizes')

        # Default keyword arguments of class ResNet.
        dkws = misc.get_default_args(ResNet.__init__)

        mnet = ResNet(in_shape=in_shape, num_classes=out_shape[0],
            n=gc('resnet_block_depth'),
            use_bias=assign(not no_bias, dkws['use_bias']),
            num_feature_maps=misc.str_to_ints(gc('resnet_channel_sizes')),
            verbose=True, #n=5,
            no_weights=no_weights,
            #init_weights=None,
            use_batch_norm=assign(use_bn, dkws['use_batch_norm']),
            bn_track_stats=assign(not bn_no_running_stats,
                                  dkws['bn_track_stats']),
            distill_bn_stats=assign(bn_distill_stats, dkws['distill_bn_stats']),
            #use_context_mod=False,
            #context_mod_inputs=False,
            #no_last_layer_context_mod=False,
            #context_mod_no_weights=False,
            #context_mod_post_activation=False,
            #context_mod_gain_offset=False,
            #context_mod_apply_pixel_wise=False
            **mnet_kwargs
        ).to(device)

    elif net_type == 'wrn':
        assert(len(out_shape) == 1)
        assert hc('wrn_block_depth') and hc('wrn_widening_factor')

        # Default keyword arguments of class WRN.
        dkws = misc.get_default_args(WRN.__init__)

        mnet = WRN(in_shape=in_shape, num_classes=out_shape[0],
            n=gc('wrn_block_depth'),
            use_bias=assign(not no_bias, dkws['use_bias']),
            #num_feature_maps=misc.str_to_ints(gc('wrn_channel_sizes')),
            verbose=True,
            no_weights=no_weights,
            use_batch_norm=assign(use_bn, dkws['use_batch_norm']),
            bn_track_stats=assign(not bn_no_running_stats,
                                  dkws['bn_track_stats']),
            distill_bn_stats=assign(bn_distill_stats, dkws['distill_bn_stats']),
            k=gc('wrn_widening_factor'),
            use_fc_bias=gc('wrn_use_fc_bias'),
            dropout_rate=gc('dropout_rate'),
            #use_context_mod=False,
            #context_mod_inputs=False,
            #no_last_layer_context_mod=False,
            #context_mod_no_weights=False,
            #context_mod_post_activation=False,
            #context_mod_gain_offset=False,
            #context_mod_apply_pixel_wise=False
            **mnet_kwargs
        ).to(device)

    elif net_type == 'iresnet':
        assert(len(out_shape) == 1)
        assert hc('iresnet_use_fc_bias') and hc('iresnet_channel_sizes') \
            and hc('iresnet_blocks_per_group') \
            and hc('iresnet_bottleneck_blocks') \
            and hc('iresnet_projection_shortcut')

        # Default keyword arguments of class WRN.
        dkws = misc.get_default_args(ResNetIN.__init__)

        mnet = ResNetIN(in_shape=in_shape, num_classes=out_shape[0],
            use_bias=assign(not no_bias, dkws['use_bias']),
            use_fc_bias=gc('wrn_use_fc_bias'),
            num_feature_maps=misc.str_to_ints(gc('iresnet_channel_sizes')),
            blocks_per_group=misc.str_to_ints(gc('iresnet_blocks_per_group')),
            projection_shortcut=gc('iresnet_projection_shortcut'),
            bottleneck_blocks=gc('iresnet_bottleneck_blocks'),
            #cutout_mod=False,
            no_weights=no_weights,
            use_batch_norm=assign(use_bn, dkws['use_batch_norm']),
            bn_track_stats=assign(not bn_no_running_stats,
                                  dkws['bn_track_stats']),
            distill_bn_stats=assign(bn_distill_stats, dkws['distill_bn_stats']),
            #chw_input_format=False,
            verbose=True,
            #use_context_mod=False,
            #context_mod_inputs=False,
            #no_last_layer_context_mod=False,
            #context_mod_no_weights=False,
            #context_mod_post_activation=False,
            #context_mod_gain_offset=False,
            #context_mod_apply_pixel_wise=False
            **mnet_kwargs
        ).to(device)

    elif net_type == 'zenke':
        assert(len(out_shape) == 1)

        # Default keyword arguments of class ZenkeNet.
        dkws = misc.get_default_args(ZenkeNet.__init__)

        mnet = ZenkeNet(in_shape=in_shape, num_classes=out_shape[0],
            verbose=True, #arch='cifar',
            no_weights=no_weights,
            #init_weights=None,
            dropout_rate=assign(dropout_rate, dkws['dropout_rate']),
            **mnet_kwargs
        ).to(device)

    elif net_type == 'bio_conv_net':
        assert(len(out_shape) == 1)

        # Default keyword arguments of class BioConvNet.
        #dkws = misc.get_default_args(BioConvNet.__init__)

        mnet = BioConvNet(in_shape=in_shape, num_classes=out_shape[0],
            no_weights=no_weights,
            #init_weights=None,
            #use_context_mod=False,
            #context_mod_inputs=False,
            #no_last_layer_context_mod=False,
            #context_mod_no_weights=False,
            #context_mod_post_activation=False,
            #context_mod_gain_offset=False,
            #context_mod_apply_pixel_wise=False
            **mnet_kwargs
        ).to(device)

    elif net_type == 'chunked_mlp':
        assert hc('cmlp_arch') and hc('cmlp_chunk_arch') and \
               hc('cmlp_in_cdim') and hc('cmlp_out_cdim') and \
               hc('cmlp_cemb_dim')
        assert len(in_shape) == 1 and len(out_shape) == 1

        # Default keyword arguments of class ChunkSqueezer.
        dkws = misc.get_default_args(ChunkSqueezer.__init__)

        mnet = ChunkSqueezer(n_in=in_shape[0], n_out=out_shape[0],
            inp_chunk_dim=gc('cmlp_in_cdim'),
            out_chunk_dim=gc('cmlp_out_cdim'),
            cemb_size=gc('cmlp_cemb_dim'),
            #cemb_init_std=1.,
            red_layers=misc.str_to_ints(gc('cmlp_chunk_arch')),
            net_layers=misc.str_to_ints(gc('cmlp_arch')),
            activation_fn=assign(net_act, dkws['activation_fn']),
            use_bias=assign(not no_bias, dkws['use_bias']),
            #dynamic_biases=None,
            no_weights=no_weights,
            #init_weights=None,
            dropout_rate=assign(dropout_rate, dkws['dropout_rate']),
            use_spectral_norm=assign(specnorm, dkws['use_spectral_norm']),
            use_batch_norm=assign(use_bn, dkws['use_batch_norm']),
            bn_track_stats=assign(not bn_no_running_stats,
                                  dkws['bn_track_stats']),
            distill_bn_stats=assign(bn_distill_stats, dkws['distill_bn_stats']),
            verbose=True,
            **mnet_kwargs
        ).to(device)

    elif net_type == 'lenet':
        assert hc('lenet_type')
        assert len(out_shape) == 1

        # Default keyword arguments of class LeNet.
        dkws = misc.get_default_args(LeNet.__init__)

        mnet = LeNet(in_shape=in_shape, num_classes=out_shape[0], verbose=True,
            arch=gc('lenet_type'),
            no_weights=no_weights,
            #init_weights=None,
            dropout_rate=assign(dropout_rate, dkws['dropout_rate']),
            # TODO Context-mod weights.
            **mnet_kwargs
        ).to(device)

    else:
        assert (net_type == 'simple_rnn')
        assert hc('srnn_rec_layers') and hc('srnn_pre_fc_layers') and \
            hc('srnn_post_fc_layers')  and hc('srnn_no_fc_out') and \
            hc('srnn_rec_type')
        assert len(in_shape) == 1 and len(out_shape) == 1

        if gc('srnn_rec_type') == 'lstm':
            use_lstm = True
        else:
            assert gc('srnn_rec_type') == 'elman'
            use_lstm = False

        # Default keyword arguments of class SimpleRNN.
        dkws = misc.get_default_args(SimpleRNN.__init__)

        rnn_layers = misc.str_to_ints(gc('srnn_rec_layers'))
        fc_layers = misc.str_to_ints(gc('srnn_post_fc_layers'))
        if gc('srnn_no_fc_out'):
            rnn_layers.append(out_shape[0])
        else:
            fc_layers.append(out_shape[0])

        mnet = SimpleRNN(n_in=in_shape[0], rnn_layers=rnn_layers,
            fc_layers_pre=misc.str_to_ints(gc('srnn_pre_fc_layers')),
            fc_layers=fc_layers,
            activation=assign(net_act, dkws['activation']),
            use_lstm=use_lstm,
            use_bias=assign(not no_bias, dkws['use_bias']),
            no_weights=no_weights,
            verbose=True,
            **mnet_kwargs
        ).to(device)

    return mnet

def get_hypernet(config, device, net_type, target_shapes, num_conds,
                 no_cond_weights=False, no_uncond_weights=False,
                 uncond_in_size=0, shmlp_chunk_shapes=None,
                 shmlp_num_per_chunk=None, shmlp_assembly_fct=None,
                 verbose=True, cprefix=None):
    """Generate a hypernetwork instance.

    A helper to generate the hypernetwork according to the given the user
    configurations.

    Args:
        config (argparse.Namespace): Command-line arguments.

            Note:
                The function expects command-line arguments available according
                to the function :func:`utils.cli_args.hnet_args`.
        device: PyTorch device.
        net_type (str): The type of network. The following options are
            available:

            - ``'hmlp'``
            - ``'chunked_hmlp'``
            - ``'structured_hmlp'``
            - ``'hdeconv'``
            - ``'chunked_hdeconv'``
        target_shapes (list): See argument ``target_shapes`` of
            :class:`hnets.mlp_hnet.HMLP`.
        num_conds (int): Number of conditions that should be known to the
            hypernetwork.
        no_cond_weights (bool): See argument ``no_cond_weights`` of
            :class:`hnets.mlp_hnet.HMLP`.
        no_uncond_weights (bool): See argument ``no_uncond_weights`` of
            :class:`hnets.mlp_hnet.HMLP`.
        uncond_in_size (int): See argument ``uncond_in_size`` of
            :class:`hnets.mlp_hnet.HMLP`.
        shmlp_chunk_shapes (list, optional): Argument ``chunk_shapes`` of
            :class:`hnets.structured_mlp_hnet.StructuredHMLP`.
        shmlp_num_per_chunk (list, optional): Argument ``num_per_chunk`` of
            :class:`hnets.structured_mlp_hnet.StructuredHMLP`.
        shmlp_assembly_fct (func, optional): Argument ``assembly_fct`` of
            :class:`hnets.structured_mlp_hnet.StructuredHMLP`.
        verbose (bool): Argument ``verbose`` of :class:`hnets.mlp_hnet.HMLP`.
        cprefix (str, optional): A prefix of the config names. It might be, that
            the config names used in this function are prefixed, since several
            hypernetworks should be generated.

            Also see docstring of parameter ``prefix`` in function
            :func:`utils.cli_args.hnet_args`.
    """
    assert net_type in ['hmlp', 'chunked_hmlp', 'structured_hmlp', 'hdeconv',
                        'chunked_hdeconv']

    hnet = None

    ### FIXME Code almost identically copied from `get_mnet_model` ###
    if cprefix is None:
        cprefix = ''

    def gc(name):
        """Get config value with that name."""
        return getattr(config, '%s%s' % (cprefix, name))
    def hc(name):
        """Check whether config exists."""
        return hasattr(config, '%s%s' % (cprefix, name))

    if hc('hnet_net_act'):
        net_act = gc('hnet_net_act')
        net_act = misc.str_to_act(net_act)
    else:
        net_act = None

    def get_val(name):
        ret = None
        if hc(name):
            ret = gc(name)
        return ret

    no_bias = get_val('hnet_no_bias')
    dropout_rate = get_val('hnet_dropout_rate')
    specnorm = get_val('hnet_specnorm')
    batchnorm = get_val('hnet_batchnorm')
    no_batchnorm = get_val('hnet_no_batchnorm')
    #bn_no_running_stats = get_val('hnet_bn_no_running_stats')
    #n_distill_stats = get_val('hnet_bn_distill_stats')

    use_bn = None
    if batchnorm is not None:
        use_bn = batchnorm
    elif no_batchnorm is not None:
        use_bn = not no_batchnorm

    # If an argument wasn't specified, then we use the default value that
    # is currently in the constructor.
    assign = lambda x, y : y if x is None else x
    ### FIXME Code copied until here                               ###

    if hc('hmlp_arch'):
        hmlp_arch_is_list = False
        hmlp_arch = gc('hmlp_arch')
        if ';' in hmlp_arch:
            hmlp_arch_is_list = True
            if net_type != 'structured_hmlp':
                raise ValueError('Option "%shmlp_arch" may only ' % (cprefix) +
                                 'contain semicolons for network type ' +
                                 '"structured_hmlp"!')
            hmlp_arch = [misc.str_to_ints(ar) for ar in hmlp_arch.split(';')]
        else:
            hmlp_arch = misc.str_to_ints(hmlp_arch)
    if hc('chunk_emb_size'):
        chunk_emb_size = gc('chunk_emb_size')
        chunk_emb_size = misc.str_to_ints(chunk_emb_size)
        if len(chunk_emb_size) == 1:
            chunk_emb_size = chunk_emb_size[0]
        else:
            if net_type != 'structured_hmlp':
                raise ValueError('Option "%schunk_emb_size" may ' % (cprefix) +
                                 'only contain multiple values for network ' +
                                 'type "structured_hmlp"!')

    if hc('cond_emb_size'):
        cond_emb_size = gc('cond_emb_size')
    else:
        cond_emb_size = 0

    if net_type == 'hmlp':
        assert hc('hmlp_arch')

        # Default keyword arguments of class HMLP.
        dkws = misc.get_default_args(HMLP.__init__)

        hnet = HMLP(target_shapes,
            uncond_in_size=uncond_in_size,
            cond_in_size=cond_emb_size,
            layers=hmlp_arch,
            verbose=verbose,
            activation_fn=assign(net_act, dkws['activation_fn']),
            use_bias=assign(not no_bias, dkws['use_bias']),
            no_uncond_weights=no_uncond_weights,
            no_cond_weights=no_cond_weights,
            num_cond_embs=num_conds,
            dropout_rate=assign(dropout_rate, dkws['dropout_rate']),
            use_spectral_norm=assign(specnorm, dkws['use_spectral_norm']),
            use_batch_norm=assign(use_bn, dkws['use_batch_norm'])).to(device)

    elif net_type == 'chunked_hmlp':
        assert hc('hmlp_arch')
        assert hc('chmlp_chunk_size')
        assert hc('chunk_emb_size')
        cond_chunk_embs = get_val('use_cond_chunk_embs')

        # Default keyword arguments of class ChunkedHMLP.
        dkws = misc.get_default_args(ChunkedHMLP.__init__)

        hnet = ChunkedHMLP(target_shapes, gc('chmlp_chunk_size'),
            chunk_emb_size=chunk_emb_size,
            cond_chunk_embs=assign(cond_chunk_embs, dkws['cond_chunk_embs']),
            uncond_in_size=uncond_in_size,
            cond_in_size=cond_emb_size,
            layers=hmlp_arch,
            verbose=verbose,
            activation_fn=assign(net_act, dkws['activation_fn']),
            use_bias=assign(not no_bias, dkws['use_bias']),
            no_uncond_weights=no_uncond_weights,
            no_cond_weights=no_cond_weights,
            num_cond_embs=num_conds,
            dropout_rate=assign(dropout_rate, dkws['dropout_rate']),
            use_spectral_norm=assign(specnorm, dkws['use_spectral_norm']),
            use_batch_norm=assign(use_bn, dkws['use_batch_norm'])).to(device)

    elif net_type == 'structured_hmlp':
        assert hc('hmlp_arch')
        assert hc('chunk_emb_size')
        cond_chunk_embs = get_val('use_cond_chunk_embs')

        assert shmlp_chunk_shapes is not None and \
            shmlp_num_per_chunk is not None and \
            shmlp_assembly_fct is not None

        # Default keyword arguments of class StructuredHMLP.
        dkws = misc.get_default_args(StructuredHMLP.__init__)
        dkws_hmlp = misc.get_default_args(HMLP.__init__)

        shmlp_hmlp_kwargs = []
        if not hmlp_arch_is_list:
            hmlp_arch = [hmlp_arch]
        for i, arch in enumerate(hmlp_arch):
            shmlp_hmlp_kwargs.append({
                'layers': arch,
                'activation_fn': assign(net_act, dkws_hmlp['activation_fn']),
                'use_bias': assign(not no_bias, dkws_hmlp['use_bias']),
                'dropout_rate': assign(dropout_rate, dkws_hmlp['dropout_rate']),
                'use_spectral_norm': \
                    assign(specnorm, dkws_hmlp['use_spectral_norm']),
                'use_batch_norm': assign(use_bn, dkws_hmlp['use_batch_norm'])
            })
        if len(shmlp_hmlp_kwargs) == 1:
            shmlp_hmlp_kwargs = shmlp_hmlp_kwargs[0]

        hnet = StructuredHMLP(target_shapes,
            shmlp_chunk_shapes,
            shmlp_num_per_chunk,
            chunk_emb_size,
            shmlp_hmlp_kwargs,
            shmlp_assembly_fct,
            cond_chunk_embs=assign(cond_chunk_embs, dkws['cond_chunk_embs']),
            uncond_in_size=uncond_in_size,
            cond_in_size=cond_emb_size,
            verbose=verbose,
            no_uncond_weights=no_uncond_weights,
            no_cond_weights=no_cond_weights,
            num_cond_embs=num_conds).to(device)

    elif net_type == 'hdeconv':
        #HDeconv
        raise NotImplementedError
    else:
        assert net_type == 'chunked_hdeconv'
        #ChunkedHDeconv
        raise NotImplementedError

    return hnet

def calc_train_iter(num_train_samples, batch_size, num_iter=-1, epochs=-1):
    """Calculate the number of training tierations.

    If ``epochs`` is specified, this method will compute the total number of
    training iterations and the number of iterations per epoch.

    Otherwise, the number of training iterations is simply set to ``num_iter``.

    Args:
        num_train_samples (int): Numbe rof training samples in dataset.
        batch_size (int): Mini-batch size during training.
        num_iter (int): Number of training iterations. Only needs to be
            specified if ``epochs`` is ``-1``.
        epochs (int, optional): Number of training epochs.

    Returns:
        (tuple): Tuple containing:
            
        - **num_train_iter**: Total number of training iterations.
        - **iter_per_epoch**: Number of training iterations per epoch. Is set to
          ``-1`` in case ``epochs`` is unspecified.
    """
    assert num_iter != -1 or epochs != -1

    iter_per_epoch = -1
    if epochs == -1:
        num_train_iter = num_iter
    else:
        assert epochs > 0
        iter_per_epoch = int(np.ceil(num_train_samples / \
                                     batch_size))
        num_train_iter = epochs * iter_per_epoch

    return num_train_iter, iter_per_epoch

if __name__ == '__main__':
    pass


