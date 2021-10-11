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
# @title           :utils/cli_args.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :08/21/2019
# @version         :1.0
# @python_version  :3.6.8
"""
Common command-line arguments
-----------------------------

This file has a collection of helper functions that can be used to specify
command-line arguments. In particular, arguments that are necessary for
multiple experiments (even though with different default values) should be
specified here, such that we do not define arguments (and their help texts)
multiple times.

All functions specified here are helper functions for a simulation specific
argument parser such as :func:`cifar.train_args.parse_cmd_arguments`.

Important note for contributors
###############################

**DO NEVER CHANGE DEFAULT VALUES.** Instead, add a keyword argument to the
corresponding method, that allows you to change the default value, when you
call the method.
"""
from datetime import datetime
from warnings import warn

def hnet_args(parser, allowed_nets=['hmlp'], dhmlp_arch='100,100',
              show_cond_emb_size=True, dcond_emb_size='8',
              dchmlp_chunk_size=1000, dchunk_emb_size=8,
              show_use_cond_chunk_embs=True, dhdeconv_shape='512,512,3',
              prefix=None, pf_name=None, **kwargs):
    """This is a helper function to add an argument group for hypernetwork-
    specific arguments to a given argument parser.

    Arguments specified in this function:
        - `hnet_type`
        - `hmlp_arch`
        - `cond_emb_size`
        - `chmlp_chunk_size`
        - `chunk_emb_size`
        - `use_cond_chunk_embs`
        - `hdeconv_shape`
        - `hdeconv_num_layers`
        - `hdeconv_filters`
        - `hdeconv_kernels`
        - `hdeconv_attention_layers`

    Args:
        parser (argparse.ArgumentParser): The parser to which an argument group
            should be added
        allowed_nets (list): List of allowed network identifiers. The following
            identifiers are considered (note, we also reference the network that
            each network type targets):

            - ``'hmlp'``: :class:`hnets.mlp_hnet.HMLP`
            - ``'chunked_hmlp'``: :class:`hnets.chunked_mlp_hnet.ChunkedHMLP`
            - ``'structured_hmlp'``:
              :class:`hnets.structured_mlp_hnet.StructuredHMLP`
            - ``'hdeconv'``: :class:`hnets.deconv_hnet.HDeconv`
            - ``'chunked_hdeconv'``:
              :class:`hnets.chunked_deconv_hnet.ChunkedHDeconv`
        dhmlp_arch (str): Default value of option `hmlp_arch`.
        show_cond_emb_size (bool): Whether the option `cond_emb_size` should be
            provided.
        dcond_emb_size (int): Default value of option `cond_emb_size`.
        dchmlp_chunk_size (int): Default value of option `chmlp_chunk_size`.
        dchunk_emb_size (int): Default value of option `chunk_emb_size`.
        show_use_cond_chunk_embs (bool): Whether the option
            `use_cond_chunk_embs` should be provided (if applicable to
            network types).
        dhdeconv_shape (str): Default value of option `hdeconv_shape`.
        prefix (str, optional): If arguments should be instantiated with a
            certain prefix. E.g., a setup requires several hypernetworks, that
            may need different settings. For instance: :code:`prefix='gen_'`.
        pf_name (str, optional): A name of type of hypernetwork for which that
            ``prefix`` is needed. For instance: :code:`prefix='generator'`.
        **kwargs: Keyword arguments to configure options that are common across
            main networks (note, a hypernet is just a special main network). See
            arguments of :func:`main_net_args`.

    Returns:
        (argparse._ArgumentGroup): The created argument group containing the
        desired options.
    """
    assert(prefix is None or pf_name is not None)

    for nt in allowed_nets:
        assert nt in ['hmlp', 'chunked_hmlp', 'structured_hmlp', 'hdeconv',
                      'chunked_hdeconv']

    heading = 'Hypernet options'

    if prefix is None:
        prefix = ''
        pf_name = ''
    else:
        heading = 'Hypernet options for %s network' % pf_name
        pf_name += ' '

    # Abbreviations.
    p = prefix
    n = pf_name

    ### CHypernet options
    agroup = parser.add_argument_group(heading)

    if len(allowed_nets) > 1:
        agroup.add_argument('--%shnet_type' % p, type=str,
                            default=allowed_nets[0],
                            help='Type of network to be used for this %s ' % n +
                                 'network. Default: %(default)s.',
                            choices=allowed_nets)

    if 'hmlp' in allowed_nets or 'chunked_hmlp' in allowed_nets or \
            'structured_hmlp' in allowed_nets:
        nets = '"hmlp", "chunked_hmlp" or "structured_hmlp"'
        agroup.add_argument('--%shmlp_arch' % p, type=str, default=dhmlp_arch,
                            help='If using a %s %s network, this will ' \
                                 % (nets, n) + 'specify the hidden layers ' +
                                 'via a comma-separated list of integers.' +
                                 'Note, if a "structured_hmlp" is used, you ' +
                                 'may enter multiple architectures separated ' +
                                 'by a semicolon. Default: %(default)s.')

    if show_cond_emb_size:
        agroup.add_argument('--%scond_emb_size' % p, type=int,
                            default=dcond_emb_size,
                            help='Size of conditional input embeddings ' +
                                 '(e.g., task embedding when using task-' +
                                 'condtioned hypernetworks). ' +
                                 'Default: %(default)s.')

    if 'chunked_hmlp' in allowed_nets:
        nets = '"chunked_hmlp"'
        agroup.add_argument('--%schmlp_chunk_size' % p, type=int,
                            default=dchmlp_chunk_size,
                            help='If using a %s %s network, this will ' \
                                 % (nets, n) + 'specify the chunk size. ' +
                                 'Hence, the target network weights are ' +
                                 'split into equally sized chunks of this ' +
                                 'size that are produced by a "small" ' +
                                 'shared hypernetwork. Default: %(default)s.')

    if 'chunked_hmlp' in allowed_nets or 'chunked_hdeconv' in allowed_nets:
        nets = '"chunked_hmlp", "chunked_hdeconv" or "structured_hmlp"'
        agroup.add_argument('--%schunk_emb_size' % p, type=str,
                            default=dchunk_emb_size,
                            help='If using a %s %s network, this will ' \
                                 % (nets, n) + 'specify the size of the ' +
                                 'chunk embeddings. Note, if a ' +
                                 '"structured_hmlp" is used, this option ' +
                                 'may be a comma-separated list of integers, ' +
                                 'denoting the chunk embedding size per ' +
                                 'internally maintained hypernetwork. ' +
                                 'Default: %(default)s.')

    if show_use_cond_chunk_embs and ('chunked_hmlp' in allowed_nets or \
            'chunked_hdeconv' in allowed_nets or \
            'structured_hmlp' in allowed_nets):
        nets = '"chunked_hmlp", "chunked_hdeconv" or "structured_hmlp"'
        agroup.add_argument('--%suse_cond_chunk_embs' % p,
                            action='store_true',
                            help='If using a %s %s network, this will make ' \
                                 % (nets,n) + 'chunk embeddings conditional. ' +
                                 'In this case, "%scond_emb_size" can ' % p +
                                 'be 0. Hence, there would be a separate ' +
                                 'set of chunk embeddings per condition.')

    if 'chunked_hdeconv' in allowed_nets or 'hdeconv' in allowed_nets:
        nets = '"hdeconv" or "chunked_hdeconv"'
        agroup.add_argument('--%shdeconv_shape' % p, type=str,
                            default=dhdeconv_shape,
                            help='If using a %s %s network, this will ' \
                                 % (nets, n) + 'specify the shape of the ' +
                                 '"hyper image" produced by the ' +
                                 'deconvolutional hypernetwork. I.e., a ' +
                                 'string of 2 or 3 comma-separated integers ' +
                                 'is expected. Default: %(default)s.')
        agroup.add_argument('--%shdeconv_num_layers' % p, type=int, default=5,
                            help='If using a %s %s network, this will ' \
                                 % (nets, n) + 'specify the number of layers ' +
                                 'in the hypernetwork. Default: %(default)s.')
        agroup.add_argument('--%shdeconv_filters' % p, type=str,
                            default='128,512,256,128',
                            help='If using a %s %s network, this is ' \
                                 % (nets, n) + 'a string of comma-separated '+
                                 'integers, each indicating the number of ' +
                                 'output channels for a layer in the-' +
                                 'deconvolutional hypernet. ' +
                                 'Default: %(default)s.')
        agroup.add_argument('--%shdeconv_kernels' % p, type=str, default=5,
                            help='If using a %s %s network, this is ' \
                                 % (nets, n) + 'a string of comma-separated ' +
                                 'integers, indicating kernel sizes in the ' +
                                 'deconvolutional hypernet. Note, to specify ' +
                                 'a distinct kernel size per dimension of ' +
                                 'each layer, just enter a list with twice ' +
                                 'the number of elements as convolutional ' +
                                 'layers in the hypernet. ' +
                                 'Default: %(default)s.')
        agroup.add_argument('--%shdeconv_attention_layers' % p, type=str,
                            default='1,3',
                            help='If using a %s %s network, this is ' \
                                 % (nets, n) + 'a string of comma-separated ' +
                                 'integers, indicating after which layers of ' +
                                 'the hypernet a self-attention unit should ' +
                                 'be added. Default: %(default)s.')

    # Note, if we wouldn't add an additional prefix, then potentially a main
    # network could have the same argument name as a hypernetwork, which would
    # cause a conflict.
    mnet_prefix = p + 'hnet_'
    _main_net_args_helper(agroup, prefix=mnet_prefix, pf_name=n, **kwargs)

    return agroup

def main_net_args(parser, allowed_nets=['mlp'],
                  dmlp_arch='100,100', dlenet_type='mnist_small',
                  dcmlp_arch='10,10', dcmlp_chunk_arch='10,10',
                  dcmlp_in_cdim=100, dcmlp_out_cdim=10, dcmlp_cemb_dim=8,
                  dresnet_block_depth=5, dresnet_channel_sizes='16,16,32,64',
                  dwrn_block_depth=4, dwrn_widening_factor=10,
                  diresnet_channel_sizes='64,64,128,256,512',
                  diresnet_blocks_per_group='2,2,2,2',
                  dsrnn_rec_layers='10', dsrnn_pre_fc_layers='',
                  dsrnn_post_fc_layers='', dsrnn_rec_type='lstm',
                  show_net_act=True, dnet_act='relu', show_no_bias=False,
                  show_dropout_rate=True, ddropout_rate=-1, show_specnorm=True,
                  show_batchnorm=True, show_no_batchnorm=False,
                  show_bn_no_running_stats=False, show_bn_distill_stats=False,
                  show_bn_no_stats_checkpointing=False,
                  prefix=None, pf_name=None):
    """This is a helper function for the function `parse_cmd_arguments` to add
    an argument group for options to a main network.

    Arguments specified in this function:
        - `net_type`
        - `fc_arch`
        - `mlp_arch`
        - `lenet_type`
        - `cmlp_arch`
        - `cmlp_chunk_arch`
        - `cmlp_in_cdim`
        - `cmlp_out_cdim`
        - `cmlp_cemb_dim`
        - `resnet_block_depth`
        - `resnet_channel_sizes`
        - `wrn_block_depth`
        - `wrn_widening_factor`
        - `wrn_use_fc_bias`
        - `iresnet_use_fc_bias`
        - `iresnet_channel_sizes`
        - `iresnet_blocks_per_group`
        - `iresnet_bottleneck_blocks`
        - `iresnet_projection_shortcut`
        - `srnn_rec_layers`
        - `srnn_pre_fc_layers`
        - `srnn_post_fc_layers`
        - `srnn_no_fc_out`
        - `srnn_rec_type`
        - `net_act`
        - `no_bias`
        - `dropout_rate`
        - `specnorm`
        - `batchnorm`
        - `no_batchnorm`
        - `bn_no_running_stats`
        - `bn_distill_stats`
        - `bn_no_stats_checkpointing`

    Args:
        parser (:class:`argparse.ArgumentParser`): The argument parser to which
            the argument group should be added.
        allowed_nets (list): List of allowed network identifiers. The following
            identifiers are considered (note, we also reference the network that
            each network type targets):

            - ``mlp``: :class:`mnets.mlp.MLP`
            - ``lenet``: :class:`mnets.lenet.LeNet`
            - ``resnet``: :class:`mnets.resnet.ResNet`
            - ``wrn``: :class:`mnets.wide_resnet.WRN`
            - ``iresnet``: :class:`mnets.resnet_imgnet.ResNetIN`
            - ``zenke``: :class:`mnets.zenkenet.ZenkeNet`
            - ``bio_conv_net``: :class:`mnets.bio_conv_net.BioConvNet`
            - ``chunked_mlp``: :class:`mnets.chunk_squeezer.ChunkSqueezer`
            - ``simple_rnn``: :class:`mnets.simple_rnn.SimpleRNN`

        dmlp_arch: Default value of option `mlp_arch`.
        dlenet_type: Default value of option `lenet_type`.
        dcmlp_arch: Default value of option `cmlp_arch`.
        dcmlp_chunk_arch: Default value of option `cmlp_chunk_arch`.
        dcmlp_in_cdim: Default value of option `cmlp_in_cdim`.
        dcmlp_out_cdim: Default value of option `cmlp_out_cdim`.
        dcmlp_cemb_dim: Default value of option `cmlp_cemb_dim`.
        dresnet_block_depth: Default value of option `resnet_block_depth`.
        dresnet_channel_sizes: Default value of option `resnet_channel_sizes`.
        dwrn_block_depth: Default value of option `wrn_block_depth`.
        dwrn_widening_factor: Default value of option `wrn_widening_factor`.
        diresnet_channel_sizes: Default value of option
            `iresnet_channel_sizes`.
        diresnet_blocks_per_group: Default value of option
            `iresnet_blocks_per_group`.
        dsrnn_rec_layers: Default value of option `srnn_rec_layers`.
        dsrnn_pre_fc_layers: Default value of option `srnn_pre_fc_layers`.
        dsrnn_post_fc_layers: Default value of option `srnn_post_fc_layers`.
        dsrnn_rec_type: Default value of option `srnn_rec_type`.
        show_net_act (bool): Whether the option `net_act` should be provided.
        dnet_act: Default value of option `net_act`.
        show_no_bias (bool): Whether the option `no_bias` should be provided.
        show_dropout_rate (bool): Whether the option `dropout_rate` should be
            provided.
        ddropout_rate: Default value of option ``dropout_rate``.
        show_specnorm (bool): Whether the option `specnorm` should be provided.
        show_batchnorm (bool): Whether the option `batchnorm` should be
            provided.
        show_no_batchnorm (bool): Whether the option `no_batchnorm` should be
            provided.
        show_bn_no_running_stats (bool): Whether the option
            `bn_no_running_stats` should be provided.
        show_bn_distill_stats (bool): Whether the option `bn_distill_stats`
            should be provided.
        show_bn_no_stats_checkpointing (bool): Whether the option
            `bn_no_stats_checkpointing` should be provided.
        prefix (optional): If arguments should be instantiated with a certain
            prefix. E.g., a setup requires several main network, that may need
            different settings. For instance: prefix=:code:`prefix='gen_'`.
        pf_name (optional): A name of the type of main net for which that prefix
            is needed. For instance: prefix=:code:`'generator'`.

    Returns:
        The created argument group, in case more options should be added.
    """
    assert prefix is None or pf_name is not None

    for nt in allowed_nets:
        assert nt in ['mlp', 'lenet', 'resnet', 'zenke', 'bio_conv_net',
                      'chunked_mlp', 'simple_rnn', 'wrn', 'iresnet']

    assert not show_batchnorm or not show_no_batchnorm

    heading = 'Main network options'

    if prefix is None:
        prefix = ''
        pf_name = ''
    else:
        heading = 'Main network options for %s network' % pf_name
        pf_name += ' '

    # Abbreviations.
    p = prefix
    n = pf_name

    ### Main network options.
    agroup = parser.add_argument_group(heading)

    if len(allowed_nets) > 1:
        agroup.add_argument('--%snet_type' % p, type=str,
                            default=allowed_nets[0],
                            help='Type of network to be used for this %s ' % n +
                                 'network. Default: %(default)s.',
                            choices=allowed_nets)

    if 'mlp' in allowed_nets:
        agroup.add_argument('--%smlp_arch' % p, type=str, default=dmlp_arch,
                            help='If using a "mlp" %s network, this will ' % n +
                                 'specify the hidden layers. ' +
                                 'Default: %(default)s.')

    if 'lenet' in allowed_nets:
        agroup.add_argument('--%slenet_type' % p, type=str, default=dlenet_type,
                            help='If using a "lenet" %s network, this ' % n +
                                 'will specify the architecture to be used. ' +
                                 'Default: %(default)s.',
                            choices=['mnist_small', 'mnist_large', 'cifar'])

    if 'chunked_mlp' in allowed_nets:
        agroup.add_argument('--%scmlp_arch' % p, type=str, default=dcmlp_arch,
                            help='If using a "chunked_mlp" %s network, ' % n +
                                 'this will specify the hidden layers of the ' +
                                 'MLP network mapping from the ' +
                                 'dimensionality reduced input to the ' +
                                 'output. Default: %(default)s.')

        agroup.add_argument('--%scmlp_chunk_arch' % p, type=str,
                            default=dcmlp_chunk_arch,
                            help='If using a "chunked_mlp" %s network, ' % n +
                                 'this will specify the hidden layers of the ' +
                                 'MLP network reducing the dimensionality of ' +
                                 'input chunks. Default: %(default)s.')

        agroup.add_argument('--%scmlp_in_cdim' % p, type=int, metavar='N',
                            default=dcmlp_in_cdim,
                            help='If using a "chunked_mlp" %s network, ' % n +
                                 'this will specify the input dimensionality ' +
                                 'of the network processing chunks (i.e., ' +
                                 'the input chunk size). Default: %(default)s.')

        agroup.add_argument('--%scmlp_out_cdim' % p, type=int, metavar='N',
                            default=dcmlp_out_cdim,
                            help='If using a "chunked_mlp" %s network, ' % n +
                                 'this will specify the output ' +
                                 'dimensionality of the network processing ' +
                                 'chunks (i.e., the desired chunk size). ' +
                                 'Default: %(default)s.')

        agroup.add_argument('--%scmlp_cemb_dim' % p, type=int, metavar='N',
                            default=dcmlp_cemb_dim,
                            help='If using a "chunked_mlp" %s network, ' % n +
                                 'this will specify the dimensionality of ' +
                                 'chunk embeddings. Default: %(default)s.')

    if 'resnet' in allowed_nets:
        agroup.add_argument('--%sresnet_block_depth' % p, type=int, metavar='N',
                            default=dresnet_block_depth,
                            help='If using a "resnet" %s network, ' %n+
                                 'this will specify the number of residual ' +
                                 'blocks. Default: %(default)s.')
        agroup.add_argument('--%sresnet_channel_sizes' % p, type=str,
                            default=dresnet_channel_sizes,
                            help='If using a "resnet" %s network, ' %n+
                                 'this will specify the number of feature ' +
                                 'maps. It has to be a list of 4 integers! ' +
                                 'Default: %(default)s.')

    if 'wrn' in allowed_nets:
        agroup.add_argument('--%swrn_block_depth' % p, type=int, metavar='N',
                            default=dwrn_block_depth,
                            help='If using a "wrn" %s network, ' %n+
                                 'this will specify the number of residual ' +
                                 'blocks. Default: %(default)s.')
        agroup.add_argument('--%swrn_widening_factor' %p, type=int, metavar='N',
                            default=dwrn_widening_factor,
                            help='If using a "wrn" %s network, ' % n +
                                 'this will specify the widening factor for ' +
                                 'the channels in the convolutional layers. ' +
                                 'Default: %(default)s.')
        agroup.add_argument('--%swrn_use_fc_bias' % p, action='store_true',
                            help='If using a "wrn" %s network, ' % n +
                                 'whenever "%sno_bias" is active, ' +
                                 'this will specifcy that a bias has to be ' +
                                 'used in the fully connected layers.')

    if 'iresnet' in allowed_nets:
        agroup.add_argument('--%siresnet_use_fc_bias' % p, action='store_true',
                            help='If using a "iresnet" %s network, ' % n +
                                 'whenever "%sno_bias" is active, ' +
                                 'this will specifcy that a bias has to be ' +
                                 'used in the fully connected layers.')
        agroup.add_argument('--%siresnet_channel_sizes' % p, type=str,
                            default=diresnet_channel_sizes,
                            help='If using a "iresnet" %s network, ' %n+
                                 'this will specify the number of feature ' +
                                 'maps. It has to be a list of 5 integers! ' +
                                 'Default: %(default)s.')
        agroup.add_argument('--%siresnet_blocks_per_group' % p, type=str,
                            default=diresnet_blocks_per_group,
                            help='If using a "iresnet" %s network, ' %n+
                                 'this will specify the number of blocks ' +
                                 'per group of convolutional layers. It has ' +
                                 'to be a list of 4 integers! ' +
                                 'Default: %(default)s.')
        agroup.add_argument('--%siresnet_bottleneck_blocks' % p,
                            action='store_true',
                            help='If using a "iresnet" %s network, ' % n +
                                 'this option decides whether normal blocks ' +
                                 'or bottleneck blocks are used.')
        agroup.add_argument('--%siresnet_projection_shortcut' % p,
                            action='store_true',
                            help='If using a "iresnet" %s network, ' % n +
                                 'this option decides whether skip ' +
                                 'connections are padded/subsampled or ' +
                                 'realized via 1x1 conv projections.')

    if 'simple_rnn' in allowed_nets:
        agroup.add_argument('--%ssrnn_rec_layers' % p, type=str,
                            default=dsrnn_rec_layers,
                            help='If using a "simple_rnn" %s network, ' % n +
                                 'this will specify the size of every ' +
                                 'hidden recurrent layer. This list can be ' +
                                 'left empty, if option ' +
                                 '"%ssrnn_no_fc_out" ' % p +
                                 'is enabled and therefore the output layer ' +
                                 'will be recurrent. Default: %(default)s.')
        agroup.add_argument('--%ssrnn_pre_fc_layers' % p, type=str,
                            default=dsrnn_pre_fc_layers,
                            help='If using a "simple_rnn" %s network, ' % n +
                                 'this will specify the sizes of all initial ' +
                                 'fully-connected latyers. If left empty, ' +
                                 'there will be no initial fully-connected ' +
                                 'layers and the first layer is going to be ' +
                                 'a recurrent layer. Default: %(default)s.')
        agroup.add_argument('--%ssrnn_post_fc_layers' % p, type=str,
                            default=dsrnn_post_fc_layers,
                            help='If using a "simple_rnn" %s network, ' % n +
                                 'this will specify the sizes of all final ' +
                                 'hidden fully-connected layers. Note, the ' +
                                 'output layer is also fully-connected, even ' +
                                 'if this option is left empty (except if ' +
                                 'option "%ssrnn_no_fc_out" ' % p + 'is ' +
                                 'used). Default: %(default)s.')
        agroup.add_argument('--%ssrnn_no_fc_out' % p, action='store_true',
                            help='If using a "simple_rnn" %s network, ' % n +
                                 'this option forces the last layer to be ' +
                                 'recurrent.')
        agroup.add_argument('--%ssrnn_rec_type' % p, type=str,
                            default=dsrnn_rec_type,
                            help='If using a "simple_rnn" %s network, ' % n +
                                 'this option determines which type of ' +
                                 'recurrent layer to be used. ' +
                                 'Default: %(default)s.',
                            choices=['lstm', 'elman'])

    _main_net_args_helper(agroup, show_net_act=show_net_act, dnet_act=dnet_act,
        show_no_bias=show_no_bias, show_dropout_rate=show_dropout_rate,
        ddropout_rate=ddropout_rate, show_specnorm=show_specnorm,
        show_batchnorm=show_batchnorm, show_no_batchnorm=show_no_batchnorm,
        show_bn_no_running_stats=show_bn_no_running_stats,
        show_bn_distill_stats=show_bn_distill_stats,
        show_bn_no_stats_checkpointing=show_bn_no_stats_checkpointing,
        prefix=p, pf_name=n)

    return agroup

def _main_net_args_helper(agroup, show_net_act=True, dnet_act='relu',
                          show_no_bias=False, show_dropout_rate=True,
                          ddropout_rate=-1, show_specnorm=True,
                          show_batchnorm=True, show_no_batchnorm=False,
                          show_bn_no_running_stats=False,
                          show_bn_distill_stats=False,
                          show_bn_no_stats_checkpointing=False,
                          prefix=None, pf_name=None):
    """Add general main-network arguments to ``agroup``.

    Helper function of :func:`main_net_args` to add arguments that are common
    to main networks and might also be applicable to other network types, like
    hypernetworks.
    """
    # Abbreviations.
    p = prefix
    n = pf_name

    # Note, if you want to add more activation function choices here, you have
    # to add them to the corresponding function `utils.misc.str_to_act` as well!
    if show_net_act:
        agroup.add_argument('--%snet_act' % p, type=str, default=dnet_act,
                        help='Activation function used in the %s network.' % n +
                             'If "linear", no activation function is used. ' +
                             'Default: %(default)s.',
                        choices=['linear', 'sigmoid', 'relu', 'elu', 'tanh'])

    if show_no_bias:
        agroup.add_argument('--%sno_bias' % p, action='store_true',
                        help='No biases will be used in the %s network. ' % n +
                             'Note, does not affect normalization (like ' +
                             'batchnorm).')

    if show_dropout_rate:
        agroup.add_argument('--%sdropout_rate' % p, type=float,
                            default=ddropout_rate,
                            help='Use dropout in the %s network with the ' % n +
                                 'given dropout probability (dropout is ' +
                                 'deactivated for a rate of -1). Default: ' +
                                 '%(default)s.')

    if show_specnorm:
        agroup.add_argument('--%sspecnorm' % p, action='store_true',
                            help='Enable spectral normalization in the ' +
                                 '%s network.' % n)

    ### Batchnorm related options.
    if show_batchnorm:
        agroup.add_argument('--%sbatchnorm' % p, action='store_true',
                            help='Enable batchnorm in the %s network.' % n)
    if show_no_batchnorm:
        agroup.add_argument('--%sno_batchnorm' % p, action='store_true',
                            help='Disable batchnorm in the %s network.' % n)

    if show_bn_no_running_stats:
        agroup.add_argument('--%sbn_no_running_stats' % p, action='store_true',
                            help='If batch normalization is used, then this ' +
                                 'option will deactivate the tracking ' +
                                 'of running statistics. Hence, statistics ' +
                                 'computed per batch will be used during ' +
                                 'evaluation.')

    if show_bn_distill_stats:
        agroup.add_argument('--%sbn_distill_stats' % p, action='store_true',
                            help='If batch normalization is used, ' +
                                 'then usually the running statistics are ' +
                                 'checkpointed for every task (e.g., in ' +
                                 'continual learning), which has linearly ' +
                                 'increasing memory requirements. If ' +
                                 'this option is activated, the running ' +
                                 'statistics will be distilled into the ' +
                                 'hypernetwork after training each task, ' +
                                 'such that only the statistics of the ' +
                                 'current and previous task have to be ' +
                                 'explicitly kept in memory')

    if show_bn_no_stats_checkpointing:
        agroup.add_argument('--%sbn_no_stats_checkpointing' % p,
                            action='store_true',
                            help='If batch normalization is used, then ' +
                                 'this option will prevent the checkpointing ' +
                                 'of batchnorm statistics for every task.' +
                                 'In this case, one set of statistics is ' +
                                 'used for all tasks.')

    return agroup

def init_args(parser, custom_option=True, show_normal_init=True,
              show_hyper_fan_init=False):
    """This is a helper method of the method `parse_cmd_arguments` to add
    an argument group for options regarding network initialization.

    Arguments specified in this function:
        - `custom_network_init`
        - `normal_init`
        - `std_normal_init`
        - `std_normal_temb`
        - `std_normal_emb`
        - `hyper_fan_init`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        custom_option (bool): Whether the option `custom_network_init` should be
            provided.
        show_normal_init (bool): Whether the option `normal_init` and
            `std_normal_init` should be provided.
        show_hyper_fan_init (bool): Whether the option `hyper_fan_init` should
            be provided.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Weight initialization.
    agroup = parser.add_argument_group('Network initialization options')
    if custom_option:
        # This option becomes important if posthoc custom init is not that
        # trivial anymore (e.g., if networks use batchnorm). Then, the network
        # init must be customized for each such network.
        agroup.add_argument('--custom_network_init', action='store_true',
                            help='Whether network parameters should be ' +
                                 'initialized in a custom way. If this flag ' +
                                 'is set, then Xavier initialization is ' +
                                 'applied to weight tensors (zero ' +
                                 'initialization for bias vectors). The ' +
                                 'initialization of chunk and task ' +
                                 'embeddings is independent of this option.')
    if show_normal_init:
        agroup.add_argument('--normal_init', action='store_true',
                            help='Use weight initialization from a zero-mean ' +
                                 'normal with std defined by the argument ' +
                                 '\'std_normal_init\'. Otherwise, Xavier ' +
                                 'initialization is used. Biases are ' +
                                 'initialized to zero.')
        agroup.add_argument('--std_normal_init', type=float, default=0.02,
                            help='If normal initialization is used, this ' +
                                 'will be the standard deviation used. ' +
                                 'Default: %(default)s.')
    agroup.add_argument('--std_normal_temb', type=float, default=1.,
                        help='Std when initializing task embeddings. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--std_normal_emb', type=float, default=1.,
                        help='If a chunked hypernetwork is used (including ' +
                             'self-attention hypernet), then this will be ' +
                             'the std of their embeddings its ' +
                             'initialization. Default: %(default)s.')

    if show_hyper_fan_init:
        agroup.add_argument('--hyper_fan_init', action='store_true',
                            help='Use the hyperfan-initialization method ' +
                                 'for hypernetworks. Note, for hypernetwork ' +
                                 'that use chunking, this requires a proper ' +
                                 'setting of "std_normal_temb". When using ' +
                                 'this option, then "std_normal_emb" has no ' +
                                 'effect just like all other init options ' +
                                 'that might effect hypernet initialization.')

    return agroup

def miscellaneous_args(parser, big_data=True, synthetic_data=False,
                       show_plots=False, no_cuda=False, dout_dir=None,
                       show_publication_style=False):
    """This is a helper method of the method `parse_cmd_arguments` to add
    an argument group for miscellaneous arguments.

    Arguments specified in this function:
        - `num_workers`
        - `out_dir`
        - `use_cuda`
        - `no_cuda`
        - `loglevel_info`
        - `deterministic_run`
        - `publication_style`
        - `show_plots`
        - `data_random_seed`
        - `random_seed`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        big_data: If the program processes big datasets that need to be loaded
            from disk on the fly. In this case, more options are provided.
        synthetic_data: If data is randomly generated, then we want to decouple
            this randomness from the training randomness.
        show_plots: Whether the option `show_plots` should be provided.
        no_cuda: If True, the user has to explicitly set the flag `--use_cuda`
            rather than using CUDA by default.
        dout_dir (optional): Default value of option `out_dir`. If :code:`None`,
            the default value will be `./out/run_<YY>-<MM>-<DD>_<hh>-<mm>-<ss>`
            that contains the current date and time.
        show_publication_style: Whether the option `publication_style` should be
            provided.

    Returns:
        The created argument group, in case more options should be added.
    """
    if dout_dir is None:
        dout_dir = './out/run_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    ### Miscellaneous arguments
    agroup = parser.add_argument_group('Miscellaneous options')
    if big_data:
        agroup.add_argument('--num_workers', type=int, metavar='N', default=8,
                            help='Number of workers per dataset loader. ' +
                                 'Default: %(default)s.')
    agroup.add_argument('--out_dir', type=str, default=dout_dir,
                        help='Where to store the outputs of this simulation.')
    if no_cuda:
        agroup.add_argument('--use_cuda', action='store_true',
                            help='Flag to enable GPU usage.')
    else:
        agroup.add_argument('--no_cuda', action='store_true',
                            help='Flag to disable GPU usage.')
    agroup.add_argument('--loglevel_info', action='store_true',
                        help='If the console log level should be raised ' +
                             'from DEBUG to INFO.')
    agroup.add_argument('--deterministic_run', action='store_true',
                        help='Enable deterministic CuDNN behavior. Note, that' +
                             'CuDNN algorithms are not deterministic by ' +
                             'default and results might not be reproducible ' +
                             'unless this option is activated. Note, that ' +
                             'this may slow down training significantly!')  
    if show_publication_style:
        agroup.add_argument('--publication_style', action='store_true',
                            help='Whether plots should be publication-ready.')
    if show_plots:
        agroup.add_argument('--show_plots', action='store_true',
                            help='Whether plots should be shown.')
    if synthetic_data:
        agroup.add_argument('--data_random_seed', type=int, metavar='N',
                            default=42,
                            help='The data is randomly generated at every ' +
                             'run. This seed ensures that the randomness ' +
                             'during data generation is decoupled from the ' +
                             'training randomness. Default: %(default)s.')
    agroup.add_argument('--random_seed', type=int, metavar='N', default=42,
                        help='Random seed. Default: %(default)s.')
    return agroup

def eval_args(parser, dval_iter=500, show_val_batch_size=False,
              dval_batch_size=256, show_val_set_size=False, dval_set_size=0):
    """This is a helper method of the method `parse_cmd_arguments` to add
    an argument group for validation and testing options.

    Arguments specified in this function:
        - `val_iter`
        - `val_batch_size`
        - `val_set_size`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        dval_iter (int): Default value of argument `val_iter`.
        show_val_batch_size (bool): Whether the `val_batch_size` argument should
            be shown.
        dval_batch_size (int): Default value of argument `val_batch_size`.
        show_val_set_size (bool): Whether the `val_set_size` argument should be
            shown.
        dval_set_size (int): Default value of argument `val_set_size`.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Eval arguments
    agroup = parser.add_argument_group('Evaluation options')
    agroup.add_argument('--val_iter', type=int, metavar='N', default=dval_iter,
                        help='How often the validation should be performed ' +
                             'during training. Default: %(default)s.')

    if show_val_batch_size:
        agroup.add_argument('--val_batch_size', type=int, metavar='N',
                            default=dval_batch_size,
                            help='Batch size during validation/testing. ' +
                                 'Default: %(default)s.')

    if show_val_set_size:
        agroup.add_argument('--val_set_size', type=int, metavar='N',
                            default=dval_set_size,
                            help='If unequal "0", a validation set will be ' +
                                 'extracted from the training set (hence, ' +
                                 'reducing the size of the training set). ' +
                                 'This can be useful for efficiency reasons ' +
                                 'if the validation set is smaller than the ' +
                                 'test set. If the training is influenced by ' +
                                 'generalization measures on the data (e.g., ' +
                                 'a learning rate schedule), then it is good ' +
                                 'practice to use a validation set for this. ' +
                                 'It is also desirable to select ' +
                                 'hyperparameters based on a validation set, ' +
                                 'if possible. Default: %(default)s.')

    return agroup

def train_args(parser, show_lr=False, dlr=0.1, show_epochs=False, depochs=-1,
               dbatch_size=32, dn_iter=100001, show_use_adam=False,
               dadam_beta1=0.9, show_use_rmsprop=False, show_use_adadelta=False,
               show_use_adagrad=False, show_clip_grad_value=False,
               show_clip_grad_norm=False, show_adam_beta1=False,
               show_momentum=True):
    """This is a helper method of the method `parse_cmd_arguments` to add
    an argument group for options to configure network training.

    Arguments specified in this function:
        - `batch_size`
        - `n_iter`
        - `epochs`
        - `lr`
        - `momentum`
        - `weight_decay`
        - `use_adam`
        - `adam_beta1`
        - `use_rmsprop`
        - `use_adadelta`
        - `use_adagrad`
        - `clip_grad_value`
        - `clip_grad_norm`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        show_lr: Whether the `lr` - learning rate - argument should be shown.
            Might not be desired if individual learning rates per optimizer
            should be specified.
        dlr: Default value for option `lr`.
        show_epochs: Whether the `epochs` argument should be shown.
        depochs: Default value for option `epochs`.
        dbatch_size: Default value for option `batch_size`.
        dn_iter: Default value for option `n_iter`.
        show_use_adam: Whether the `use_adam` argument should be shown. Will
            also show the `adam_beta1` argument.
        dadam_beta1: Default value for option `adam_beta1`.
        show_use_rmsprop: Whether the `use_rmsprop` argument should be shown.
        show_use_adadelta: Whether the `use_adadelta` argument should be shown.
        show_use_adagrad: Whether the `use_adagrad` argument should be shown.
        show_clip_grad_value: Whether the `clip_grad_value` argument should be
            shown.
        show_clip_grad_norm: Whether the `clip_grad_norm` argument should be
            shown.
        show_adam_beta1: Whether the `adam_beta1` argument should be
            shown. Note, this argument is also shown when ``show_use_adam`` is
            ``True``.
        show_momentum: Whether the `momentum` argument should be
            shown.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Training options.
    agroup = parser.add_argument_group('Training options')
    agroup.add_argument('--batch_size', type=int, metavar='N',
                        default=dbatch_size,
                        help='Training batch size. Default: %(default)s.')
    agroup.add_argument('--n_iter', type=int, metavar='N', default=dn_iter,
                        help='Number of training iterations per task. ' +
                             'Default: %(default)s.')
    if show_epochs:
        agroup.add_argument('--epochs', type=int, metavar='N', default=depochs,
                            help='Number of epochs per task. If -1, "n_iter" ' +
                                 'is used instead. Default: %(default)s.')
    if show_lr:
        agroup.add_argument('--lr', type=float, default=dlr,
                            help='Learning rate of optimizer(s). Default: ' +
                                 '%(default)s.')
    if show_momentum:
        agroup.add_argument('--momentum', type=float, default=0.0,
                            help='Momentum of the optimizer (only used in ' +
                                 'SGD and RMSprop). Default: %(default)s.')
    agroup.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay of the optimizer(s). Default: ' +
                             '%(default)s.')
    if show_use_adam:
        agroup.add_argument('--use_adam', action='store_true',
                            help='Use Adam rather than SGD optimizer.')
    if show_use_adam or show_adam_beta1:
        agroup.add_argument('--adam_beta1', type=float, default=dadam_beta1,
                        help='The "beta1" parameter when using torch.optim.' +
                             'Adam as optimizer. Default: %(default)s.')
    if show_use_rmsprop:
        agroup.add_argument('--use_rmsprop', action='store_true',
                            help='Use RMSprop rather than SGD optimizer.')
    if show_use_adadelta:
        agroup.add_argument('--use_adadelta', action='store_true',
                            help='Use Adadelta rather than SGD optimizer.')
    if show_use_adagrad:
        agroup.add_argument('--use_adagrad', action='store_true',
                            help='Use Adagrad rather than SGD optimizer.')

    if show_clip_grad_value:
        agroup.add_argument('--clip_grad_value', type=float, default=-1,
                        help='If not "-1", gradients will be clipped using ' +
                             '"torch.nn.utils.clip_grad_value_". Default: ' +
                             '%(default)s.')

    if show_clip_grad_norm:
        agroup.add_argument('--clip_grad_norm', type=float, default=-1,
                        help='If not "-1", gradient norms will be clipped ' +
                             'using "torch.nn.utils.clip_grad_norm_". ' +
                             'Default: %(default)s.')

    return agroup

def cl_args(parser, show_beta=True, dbeta=0.01, show_from_scratch=False,
            show_multi_head=False, show_cl_scenario=False,
            show_split_head_cl3=True, dcl_scenario=1,
            show_num_tasks=False, dnum_tasks=1, show_num_classes_per_task=False,
            dnum_classes_per_task=2):
    """This is a helper method of the method `parse_cmd_arguments` to add
    an argument group for typical continual learning arguments.

    Arguments specified in this function:
        - `beta`
        - `train_from_scratch`
        - `multi_head`
        - `cl_scenario`
        - `split_head_cl3`
        - `num_tasks`
        - `num_classes_per_task`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        show_beta: Whether option `beta` should be shown.
        dbeta: Default value of option `beta`.
        show_from_scratch: Whether option `train_from_scratch` should be shown.
        show_multi_head: Whether option `multi_head` should be shown.
        show_cl_scenario: Whether option `cl_scenario` should be shown.
        show_split_head_cl3: Whether option `split_head_cl3` should be shown.
            Only has an effect if ``show_cl_scenario`` is ``True``.
        dcl_scenario: Default value of option `cl_scenario`.
        show_num_tasks: Whether option `num_tasks` should be shown.
        dnum_tasks: Default value of option `num_tasks`.
        show_num_classes_per_task: Whether option `show_num_classes_per_task`
            should be shown.
        dnum_classes_per_task: Default value of option `dnum_classes_per_task`.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Continual learning options.
    agroup = parser.add_argument_group('Continual learning options')

    if show_beta:
        agroup.add_argument('--beta', type=float, default=dbeta,
                            help='Trade-off for the CL regularizer. ' +
                                 'Default: %(default)s.')

    if show_from_scratch:
        agroup.add_argument('--train_from_scratch', action='store_true',
                        help='If set, all networks are recreated after ' +
                             'training on each task. Hence, training starts ' +
                             'from scratch.')

    if show_multi_head:
        agroup.add_argument('--multi_head', action='store_true',
                        help='Use a multihead setting, where each task has ' +
                             'its own output head.')

    if show_cl_scenario:
        agroup.add_argument('--cl_scenario', type=int, default=dcl_scenario,
                            help='Continual learning scenarios according to ' +
                                 'https://arxiv.org/pdf/1809.10635.pdf. ' +
                                 '"1" - Task-incremental learning; ' +
                                 '"2" - Domain-incremental learning; ' +
                                 '"3" - Class-incremental learning. ' +
                                 'Default: %(default)s.',
                            choices=[1, 2, 3])

    if show_cl_scenario and show_split_head_cl3:
        agroup.add_argument('--split_head_cl3', action='store_true',
                            help='CL scenario 3 (CL3, cmp. "cl_scenario") ' +
                                 'originally requires to compute the softmax ' +
                                 'across all output neurons. Though, if a ' +
                                 'task-conditioned hypernetwork is used, the ' +
                                 'task identity had to be inferred a priori. ' +
                                 'Hence, in CL2 and CL3 we always know the ' +
                                 'task identity, which is why we can also ' +
                                 'compute the softmax over single output ' +
                                 'heads in CL3 using this option.')

    if show_num_tasks:
        agroup.add_argument('--num_tasks', type=int, metavar='N',
                            default=dnum_tasks,
                            help='Number of tasks. Default: %(default)s.')

    if show_num_classes_per_task:
        agroup.add_argument('--num_classes_per_task', type=int, metavar='N',
                            default=dnum_classes_per_task,
                            help='Number of classes per task. ' +
                                 'Default: %(default)s.')

    return agroup

def gan_args(parser):
    """This is a helper method of the method `parse_cmd_arguments` to add
    an argument group for options to configure the generator and discriminator
    network.

    .. deprecated:: 1.0
        Please use method :func:`main_net_args` and :func:`generator_args`
        instead.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.

    Returns:
        The created argument group, in case more options should be added.
    """
    warn('Please use method "main_net_args" and "generator_args" instead.',
         DeprecationWarning)

    ### Training options.
    agroup = parser.add_argument_group('Generator and discriminator network ' +
                                       'options')
    for n, m in [('gen', 'generator'), ('dis', 'discriminator')]:
        agroup.add_argument('--%s_net_type' % n, type=str, default='fc',
                            help='Type of network to be used for the %s ' % m +
                                 'network. Default: %(default)s.',
                            choices=['fc'])
        agroup.add_argument('--%s_fc_arch' % n, type=str, default='100,100',
                            help='If using a "fc" %s network, this will ' % m +
                                 'specify the hidden layers. ' +
                                 'Default: %(default)s.')
        agroup.add_argument('--%s_net_act' % n, type=str, default='relu',
                        help='Activation function used in the %s network.' % m +
                             'If "linear", no activation function is used. ' +
                             'Default: %(default)s.',
                        choices=['linear', 'sigmoid', 'relu', 'elu'])
        agroup.add_argument('--%s_dropout_rate' % n, type=float, default=-1,
                        help='Use dropout in the %s with the given ' % m +
                             'dropout probability (dropout is deactivated ' +
                             'for a rate of -1). Default: %(default)s.')
        agroup.add_argument('--%s_batchnorm' % n, action='store_true',
                            help='Enable batchnorm in the %s.' % m)
        agroup.add_argument('--%s_specnorm' % n, action='store_true',
                            help='Enable spectral normalization in the %s.' % m)

    agroup.add_argument('--latent_dim', type=int, metavar='N', default=3,
                        help='Dimensionality of the latent vector (noise ' +
                             'input to the generator. Default: %(default)s.')
    agroup.add_argument('--latent_std', type=float, default=1.0,
                        help='Standard deviation of the latent space. ' +
                             'Default: %(default)s.')
    return agroup

def generator_args(agroup, dlatent_dim=3):
    """This is a helper method of the method `parse_cmd_arguments` (or more
    specifically an auxillary method to :func:`train_args`) to add arguments to
    an argument group for options specific to a main network that should act as
    a generator.

    Arguments specified in this function:
        - `latent_dim`
        - `latent_std`

    Args:
        agroup: The argument group returned by, for instance, function
            :func:`main_net_args`.
        dlatent_dim: Default value of option `latent_dim`.
    """
    ### Generator options.
    agroup.add_argument('--latent_dim', type=int, metavar='N',
                        default=dlatent_dim,
                        help='Dimensionality of the latent vector (noise ' +
                             'input to the generator). Default: %(default)s.')
    agroup.add_argument('--latent_std', type=float, default=1.0,
                        help='Standard deviation of the latent space. ' +
                             'Default: %(default)s.')

def data_args(parser, show_disable_data_augmentation=False, show_data_dir=False,
              ddata_dir='.'):
    """This is a helper method of the function `parse_cmd_arguments` to add
    an argument group for typical dataset related options.

    Arguments specified in this function:
        - `disable_data_augment`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        show_disable_data_augmentation (bool): Whether option
            `disable_data_augmentation` should be shown.
        show_data_dir (bool): Whether option `data_dir` should be shown.
        ddata_dir (str): Default value of option `data_dir`.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Continual learning options.
    agroup = parser.add_argument_group('Data-specific options')

    if show_disable_data_augmentation:
        agroup.add_argument('--disable_data_augmentation', action='store_true',
                        help='If activated, no data augmentation will be ' +
                             'applied. Note, this option only affects ' +
                             'datasets that have preprocessing implemented ' +
                             '(such CIFAR-10).')

    if show_data_dir:
        agroup.add_argument('--data_dir', type=str, default=ddata_dir,
                        help='The directory from which the dataset should be ' +
                             'loaded. Default: %(default)s.')

    return agroup

def check_invalid_argument_usage(args):
    """This method checks for common conflicts when using the arguments defined
    by methods in this module.

    The following things will be checked:

        - Based on the optimizer choices specified in :func:`train_args`, we
          assert here that only one optimizer is selected at a time.
        - Assert that `clip_grad_value` and `clip_grad_norm` are not set at the
          same time.
        - Assert that `split_head_cl3` is only set for `cl_scenario=3`
        - Assert that the arguments specified in function :func:`main_net_args`
          are correctly used.

          .. note::
              The checks can't handle prefixes yet.

    Args:
        args: The parsed command-line arguments, i.e., the output of method
            :meth:`argparse.ArgumentParser.parse_args`.

    Raises:
        ValueError: If invalid argument combinations are used.
    """
    optim_args = ['use_adam', 'use_rmsprop', 'use_adadelta', 'use_adagrad']
    for i, o1 in enumerate(optim_args):
        if not hasattr(args, o1):
            continue

        for j, o2 in enumerate(optim_args):
            if i == j or not hasattr(args, o2):
                continue

            if getattr(args, o1) and getattr(args, o2):
                raise ValueError('Cannot simultaneously use 2 optimizers ' +
                                 '(arguments "%s" and "%s").' % (o1, o2))

    if hasattr(args, 'clip_grad_value') and hasattr(args, 'clip_grad_norm'):
        if args.clip_grad_value != -1 and args.clip_grad_norm != -1:
            raise ValueError('Cannot simultaneously clip gradiant values and ' +
                             'gradient norm.')

    if hasattr(args, 'cl_scenario') and hasattr(args, 'split_head_cl3'):
        if args.cl_scenario != 3 and args.split_head_cl3:
            raise ValueError('Flag "split_head_cl3" may only be set when ' +
                             'running CL scenario 3 (CL3)!')

    # TODO if `custom_network_init` is used but deactivated, then the other init
    # options have no effect -> user should be warned.

    ### Check consistent use of arguments from `main_net_args`.
    # FIXME These checks don't deal with prefixes yet!
    if hasattr(args, 'net_type') and hasattr(args, 'dropout_rate'):
        if args.net_type in ['resnet', 'iresnet', 'bio_conv_net',
                             'simple_rnn'] and \
                args.dropout_rate != -1:
            warn('Dropout is not implemented for network %s.' % args.net_type)

    if hasattr(args, 'net_type') and hasattr(args, 'specnorm'):
        if args.net_type in ['lenet', 'resnet', 'wrn', 'iresnet', 'zenke',
                             'bio_conv_net', 'simple_rnn'] and \
                args.specnorm:
            warn('Spectral Normalization is not implemented for network %s.'
                 % args.net_type)

    if hasattr(args, 'net_type') and hasattr(args, 'net_act'):
        if args.net_type in ['lenet', 'resnet', 'wrn', 'iresnet', 'zenke'] and \
                args.net_act != 'relu':
            warn('%s network uses ReLU activation functions. ' % args.net_type +
                 'Ignoring option "net_act".')

        if args.net_type in ['bio_conv_net']: # and args.net_act != 'tanh':
            warn('%s network uses Tanh activation functions. ' % args.net_type +
                 'Ignoring option "net_act".')

        if args.net_type in ['simple_rnn'] and \
                args.net_act not in ['relu', 'tanh']:
            raise ValueError('%s network  only support ReLU ' % args.net_type +
                             'and tanh activation functions.')

    if hasattr(args, 'net_type') and hasattr(args, 'no_bias'):
        if args.net_type in ['lenet', 'zenke', 'bio_conv_net'] and \
                args.no_bias:
            warn('%s network always uses biases!' % args.net_type)

    bn_used = False
    if hasattr(args, 'batchnorm'):
        bn_used = args.batchnorm
    elif hasattr(args, 'no_batchnorm'):
        bn_used = not args.no_batchnorm
    else:
        # We don't know whether it is used.
        bn_used = None

    if bn_used is not None and bn_used and hasattr(args, 'net_type'):
        if args.net_type in ['lenet', 'zenke', 'bio_conv_net', 'simple_rnn']:
            warn('Batch Normalization is not implemented for network %s.'
                 % args.net_type)

    if bn_used is not None and hasattr(args, 'bn_no_running_stats'):
        if not bn_used and args.bn_no_running_stats:
            warn('Option "bn_no_running_stats" has no effect if batch ' +
                 'normalization not activated.')

    if bn_used is not None and hasattr(args, 'bn_distill_stats'):
        if not bn_used and args.bn_distill_stats:
            warn('Option "bn_distill_stats" has no effect if batch ' +
                 'normalization not activated.')

    if bn_used is not None and hasattr(args, 'bn_no_stats_checkpointing'):
        if not bn_used and args.bn_no_stats_checkpointing:
            warn('Option "bn_no_stats_checkpointing" has no effect if batch ' +
                 'normalization not activated.')

    if hasattr(args, 'bn_no_stats_checkpointing') and \
            hasattr(args, 'bn_no_running_stats') and \
            args.bn_no_stats_checkpointing and args.bn_no_running_stats:
        raise ValueError('Options "bn_no_stats_checkpointing" and ' +
                         '"bn_no_running_stats" are not compatible')
    if hasattr(args, 'bn_no_stats_checkpointing') and \
            hasattr(args, 'bn_distill_stats') and \
            args.bn_no_stats_checkpointing and args.bn_distill_stats:
        raise ValueError('Options "bn_no_stats_checkpointing" and ' +
                         '"bn_distill_stats" are not compatible')
    if hasattr(args, 'bn_no_running_stats') and \
            hasattr(args, 'bn_distill_stats') and \
            args.bn_no_running_stats and args.bn_distill_stats:
        raise ValueError('Options "bn_no_running_stats" and ' +
                         '"bn_distill_stats" are not compatible')

if __name__ == '__main__':
    pass


