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
# @title          :mnets/lenet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :02/28/2020
# @version        :1.0
# @python_version :3.6.10
"""
LeNet
-----

This module contains a general classifier template and a LeNet-like network
to classify either MNIST or CIFAR-10 images. The network is implemented in a
way that it might not have trainable parameters. Instead, the network weights
would have to be passed to the ``forward`` method. This makes the usage of a
hypernetwork (a network that generates the weights of another network)
particularly easy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hypnettorch.mnets.classifier_interface import Classifier
from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils.torch_utils import init_params

class LeNet(Classifier):
    """The network consists of two convolutional layers followed by two fully-
    connected layers. See implementation for details.

    LeNet was originally introduced in

        "Gradient-based learning applied to document recognition", LeCun et
        al., 1998.

    Though, the implementation provided here has several difference compared
    to the original LeNet architecture (e.g., the LeNet-5 architecture):

    - There is no special connectivity map before the second convolutional
      layer as described by table 1 in the original paper.
    - The dimensions of layers and their activation functions are dfferent.
    - The original LeNet-5 has a third fully connected layer with 1x1 kernels.

    We mainly use this modified LeNet architecture for MNIST:

    - A small architecture with only 21,840 weights.
    - A larger architecture with 431,080 weights.

    Both of these architectures are typically used for MNIST nowadays.

    Note, a variant of this architecture is also used for CIFAR-10, e.g. in

        "Bayesian Convolutional Neural Networks with Bernoulli Approximate
        Variational Inference", Gal et al., 2015.

    and

        "Multiplicative Normalizing Flows for Variational Bayesian Neural
        Networks", Louizos et al., 2017.

    In both these works, the dimensions of the weight parameters are:

    .. code-block:: python

        main_dims=[[192,3,5,5],[192],[192,192,5,5],[192],[1000,4800],
                   [1000],[10,1000],[10]],

    which is an architecture with 5,747,394 weights. Note, the authors used
    dropout in different configurations, e.g., after each layer, only after
    the fully-connected layer or no dropout at all.

    Args:
        in_shape (tuple or list): The shape of an input sample.

            .. note::
                We assume the Tensorflow format, where the last entry
                denotes the number of channels.
        num_classes (int): The number of output neurons.
        verbose (bool): Allow printing of general information about the
            generated network (such as number of weights).
        arch (str): The architecture to be employed. The following options are
            available:

            - ``'mnist_small'``: A small LeNet with 21,840 weights suitable
              for MNIST
            - ``'mnist_large'``: A larger LeNet with 431,080 weights suitable
              for MNIST
            - ``'cifar'``: A huge LeNet with 5,747,394 weights designed for
              CIFAR-10.
        no_weights (bool): If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.
        init_weights (optional): This option is for convinience reasons.
            The option expects a list of parameter values that are used to
            initialize the network weights. As such, it provides a
            convinient way of initializing a network with a weight draw
            produced by the hypernetwork.
        dropout_rate (float): If ``-1``, no dropout will be applied. Otherwise a
            number between 0 and 1 is expected, denoting the dropout rate.

            Dropout will be applied after the convolutional layers
            (before pooling) and after the first fully-connected layer
            (after the activation function).
        **kwargs: Keyword arguments regarding context modulation. This class
            can process the same context-modulation related arguments as class
            :class:`mnets.mlp.MLP`. One may additionally specify the argument
            ``context_mod_apply_pixel_wise`` (see class
            :class:`mnets.resnet.ResNet`).
    """
    _ARCHITECTURES = {
        'mnist_small': [[10,1,5,5],[10],[20,10,5,5],[20],
                        [50,320],[50],[10,50],[10]],
        'mnist_large': [[20,1,5,5],[20],[50,20,5,5],[50],[500,800],[500],
                        [10,500],[10]],
        'cifar': [[192,3,5,5],[192],[192,192,5,5],[192],[1000,4800],[1000],
                  [10,1000],[10]]
        }

    def __init__(self, in_shape=(28, 28, 1), num_classes=10, verbose=True,
                 arch='mnist_large', no_weights=False, init_weights=None,
                 dropout_rate=-1, #0.5
                 **kwargs):
        super(LeNet, self).__init__(num_classes, verbose)

        self._in_shape = in_shape
        assert arch in LeNet._ARCHITECTURES.keys()
        self._chosen_arch = LeNet._ARCHITECTURES[arch]
        if num_classes != 10:
            self._chosen_arch[-2][0] = num_classes
            self._chosen_arch[-1][0] = num_classes

        # Sanity check, given current implementation.
        if arch.startswith('mnist'):
            if not in_shape[0] == in_shape[1] == 28:
                raise ValueError('MNIST LeNet architectures expect input ' +
                                 'images of size 28x28.')
        else:
            if not in_shape[0] == in_shape[1] == 32:
                raise ValueError('CIFAR LeNet architectures expect input ' +
                                 'images of size 32x32.')

        ### Parse or set context-mod arguments ###
        rem_kwargs = MainNetInterface._parse_context_mod_args(kwargs)
        if len(rem_kwargs) > 0:
            raise ValueError('Keyword arguments %s unknown.' % str(rem_kwargs))
        # Since this is a conv-net, we may also want to add the following.
        if 'context_mod_apply_pixel_wise' not in kwargs.keys():
            kwargs['context_mod_apply_pixel_wise'] = False

        self._use_context_mod = kwargs['use_context_mod']
        self._context_mod_inputs = kwargs['context_mod_inputs']
        self._no_last_layer_context_mod = kwargs['no_last_layer_context_mod']
        self._context_mod_no_weights = kwargs['context_mod_no_weights']
        self._context_mod_post_activation = \
            kwargs['context_mod_post_activation']
        self._context_mod_gain_offset = kwargs['context_mod_gain_offset']
        self._context_mod_gain_softplus = kwargs['context_mod_gain_softplus']
        self._context_mod_apply_pixel_wise = \
            kwargs['context_mod_apply_pixel_wise']

        ### Setup class attributes ###
        assert(init_weights is None or \
               (not no_weights or not self._context_mod_no_weights))
        self._no_weights = no_weights
        self._dropout_rate = dropout_rate

        self._has_bias = True
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer!
        self._mask_fc_out = True
        self._has_linear_out = True

        self._param_shapes = []
        self._param_shapes_meta = []
        self._internal_params = None if no_weights and \
            self._context_mod_no_weights else nn.ParameterList()
        self._hyper_shapes_learned = None \
            if not no_weights and not self._context_mod_no_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        if dropout_rate != -1:
            assert(dropout_rate >= 0. and dropout_rate <= 1.)
            # FIXME `nn.Dropout2d` zeroes out whole feature maps. Is that really
            # desired here?
            self._drop_conv1 = nn.Dropout2d(p=dropout_rate)
            self._drop_conv2 = nn.Dropout2d(p=dropout_rate)
            self._drop_fc1 = nn.Dropout(p=dropout_rate)

        ### Define and initialize context mod layers/weights ###
        self._context_mod_layers = nn.ModuleList() if self._use_context_mod \
            else None

        if self._use_context_mod:
            cm_layer_inds = []
            cm_shapes = [] # Output shape of all context-mod layers.
            if self._context_mod_inputs:
                cm_shapes.append([in_shape[2], *in_shape[:2]])
                # We reserve layer zero for input context-mod. Otherwise, there
                # is no layer zero.
                cm_layer_inds.append(0)

            layer_out_shapes = self._compute_layer_out_sizes()
            # Context-modulation is applied after the pooling layers.
            # So we delete the shapes of the conv-layer outputs and keep the
            # ones of the pooling layer outputs.
            del layer_out_shapes[2]
            del layer_out_shapes[0]
            cm_shapes.extend(layer_out_shapes)
            cm_layer_inds.extend(range(2, 2*len(layer_out_shapes)+1, 2))
            if self._no_last_layer_context_mod:
                cm_shapes = cm_shapes[:-1]
                cm_layer_inds = cm_layer_inds[:-1]

            if not self._context_mod_apply_pixel_wise:
                # Only scalar gain and shift per feature map!
                for i, s in enumerate(cm_shapes):
                    if len(s) == 3:
                        cm_shapes[i] = [s[0], 1, 1]

            self._add_context_mod_layers(cm_shapes, cm_layers=cm_layer_inds)

        ### Define and add conv- and fc-layer weights.
        for i, s in enumerate(self._chosen_arch):
            if not no_weights:
                self._internal_params.append(nn.Parameter(torch.Tensor(*s),
                                                          requires_grad=True))
                if len(s) == 1:
                    self._layer_bias_vectors.append(self._internal_params[-1])
                else:
                    self._layer_weight_tensors.append(self._internal_params[-1])
            else:
                self._hyper_shapes_learned.append(s)
                self._hyper_shapes_learned_ref.append(len(self.param_shapes))

            self._param_shapes.append(s)
            self._param_shapes_meta.append({
                'name': 'weight' if len(s) != 1 else 'bias',
                'index': -1 if no_weights else len(self._internal_params)-1,
                'layer': 2 * (i // 2) + 1
            })

        ### Initialize weights.
        if init_weights is not None:
            assert len(init_weights) == len(self.weights)
            for i in range(len(init_weights)):
                assert np.all(np.equal(list(init_weights[i].shape),
                                       self.weights[i].shape))
                self.weights[i].data = init_weights[i]
        else:
            for i in range(len(self._layer_weight_tensors)):
                init_params(self._layer_weight_tensors[i],
                            self._layer_bias_vectors[i])

        ### Print user info.
        if verbose:
            if self._use_context_mod:
                cm_param_shapes = []
                for cm_layer in self.context_mod_layers:
                    cm_param_shapes.extend(cm_layer.param_shapes)
                cm_num_weights = \
                    MainNetInterface.shapes_to_num_weights(cm_param_shapes)

            print('Creating a LeNet with %d weights' % self.num_params
                  + (' (including %d weights associated with-' % cm_num_weights
                     + 'context modulation)' if self._use_context_mod else '')
                  + '.'
                  + (' The network uses dropout.' if dropout_rate != -1 \
                     else ''))

        self._is_properly_setup()

    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            weights (list or dict): See argument ``weights`` of method
                :meth:`mnets.mlp.MLP.forward`.
            condition (int, optional): If provided, then this argument will be
                passed as argument ``ckpt_id`` to the method
                :meth:`utils.context_mod_layer.ContextModLayer.forward`.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # FIXME Code copied from MLP its `forward` method.
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
        n_cm = self._num_context_mod_shapes()

        if weights is None:
            weights = self.weights

            if self._use_context_mod:
                cm_weights = weights[:n_cm]
                int_weights = weights[n_cm:]
            else:
                cm_weights = None
                int_weights = weights
        else:
            int_weights = None
            cm_weights = None

            if isinstance(weights, dict):
                assert('internal_weights' in weights.keys() or \
                       'mod_weights' in weights.keys())
                if 'internal_weights' in weights.keys():
                    int_weights = weights['internal_weights']
                if 'mod_weights' in weights.keys():
                    cm_weights = weights['mod_weights']
            else:
                if self._use_context_mod and \
                        len(weights) == n_cm:
                    cm_weights = weights
                else:
                    assert len(weights) == len(self.param_shapes)
                    if self._use_context_mod:
                        cm_weights = weights[:n_cm]
                        int_weights = weights[n_cm:]
                    else:
                        int_weights = weights

            if self._use_context_mod and cm_weights is None:
                if self._context_mod_no_weights:
                    raise Exception('Network was generated without weights ' +
                        'for context-mod layers. Hence, they must be passed ' +
                        'via the "weights" option.')
                cm_weights = self.weights[:n_cm]
            if int_weights is None:
                if self._no_weights:
                    raise Exception('Network was generated without internal ' +
                        'weights. Hence, they must be passed via the ' +
                        '"weights" option.')
                if self._context_mod_no_weights:
                    int_weights = self.weights
                else:
                    int_weights = self.weights[n_cm:]

            # Note, context-mod weights might have different shapes, as they
            # may be parametrized on a per-sample basis.
            if self._use_context_mod:
                assert len(cm_weights) == n_cm
            int_shapes = self.param_shapes[n_cm:]
            assert len(int_weights) == len(int_shapes)
            for i, s in enumerate(int_shapes):
                assert np.all(np.equal(s, list(int_weights[i].shape)))

        cm_ind = 0
        # Split context-mod weights per context-mod layer.
        if cm_weights is not None:
            cm_weights_layer = []
            cm_start = 0
            for cm_layer in self.context_mod_layers:
                cm_end = cm_start + len(cm_layer.param_shapes)
                cm_weights_layer.append(cm_weights[cm_start:cm_end])
                cm_start = cm_end

        #######################
        ### Parse condition ###
        #######################

        cmod_cond = None

        if condition is not None:
            assert isinstance(condition, int)
            cmod_cond = condition

            # FIXME We always require context-mod weight above, but
            # we can't pass both (a condition and weights) to the
            # context-mod layers.
            # An unelegant solution would be, to just set all
            # context-mod weights to None.
            raise NotImplementedError('CM-conditions not implemented!')
            cm_weights_layer = [None] * len(cm_weights_layer)

        ###########################
        ### Forward Computation ###
        ###########################
        ### Helper function to handle context-mod and non-linearities.
        def modulate_layer(h):
            """Compute context-modulation and non-linearity.

            The order if the following:

            context-mod (if pre-activation) -> non-linearity ->
            context-mod (if post-activation)

            This method increments the index ``cm_ind``.

            Args:
                h: Input activity.

            Returns:
                Output of layer.
            """
            nonlocal cm_ind

            # Context-dependent modulation (pre-activation).
            if self._use_context_mod and \
                    not self._context_mod_post_activation:
                h = self._context_mod_layers[cm_ind].forward(h,
                    weights=cm_weights_layer[cm_ind], ckpt_id=cmod_cond)
                cm_ind += 1

            # Non-linearity
            h = F.relu(h)

            # Context-dependent modulation (post-activation).
            if self._use_context_mod and self._context_mod_post_activation:
                h = self._context_mod_layers[cm_ind].forward(h,
                    weights=cm_weights_layer[cm_ind], ckpt_id=cmod_cond)
                cm_ind += 1

            return h

        x = x.view(-1, *self._in_shape)
        x = x.permute(0, 3, 1, 2)
        h = x

        # Context-dependent modulation of inputs directly.
        if self._use_context_mod and self._context_mod_inputs:
            h = self._context_mod_layers[cm_ind].forward(h,
                weights=cm_weights_layer[cm_ind], ckpt_id=cmod_cond)
            cm_ind += 1

        h = F.conv2d(h, int_weights[0], bias=int_weights[1])
        if self._dropout_rate != -1:
            h = self._drop_conv1(h)
        h = F.max_pool2d(h, 2)
        h = modulate_layer(h)

        h = F.conv2d(h, int_weights[2], bias=int_weights[3])
        if self._dropout_rate != -1:
            h = self._drop_conv2(h)
        h = F.max_pool2d(h, 2)
        h = modulate_layer(h)

        h = h.reshape(-1, int_weights[4].size()[1])

        h = F.linear(h, int_weights[4], bias=int_weights[5])
        h = modulate_layer(h)
        # FIXME Before we applied context-modulation after dropout, since
        # dropout was before the non-linearity and not after.
        if self._dropout_rate != -1:
            h = self._drop_fc1(h)

        h = F.linear(h, int_weights[6], bias=int_weights[7])
        
        # Context-dependent modulation in output layer.
        if self._use_context_mod and not self._no_last_layer_context_mod:
            h = self._context_mod_layers[cm_ind].forward(h,
                weights=cm_weights_layer[cm_ind], ckpt_id=cmod_cond)

        return h

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This network does not have any distillation targets.

        Returns:
            ``None``
        """
        return None

    def _compute_layer_out_sizes(self):
        """Compute the output shapes of all layers in this network.

        This method will compute the output shape of each layer in this network,
        including the output layer, which just corresponds to the number of
        classes.

        Returns:
            (list): A list of shapes (lists of integers). The first entry will
            correspond to the shape of the output of the first convolutional
            layer. The last entry will correspond to the output shape.

            .. note::
                Output shapes of convolutional layers will adhere PyTorch
                convention, i.e., ``[C, H, W]``, where ``C`` denotes the channel
                dimension.

            .. note::
                Pooling layers are considered as individual layers.
        """
        in_shape = self._in_shape
        assert len(self._chosen_arch) == 8
        fs = [self._chosen_arch[0][0], self._chosen_arch[2][0]]
        ks = [self._chosen_arch[0][2], 2, self._chosen_arch[2][2], 2]
        assert self._chosen_arch[0][2] == self._chosen_arch[0][3] and \
            self._chosen_arch[2][2] == self._chosen_arch[2][3]
        pd = 0 # all paddings are 0.

        # Note, `in_shape` is in Tensorflow layout.
        assert(len(in_shape) == 3)
        in_shape = [in_shape[2], *in_shape[:2]]

        ret = []

        C, H, W = in_shape

        # First conv layer (stride 1).
        C = fs[0]
        H = (H - ks[0] + 2*pd) // 1 + 1
        W = (W - ks[0] + 2*pd) // 1 + 1
        ret.append([C, H, W])
        # First pooling layer (stride == kernel size)
        H = (H - ks[1] + 2*pd) // ks[1] + 1
        W = (W - ks[1] + 2*pd) // ks[1] + 1
        ret.append([C, H, W])

        # Second conv layer (stride 1).
        C = fs[1]
        H = (H - ks[2] + 2*pd) // 1 + 1
        W = (W - ks[2] + 2*pd) // 1 + 1
        ret.append([C, H, W])
        # Second pooling layer (stride == kernel size)
        H = (H - ks[3] + 2*pd) // ks[3] + 1
        W = (W - ks[3] + 2*pd) // ks[3] + 1
        ret.append([C, H, W])

        assert C * H * W == self._chosen_arch[4][1]

        # First fully-connected layer.
        ret.append([self._chosen_arch[6][1]])
        # Output layer.
        assert self._num_classes == self._chosen_arch[6][0]
        ret.append([self._num_classes])

        return ret

if __name__ == '__main__':
    pass


