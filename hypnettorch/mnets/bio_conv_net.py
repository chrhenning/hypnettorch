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
# @title          :mnets/bio_conv_net.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :10/30/2019
# @version        :1.0
# @python_version :3.6.8
"""
A bio-plausible convolutional network for CIFAR
-----------------------------------------------

The module :mod:`mnets.bio_conv_net` implements a simple biologically-plausible
network with convolutional and fully-connected layers. The bio-plausibility
arises through the usage of conv-layers without weight sharing, i.e., layers
from class :class:`utils.local_conv2d_layer.LocalConv2dLayer`. The network
specification has been taken from the following paper

    `Bartunov et al., "Assessing the Scalability of Biologically-Motivated Deep
    Learning Algorithms and Architectures", NeurIPS 2018.
    <http://papers.nips.cc/paper/8148-assessing-the-scalability-of-biologically\
-motivated-deep-learning-algorithms-and-architectures>`_

in which this kind of network has been termed "locally-connected network".

In particular, we consider the network architecture specified in table 3 on page
13 for the CIFAR dataset.

.. autosummary::

    hypnettorch.mnets.bio_conv_net.BioConvNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hypnettorch.mnets.classifier_interface import Classifier
from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils.context_mod_layer import ContextModLayer
from hypnettorch.utils.local_conv2d_layer import LocalConv2dLayer
from hypnettorch.utils.torch_utils import init_params

class BioConvNet(Classifier):
    """Implementation of a locally-connected network for CIFAR.

    The network consists of 3 bio-plausible convolutional layers
    (using class :class:`utils.local_conv2d_layer.LocalConv2dLayer`)
    followed by two fully-connected layers.

    Assume conv layers are specified by the tuple ``(K x K, C, S, P)``, where
    ``K`` denotes the kernel size, ``C`` the number of channels, ``S`` the
    stride and ``P`` the padding. The network is defined as follows

        - Bio-conv layer (5 x 5, 64, 2, 0)
        - Bio-conv layer (5 x 5, 128, 2, 0)
        - Bio-conv layer (3 x 3, 256, 1, 1)
        - FC layer with 1024 outputs
        - FC layer with 10 outputs

    Note, the padding for the first two convolutional layers was not specified
    in the paper, so we just assumed it to be zero.

    The **network output will be linear**, so we do not apply the softmax
    inside the :meth:`forward` method.

    Note, the paper states that ``tanh`` was used in all networks as
    non-linearity. Therefore, we use this non-linearity too.

    Args:
        in_shape: The shape of an input sample.

            .. note::
                We assume the Tensorflow format, where the last entry
                denotes the number of channels.
        num_classes: The number of output neurons.
        no_weights (bool): If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.
        init_weights (optional): This option is for convinience reasons.
            The option expects a list of parameter values that are used to
            initialize the network weights. As such, it provides a
            convinient way of initializing a network with a weight draw
            produced by the hypernetwork.

            Note, internal weights (see
            :attr:`mnets.mnet_interface.MainNetInterface.weights`) will be
            affected by this argument only.
        use_context_mod (bool): Add context-dependent modulation layers
            :class:`utils.context_mod_layer.ContextModLayer` after the linear
            computation of each layer.
        context_mod_inputs (bool): Whether context-dependent modulation should
            also be applied to network intpus directly. I.e., assume
            :math:`\mathbf{x}` is the input to the network. Then the first
            network operation would be to modify the input via
            :math:`\mathbf{x} \cdot \mathbf{g} + \mathbf{s}` using context-
            dependent gain and shift parameters.

            Note:
                Argument applies only if ``use_context_mod`` is ``True``.
        no_last_layer_context_mod (bool): If ``True``, context-dependent
            modulation will not be applied to the output layer.

            Note:
                Argument applies only if ``use_context_mod`` is ``True``.
        context_mod_no_weights (bool): The weights of the context-mod layers
            (:class:`utils.context_mod_layer.ContextModLayer`) are treated
            independently of the option ``no_weights``.
            This argument can be used to decide whether the context-mod
            parameters (gains and shifts) are maintained internally or
            externally.

            Note:
                Check out argument ``weights`` of the :meth:`forward` method
                on how to correctly pass weights to the network that are
                externally maintained.
        context_mod_post_activation (bool): Apply context-mod layers after the
            activation function (``activation_fn``) in hidden layer rather than
            before, which is the default behavior.

            Note:
                This option only applies if ``use_context_mod`` is ``True``.

            Note:
                This option does not affect argument ``context_mod_inputs``.

            Note:
                Note, there is no non-linearity applied to the output layer,
                such that this argument has no effect there.
        context_mod_gain_offset (bool): Activates option ``apply_gain_offset``
            of class :class:`utils.context_mod_layer.ContextModLayer` for all
            context-mod layers that will be instantiated.
        context_mod_gain_softplus (bool): Activates option
            ``apply_gain_softplus`` of class
            :class:`utils.context_mod_layer.ContextModLayer` for all
            context-mod layers that will be instantiated.
        context_mod_apply_pixel_wise (bool): If ``False``, the context-dependent
            modulation applies a scalar gain and shift to all feature maps in
            the output of a convolutional layer. When activating this option,
            the gain and shift will be a per-pixel parameter in all feature
            maps.

            To be more precise, consider the output of a convolutional layer
            of shape ``[C,H,W]``. If ``False``, there will be ``C`` gain and
            shift parameters for such a layer. Upon activating this option, the
            number of gain and shift parameters for such a layer will increase
            to ``C x H x W``.
    """
    def __init__(self, in_shape=(32, 32, 3), num_classes=10, no_weights=False,
                 init_weights=None, use_context_mod=False,
                 context_mod_inputs=False, no_last_layer_context_mod=False,
                 context_mod_no_weights=False,
                 context_mod_post_activation=False,
                 context_mod_gain_offset=False,
                 context_mod_gain_softplus=False,
                 context_mod_apply_pixel_wise=False):
        super(BioConvNet, self).__init__(num_classes, True)

        assert(len(in_shape) == 3)
        # FIXME This assertion is not mandatory but a sanity check that the user
        # uses the Tensorflow layout.
        assert(in_shape[2] in [1, 3])
        assert(init_weights is None or \
               (not no_weights or not context_mod_no_weights))
        self._in_shape = in_shape
        self._no_weights = no_weights
        self._use_context_mod = use_context_mod
        self._context_mod_inputs = context_mod_inputs
        self._no_last_layer_context_mod = no_last_layer_context_mod
        self._context_mod_no_weights = context_mod_no_weights
        self._context_mod_post_activation = context_mod_post_activation
        self._context_mod_gain_offset = context_mod_gain_offset
        self._context_mod_gain_softplus = context_mod_gain_softplus
        self._context_mod_apply_pixel_wise = context_mod_apply_pixel_wise

        self._has_bias = True
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True
        self._has_linear_out = True

        self._param_shapes = []
        self._weights = None if no_weights and context_mod_no_weights \
            else nn.ParameterList()
        self._hyper_shapes_learned = None \
            if not no_weights and not context_mod_no_weights else []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        # Shapes of output activities for context-modulation, if used.
        cm_shapes = [] # Output shape of all layers.
        if context_mod_inputs:
            cm_shapes.append([in_shape[2], *in_shape[:2]])

        ### Define and initialize all conv and linear layers
        ### Bio-conv layers.
        H = in_shape[0]
        W = in_shape[1]
        C_in = in_shape[2]

        C = [64, 128, 256]
        K = [5, 5, 3]
        S = [2, 2, 1]
        P = [0, 0, 1]

        self._conv_layer = []

        for i, C_out in enumerate(C):
            self._conv_layer.append(LocalConv2dLayer(C_in, C_out, H, W, K[i],
                stride=S[i], padding=P[i], no_weights=no_weights))
            H = self._conv_layer[-1].out_height
            W = self._conv_layer[-1].out_width

            cm_shapes.append([C_out, H, W])

            C_in = C_out

            self._param_shapes.extend(self._conv_layer[-1].param_shapes)
            if no_weights:
                self._hyper_shapes_learned.extend( \
                    self._conv_layer[-1].param_shapes)
            else:
                self._weights.extend(self._conv_layer[-1].weights)

                assert(len(self._conv_layer[-1].weights) == 2)
                self._layer_weight_tensors.append( \
                    self._conv_layer[-1].filters)
                self._layer_bias_vectors.append( \
                    self._conv_layer[-1].bias)

        ### Linear layers
        n_in = H * W * C_out
        assert(n_in == 6400)
        n = [1024, num_classes]
        for i, n_out in enumerate(n):
            W_shape = [n_out, n_in]
            b_shape = [n_out]

            # Note, that the last layer shape might not be used for context-
            # modulation.
            if i < (len(n)-1) or not no_last_layer_context_mod:
                cm_shapes.append([n_out])

            n_in = n_out

            self._param_shapes.extend([W_shape, b_shape])
            if no_weights:
                self._hyper_shapes_learned.extend([W_shape, b_shape])
            else:
                W = nn.Parameter(torch.Tensor(*W_shape), requires_grad=True)
                b = nn.Parameter(torch.Tensor(*b_shape), requires_grad=True)

                init_params(W, b)

                self._weights.extend([W, b])
                self._layer_weight_tensors.append(W)
                self._layer_bias_vectors.append(b)

        ### Define and initialize context mod weights.
        self._context_mod_layers = nn.ModuleList() if use_context_mod else None
        self._context_mod_shapes = [] if use_context_mod else None
        self._context_mod_weights = nn.ParameterList() if use_context_mod \
            else None

        if use_context_mod:
            if not context_mod_apply_pixel_wise:
                # Only scalar gain and shift per feature map!
                for i, s in enumerate(cm_shapes):
                    if len(s) == 3:
                        cm_shapes[i] = [s[0], 1, 1]

            for i, s in enumerate(cm_shapes):
                cmod_layer = ContextModLayer(s,
                    no_weights=context_mod_no_weights,
                    apply_gain_offset=context_mod_gain_offset,
                    apply_gain_softplus=context_mod_gain_softplus)
                self._context_mod_layers.append(cmod_layer)

                self._context_mod_shapes.extend(cmod_layer.param_shapes)
                if not context_mod_no_weights:
                    self._context_mod_weights.extend(cmod_layer.weights)

            # We always had the context mod weights/shapes at the beginning of
            # our list attributes.
            self._param_shapes = self._context_mod_shapes + self._param_shapes
            if context_mod_no_weights:
                self._hyper_shapes_learned = self._context_mod_shapes + \
                    self._hyper_shapes_learned
            else:
                tmp = self._weights
                self._weights = nn.ParameterList(self._context_mod_weights)
                for w in tmp:
                    self._weights.append(w)

        ### Apply custom init if given.
        if init_weights is not None:
            assert(len(self.weights) == len(init_weights))
            for i in range(len(init_weights)):
                assert(np.all(np.equal(list(init_weights[i].shape),
                                       list(self._weights[i].shape))))
                self._weights[i].data = init_weights[i]

        ### Print user info.
        num_weights = MainNetInterface.shapes_to_num_weights( \
            self._param_shapes)
        if use_context_mod:
            cm_num_weights = MainNetInterface.shapes_to_num_weights( \
                self._context_mod_shapes)

        print('Creating bio-plausible convnet with %d weights' % num_weights
              + (' (including %d weights associated with-' % cm_num_weights
                 + 'context modulation)' if use_context_mod else '') + '.')

        self._is_properly_setup()

    def forward(self, x, weights=None, distilled_params=None, condition=None,
                collect_activations=False):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            x: Input image.

                .. note::
                    We assume the Tensorflow format, where the last entry
                    denotes the number of channels.
            weights (list or dict): If a list of parameter tensors is given and
                context modulation is used (see argument ``use_context_mod`` in
                constructor), then these parameters are interpreted as context-
                modulation parameters if the length of ``weights`` equals
                :code:`2*len(net.context_mod_layers)`. Otherwise, the length is
                expected to be equal to the length of the attribute
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.

                Alternatively, a dictionary can be passed with the possible
                keywords ``internal_weights`` and ``mod_weights``. Each keyword
                is expected to map onto a list of tensors.
                The keyword ``internal_weights`` refers to all weights of this
                network except for the weights of the context-modulation layers.
                The keyword ``mod_weights``, on the other hand, refers
                specifically to the weights of the context-modulation layers.
                It is not necessary to specify both keywords.
            condition (int, optional): Will be passed as argument ``ckpt_id``
                to the method
                :meth:`utils.context_mod_layer.ContextModLayer.forward` for
                all context-mod layers in this network.
            collect_activations (bool, optional): If one wants to return the
                activations in the network. This information can be used for
                credit assignment later on, in case an alternative to PyTorch
                its :mod:`torch.autograd` should be used.
        Returns:
            (:class:`torch.Tensor` or tuple): Tuple containing:

            - **y**: The output of the network.
            - **layer_activation** (optional): The activations of the network.
              Only returned if ``collect_activations`` was set to ``True``. The
              list will contain the activations of all convolutional and linear
              layers.
        """
        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
        # FIXME code mostly copied from MLP forward method.
        n_cm = 0 if self.context_mod_layers is None else \
            2 * len(self.context_mod_layers)

        if weights is None:
            weights = self.weights

            if self._use_context_mod:
                cm_weights = weights[:n_cm]
                int_weights = weights[n_cm:]
            else:
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
                    assert(len(weights) == len(self.param_shapes))
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
                assert(len(cm_weights) == len(self._context_mod_shapes))
            int_shapes = self.param_shapes[n_cm:]
            assert(len(int_weights) == len(int_shapes))
            for i, s in enumerate(int_shapes):
                assert(np.all(np.equal(s, list(int_weights[i].shape))))

        ###############################################
        ### Extract weight tensors and bias vectors ###
        ###############################################
        w_weights = []
        b_weights = []
        for i, p in enumerate(int_weights):
            if self.has_bias and i % 2 == 1:
                b_weights.append(p)
            else:
                w_weights.append(p)

        ########################
        ### Parse condition ###
        #######################

        cmod_cond = condition

        ###########################
        ### Forward Computation ###
        ###########################
        cm_ind = 0
        layer_ind = 0

        x = x.view(-1, *self._in_shape)
        x = x.permute(0, 3, 1, 2)
        h = x

        activations = []

        # Context-dependent modulation of inputs directly.
        if self._use_context_mod and self._context_mod_inputs:
            h = self._context_mod_layers[cm_ind].forward(h,
                weights=cm_weights[2*cm_ind:2*cm_ind+2], ckpt_id=cmod_cond)
            cm_ind += 1

        # Convolutional layers.
        for i, conv_layer in enumerate(self._conv_layer):
            h = conv_layer.forward(h, weights=[w_weights[layer_ind],
                                               b_weights[layer_ind]])
            layer_ind += 1

            if collect_activations:
                activations.append(h.clone())

            # Non-linearity (if context-dependent mod is applied post non-lin.)
            if self._context_mod_post_activation:
                h = torch.tanh(h)

            # Context-dependent modulation.
            if self._use_context_mod:
                h = self._context_mod_layers[cm_ind].forward(h,
                    weights=cm_weights[2*cm_ind:2*cm_ind+2],
                    ckpt_id=cmod_cond)
                cm_ind += 1

            # Non-linearity (if context-dependent mod is applied pre non-lin.)
            if not self._context_mod_post_activation:
                h = torch.tanh(h)
            
        # Flatten feature maps.
        h = h.view(h.size(0), -1)

        # Linear layers.
        assert(len(w_weights) == layer_ind + 2)
        for _ in range(2):
            W = w_weights[layer_ind]
            if self.has_bias:
                b = b_weights[layer_ind]
            else:
                b = None

            # Linear layer.
            h = F.linear(h, W, bias=b)

            if collect_activations:
                activations.append(h.clone())

            # Do not consider the output layer.
            if layer_ind < len(w_weights) - 1:
                # Non-linearity (if context-dependent mod is applied post
                # non-lin.)
                if self._context_mod_post_activation:
                    h = torch.tanh(h)

                # Context-dependent modulation.
                if self._use_context_mod:
                    h = self._context_mod_layers[cm_ind].forward(h,
                        weights=cm_weights[2*cm_ind:2*cm_ind+2],
                        ckpt_id=cmod_cond)
                    cm_ind += 1

                # Non-linearity (if context-dependent mod is applied pre
                # non-lin.)
                if not self._context_mod_post_activation:
                    h = torch.tanh(h)
            
            if collect_activations and layer_ind == len(w_weights) - 2:
                last_hidden = h

            layer_ind += 1

        # Context-dependent modulation in output layer.
        if self._use_context_mod and not self._no_last_layer_context_mod:
            h = self._context_mod_layers[cm_ind].forward(h,
                weights=cm_weights[2*cm_ind:2*cm_ind+2], ckpt_id=cmod_cond)

        if collect_activations:
            return h, activations, last_hidden
        else:
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

if __name__ == '__main__':
    pass



