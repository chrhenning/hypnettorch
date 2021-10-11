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
# @title          :utils/torch_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :09/11/2019
# @version        :1.0
# @python_version :3.6.8
"""
A collection of helper functions that should capture common functionalities
needed when working with PyTorch.
"""
import math
import numpy as np
import torch
from torch import nn
import types

def init_params(weights, bias=None):
    """Initialize the weights and biases of a linear or (transpose) conv layer.

    Note, the implementation is based on the method "reset_parameters()",
    that defines the original PyTorch initialization for a linear or
    convolutional layer, resp. The implementations can be found here:

        https://git.io/fhnxV

        https://git.io/fhnx2

    Args:
        weights: The weight tensor to be initialized.
        bias (optional): The bias tensor to be initialized.
    """
    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)

def get_optimizer(params, lr, momentum=0, weight_decay=0, use_adam=False,
                  adam_beta1=0.9, use_rmsprop=False, use_adadelta=False,
                  use_adagrad=False, pgroup_ids=None):
    """Create an optimizer instance for the given set of parameters. Default
    optimizer is :class:`torch.optim.SGD`.

    Args:
        params (list): The parameters passed to the optimizer.
        lr: Learning rate.
        momentum (optional): Momentum (only applicable to
            :class:`torch.optim.SGD` and :class:`torch.optim.RMSprop`.
        weight_decay (optional): L2 penalty.
        use_adam: Use :class:`torch.optim.Adam` optimizer.
        adam_beta1: First parameter in the `betas` tuple that is passed to the
            optimizer :class:`torch.optim.Adam`:
            :code:`betas=(adam_beta1, 0.999)`.
        use_rmsprop: Use :class:`torch.optim.RMSprop` optimizer.
        use_adadelta: Use :class:`torch.optim.Adadelta` optimizer.
        use_adagrad: Use :class:`torch.optim.Adagrad` optimizer.
        pgroup_ids (list, optional): If passed, a list of integers of the same
            length as params is expected. In this case, each integer states to
            which parameter group the corresponding parameter in ``params``
            shall belong. Parameter groups may have different optimizer
            settings. Therefore, options like ``lr``, ``momentum``,
            ``weight_decay``, ``adam_beta1`` may be lists in this case that have
            a length corresponding to the number of parameter groups.

    Returns:
        Optimizer instance.
    """
    use_sgd = not use_adam and not use_rmsprop and not use_adadelta \
        and not use_adagrad

    if isinstance(params, types.GeneratorType):
        params = list(params)

    # Transform list of parameter tensors `params` into list of dictionaries.
    if pgroup_ids is None:
        pgroup_ids = [0] * len(params)
        npgroups = 1
    else:
        assert len(pgroup_ids) == len(params)
        npgroups = max(pgroup_ids) + 1

    plist = params
    params = []

    # Initialize parameter group dictionaries.
    for i in range(npgroups):
        pdict = {}
        pdict['params'] = []
        pdict['lr'] = lr[i] if isinstance(lr, (list, tuple)) else lr
        pdict['weight_decay'] = weight_decay[i] \
            if isinstance(weight_decay, (list, tuple)) else weight_decay
        if use_adam:
            ab1 = adam_beta1[i] if isinstance(adam_beta1, (list, tuple)) \
                else adam_beta1
            pdict['betas'] = [ab1, 0.999]
        if use_sgd or use_rmsprop:
            pdict['momentum'] = momentum[i] \
                if isinstance(momentum, (list, tuple)) else momentum
        params.append(pdict)

    # Fill parameter groups.
    for pgid, p in zip(pgroup_ids, plist):
        params[pgid]['params'].append(p)


    if use_adam:
        optimizer = torch.optim.Adam(params)
    elif use_rmsprop:
        optimizer = torch.optim.RMSprop(params)
    elif use_adadelta:
        optimizer = torch.optim.Adadelta(params)
    elif use_adagrad:
        optimizer = torch.optim.Adagrad(params)
    else:
        assert use_sgd
        optimizer = torch.optim.SGD(params)

    return optimizer

def lambda_lr_schedule(epoch):
    """Multiplicative Factor for Learning Rate Schedule.

    Computes a multiplicative factor for the initial learning rate based
    on the current epoch. This method can be used as argument
    ``lr_lambda`` of class :class:`torch.optim.lr_scheduler.LambdaLR`.

    The schedule is inspired by the Resnet CIFAR-10 schedule suggested
    here https://keras.io/examples/cifar10_resnet/.

    Args:
        epoch (int): The number of epochs

    Returns:
        lr_scale (float32): learning rate scale
    """
    lr_scale = 1.
    if epoch > 180:
        lr_scale = 0.5e-3
    elif epoch > 160:
        lr_scale = 1e-3
    elif epoch > 120:
        lr_scale = 1e-2
    elif epoch > 80:
        lr_scale = 1e-1
    return lr_scale

class CutoutTransform(object):
    """Randomly mask out one or more patches from an image.

    The cutout transformation as preprocessing step has been proposed by

        DeVries et al., `Improved Regularization of Convolutional Neural \
Networks with Cutout <https://arxiv.org/abs/1708.04552>`__, 2017.

    The original implementation can be found `here <https://github.com/\
uoguelph-mlrg/Cutout/blob/master/util/cutout.py>`__.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    # This code of this class has been copied from (accessed 04/08/2020):
    #   https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    #
    # NOTE Our copyright and license does not apply for this function.
    # We use this code WITHOUT ANY WARRANTIES.
    #
    # The code is licensed according to
    # Educational Community License, Version 2.0 (ECL-2.0)
    #   https://github.com/uoguelph-mlrg/Cutout/blob/master/LICENSE.md
    #
    # Copyright 2017 Terrance DeVries, Raeid Saqur
    # Licensed under the Educational Community License, Version 2.0
    # (the "License"); you may not use this file except in compliance with the
    # License. You may obtain a copy of the License at
    #
    # http://www.osedu.org/licenses /ECL-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
    # WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
    # License for the specific language governing permissions and limitations
    # under the License.
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """Perform cutout to given image.

        Args:
            img (Tensor): Tensor image of size ``(C, H, W)``.

        Returns:
            (torch.Tensor): Image with ``n_holes`` of dimension 
                ``length x length`` cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

if __name__ == '__main__':
    pass


