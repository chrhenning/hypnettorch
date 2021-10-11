# TODOs

* The [hnet regularizer](../utils/hnet_regularizer) can't deal with online computation of targets correctly if `hyper_shapes_distilled` is non-empty for the main network. Note, the distillation target for the most recent task should always be obtained via method `distillation_targets` after training on that task.
* The constructor code of class [MLP](mlp.py) is quite ugly as we always added more and more capabilities. Someone should refactor this. Otherwise things are likely to break once we add more functionalities.
* Class [ResNet](resnet.py) only allows having completely no bias terms or bias terms everywhere. Though, would be nice to have bias terms in the fully-connected output layer but not in the convolutional layers.
* Class [ResNet](resnet.py) is still missing argument `param_meta_shapes`. If this is added, one could also replace the current padding of feature maps for residual connection with 1x1 conv layers, as well as adding dropout within residual blocks just like class [WRN](wide_resnet.py).
