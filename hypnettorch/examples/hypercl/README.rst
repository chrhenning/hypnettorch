.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.

Continual learning with hypernetworks
=====================================

.. content-inclusion-marker-do-not-remove

In continual learning (CL), a series of tasks (represented as datasets) :math:`\mathcal{D}_1, ..., \mathcal{D}_T` is learned sequentially, where only one dataset at a time is available and at the end of training performance on all tasks should be high.

An approach based on hypernets for tackling this problem was introduced by `von Oswald, Henning, Sacramento et al. <https://arxiv.org/abs/1906.00695>`__. The official implementation can be found `here <https://github.com/chrhenning/hypercl>`__. Goal of this example is it to demonstrate how ``hypnettorch`` can be used to implement such CL approach. Therefore, we provide a simple and light implementation that showcases many functionalities inherent to the package, **but do not focus on being able to reproduce the variety of experiments explored in the original paper**.

For the sake of simplicity, we only focus on the simplest CL scenario, `called <https://arxiv.org/abs/1904.07734>`__ task-incremental CL or ``CL1`` (note, that the original paper proposes three ways of tackling more complex CL scenarios, one of which has been further studied in `this paper <https://arxiv.org/abs/2103.01133>`__). Predictions according to a task :math:`t` are made by inputting the corresponding task embedding :math:`\mathbf{e}^{(t)}` into the hypernetwork in order to obtain the main network's weights :math:`\omega^{(t)} = h(\mathbf{e}^{(t)}, \theta)`, which in turn can be used for processing inputs via :math:`f(x, \omega^{(t)})`. Forgetting is prevented by adding a simple regularizer to the loss while learning task :math:`t`:

.. math::
    :label: eq-hnet-reg

    \frac{\beta}{t-1} \sum_{t<t'} \lVert h(\mathbf{e}^{(t')}, \theta) - h(\mathbf{e}^{(t',*)}, \theta^{(*)}) \rVert_2^2
    
where :math:`\beta` is a regularization constant, :math:`\mathbf{e}^{(t')}` are the task-embeddings, :math:`\theta` are the hypernets' parameters and parameters denoted by :math:`{}^{(*)}` are checkpointed from before starting to learn task :math:`t`. Simply speaking, the regularizer aims to prevent that the hypernetwork output :math:`h(\mathbf{e}^{(t')}, \theta)` for a previous task :math:`t'` changes compared to what was outputted before we started to learn task :math:`t`.

.. note::

    The original paper uses a lookahead in the regularizer which showed marginal performance improvements. Follow-up work (e.g., `here <https://github.com/mariacer/cl_in_rnns>`__ and `here <https://github.com/chrhenning/posterior_replay_cl>`__) discarded this lookahead for computational convenience. We ignore it as well!

Usage instructions
^^^^^^^^^^^^^^^^^^

The script :mod:`hypnettorch.examples.hypercl.run` showcases how a versatile simulation can be build with relatively little coding effort. You can explore the basic functionality of the script via

.. code-block:: console

   $ python run.py --help

.. note::

    The default arguments have **not** been hyperparameter-searched and may thus not reflect best possible performance.

By default, the script will run a **SplitMNIST** simulation (argument ``--cl_exp``)

.. code-block:: console

   $ python run.py

The default network (argument ``--net_type``) is a 2-hidden-layer MLP and the corresponding hypernetwork has been chosen to have roughly the same number of parameters (compression ratio is approx. 1).

Via the argument ``--hnet_reg_batch_size`` you can choose up to how many task should be used for the regularization in Eq. :eq:`eq-hnet-reg` (rather than always evaluating the sum over all previous tasks). This ensures that the computational budget of the regularization doesn't grow with the number of tasks. For instance, if at every iteration a **single** random (previous) task should be selected for regularization, just use

.. code-block:: console

   $ python run.py --hnet_reg_batch_size=1

You can also run other CL experiments, such as **PermutedMNIST** (e.g., via arguments ``--cl_exp=permmnist --num_classes_per_task=10 --num_tasks=10``) or **SplitCIFAR-10/100** (e.g., via arguments ``--cl_exp=splitcifar --num_classes_per_task=10 --num_tasks=6 --net_type=resnet``). Keep in mind, that with a change in dataset or main network, model sizes change and thus another hypernetwork should be chosen if a certain compression ratio should be accomplished.

Learning from the example
^^^^^^^^^^^^^^^^^^^^^^^^^

Goal of this example is it to get familiar with the capabilities of the package ``hypnettorch``. This can best be accomplished by reading through the source code, starting with the main function :func:`hypnettorch.examples.hypercl.run.run`.

#. The script makes use of module :mod:`hypnettorch.utils.cli_args` for defining command-line arguments. With a few lines of code, a large variety of arguments are created to, for instance, flexibly determine the architecture of the main- and hypernetwork.
#. Using those predefined arguments allows to quickly instantiate the corresponding networks by using functions of module :mod:`hypnettorch.utils.sim_utils`.
#. Continual learning datasets are generated with the help of specialized data handlers, e.g., :func:`hypnettorch.data.special.split_mnist.get_split_mnist_handlers`.
#. Hypernet regularization (Eq. :eq:`eq-hnet-reg`) is easily realized via the helper functions in module :mod:`hypnettorch.utils.hnet_regularizer`.

There are many other utilities that might be useful, but that are not incorporated in the example for the sake of simplicity. For instance:

* The module :mod:`hypnettorch.utils.torch_ckpts` can be used to easily save and load networks.
* The script can be emebedded into the hyperparameter-search framework of subpackage :ref:`hpsearch <hpsearch-reference-label>` to easily scan for hyperparameters that yield good performance.

More sophisticated examples can also be explored in the `PR-CL repository <https://github.com/chrhenning/posterior_replay_cl>`__ (note, the interface used in this repository is almost identical to ``hypnettorch``'s interface, except that the package wasn't called ``hypnettorch`` back then yet).
