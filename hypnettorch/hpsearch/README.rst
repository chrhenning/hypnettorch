.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

A general framework to perform hyperparameter searches on single- and multi-GPU systems
=======================================================================================

**Note, we currently only support simple grid searches.**
   
How to run a hyperparameter search
----------------------------------

The main script in this package is :mod:`hypnettorch.hpsearch.hpsearch`.

.. code-block:: console

   $ python -m hypnettorch.hpsearch.hpsearch --help

Though, before being able to run a hyperparameter search, the search grid has to be configured. Therefore, your simulation has its own implementation of the configuration file :mod:`hypnettorch.hpsearch.hpsearch_config_template`. Please refer to the corresponding documentation to obtain information on how to configure a hyperparameter search.

Execute on a single- or multi-GPU system without job scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way of execution is to run all hyperparameter configurations sequentially in the foreground. For instance, on a computer without GPUs, you could start the hpsearch on the CPU as follows

.. code-block:: console

  $ python -m hypnettorch.hpsearch.hpsearch --visible_gpus=-1

Though, assuming that your simulations automatically run on a visible GPU, you can also apply this sequential foreground execution to a GPU of your choice (e.g., GPU 2):

.. code-block:: console

  $ CUDA_VISIBLE_DEVICES=2 python -m hypnettorch.hpsearch.hpsearch --visible_gpus=-1

Alternatively, the hpsearch may assign GPU ressources to jobs. In this case, multiple hyperparameter configurations may run in parallel (on multiple GPUs as well as multiple runs per GPU). For this operation mode, you are required to install the package `GPUtil <https://github.com/anderskm/gputil>`__.

Please carefully study the arguments of the hpsearch.

.. code-block:: console

   $ python -m hypnettorch.hpsearch.hpsearch --help

Assume you may want to run your search on GPUs 0,1,2,7 and that there should be a hard limit of 5 jobs assigned to a GPU by the hpsearch (which you decide based on available CPU and RAM ressources). Note, option ``--max_num_jobs_per_gpu`` currently does not account for other processes that may be running on the GPU which are not assigned by this hpsearch. In addition, a run may only be assigned to a GPU if at maximum 75% of its memory is in use and its compute utilization is maximally at 60%. Since runs take some time to properly startup and allocate GPU ressources, you additionally specify argument ``--sim_startup_time``. Every time a job is assigned to a GPU, this time has to pass before a new job may be assigned (such that the first job had time to acquire GPU memory and compute ressources)

.. code-block:: console

   $ python -m hypnettorch.hpsearch.hpsearch --visible_gpus=0,1,2,7 --max_num_jobs_per_gpu=5 --allowed_memory=0.75 --allowed_load=0.6 --sim_startup_time=30

Execute on a cluster with IBM Platform LSF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may also run the hpsearch on a cluster that uses the IBM Platform LSF job scheduler. In this case, you have to install the package `bsub <https://pypi.org/project/bsub/>`__. To tell the hpsearch that should schedule jobs via ``bsub``, simply append the options ``--run_cluster --scheduler=lsf``. Here is an example call:

.. code-block:: console

    $ bsub -n 1 -W 120:00 -e hpsearch_mysim.err -o hpsearch_mysim.out -R "rusage[mem=8000]" python -m hypnettorch.hpsearch.hpsearch --grid_module=my_hpsearch_config --run_cluster --scheduler=lsf --num_jobs=50 --num_hours=24 --num_searches=1000 --resources="\"rusage[mem=8000, ngpus_excl_p=1]\""

In the example above, the hpsearch should run for 120 hours on the cluster, requiring 8GB of RAM during that time. Individual jobs will run for 24 hours. The hpsearch will maximally explore 1000 hyperparameter configurations. At most 50 jobs will be scheduled in parallel (new jobs will be scheduled as soon as old ones finished until the hard limit of 1000 runs is reached). Each job will require 1 GPU and 8GB of RAM.

Execute on a cluster with Slurm Workload Manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hpsearch can also be run on a cluster with the SLURM job scheduler via the arguments ``--run_cluster --scheduler=slurm``. Therefore, simply create a job script ``my_hpsearch.sh`` for the hpsearch as follows

.. code-block:: console

    #!/bin/bash
    #SBATCH --job-name=hpsearch
    #SBATCH --output=hpsearch_%j.out
    #SBATCH --error=hpsearch_%j.err
    #SBATCH --time=24:00:00
    #SBATCH --mem=8G
    python -m hypnettorch.hpsearch.hpsearch --grid_module=my_hpsearch_config --run_cluster --scheduler=slurm --slurm_mem=8G --slurm_gres=gpu:1 --num_jobs=25 --num_hours=4

The hpsearch can be executed via the command:

.. code-block:: console

    $ sbatch my_hpsearch.sh

Execute on a cluster with unsupported job scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unfortunately, you can only execute the hpsearch on a cluster with unsupported job scheduler in the sequential foreground mode via ``--visible_gpus=-1``. For instance, on a cluster running the SLURM job scheduler (note, SLURM is supported, see above) you can run the hpsearch in sequential forground mode via a script ``my_hpsearch.sh``:

.. code-block:: console

    #!/bin/bash
    #SBATCH --job-name=hpsearch
    #SBATCH --output=hpsearch_%j.out
    #SBATCH --error=hpsearch_%j.err
    #SBATCH --time=120:00:00
    #SBATCH --mem=8G
    #SBATCH --gres gpu:1
    python -m hypnettorch.hpsearch.hpsearch --grid_module=my_hpsearch_config --visible_gpus=-1

Note, in this case, you request the ressources required for your jobs for the hpsearch itself. Now, you could execute the hpsearch via

.. code-block:: console

    $ sbatch my_hpsearch.sh

Postprocessing
--------------

The post processing script :mod:`hypnettorch.hpsearch.hpsearch_postprocessing` is currently very rudimentary. Its most important task is to make sure that the results of all completed runs are listed in a CSV file (note, that the hpsearch might be killed prematurely while some jobs are still running).

Please checkout

.. code-block:: console

    $ python3 -m hypnettorch.hpsearch.hpsearch_postprocessing --help

How to use this framework with your simulation
----------------------------------------------

In order to utilize the scripts in this subpackage, you have to create a copy of the template :mod:`hypnettorch.hpsearch.hpsearch_config_template` and fill the template with content as described inside the module. For instance, see `probabilistic.prob_mnist.hpsearch_config_split_bbb <https://github.com/chrhenning/posterior_replay_cl/blob/master/probabilistic/prob_mnist/hpsearch_config_split_bbb.py>`__ as an example.

Additionally, you need to make sure that your simulation has a command-line option like ``--out_dir`` (that specifies the output directory) and that your simulation writes a performance summary file, that can be used to evaluate simulations.
