#!/usr/bin/env python3
# Copyright 2021 Maria Cervera
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
# @title          :hpsearch/gather_random_seeds.py
# @author         :mc, ch
# @contact        :mariacer@ethz.ch
# @created        :01/06/2021
# @version        :1.0
# @python_version :3.8.5
"""
Gather random seeds for a given experiment
------------------------------------------

This script can be used to gather random seeds for a given configuration. Thus,
it is intended to test the robustness of this certain configuration.

The configuration can either be provided directly, or the path to a simulation
output folder or hyperparameter search output folder is provided. A simulation
output folder is recognized by the file ``config.pickle`` which contains the
`configuration`, i.e., all command-line arguments (cf. function
:func:`hypnettorch.sim_utils.setup_environment`). If a hyperparameter search
output folder (cf. :mod:`hypnettorch.hpsearch.hpsearch`) is provided, the best
run will be selected.

**Example 1:** Assume you are in the simulation directory ``sims/my_sim`` and
want to start the random seed gathering from there for a simulation in folder
``sims/my_sim/out/example_run``. Note, we assume here that the base run in
``sims/my_sim/out/example_run`` finished successfully and can already be used
as 1 random seed.

.. code-block:: console

   $ TMP_CUR_DIR="$(pwd -P)" && pushd ../../hpsearch && python3 \
gather_random_seeds.py --grid_module=sims.my_sim.hpsearch_config \
--run_dir=../psims/my_sim/out/example_run --num_seeds=10 | tee /dev/tty | \
awk 'END{print}' | xargs bash -c 'echo --grid_module=$0 --grid_config=$1 \
--force_out_dir --dont_force_new_dir --out_dir=$2' | xargs python3 hpsearch.py \
--run_cwd=$TMP_CUR_DIR && popd

**Example 2:** Alternatively, the hpsearch can be started directly via the
option ``--start_gathering``.

.. code-block:: console

   $ TMP_CUR_DIR="$(pwd -P)" && pushd ../../hpsearch && python3 \
gather_random_seeds.py --grid_module=sims.my_sim.hpsearch_config \
--run_dir=../psims/my_sim/out/example_run --num_seeds=4 --start_gathering \
--config_name=example_run_seed_gathering --run_cwd=$TMP_CUR_DIR && popd

**Example 3:** An example instantiation of this script can be found in module
`probabilistic.regression.gather_seeds_bbb <https://git.io/J9quN>`__.
"""
import argparse
from argparse import Namespace
import importlib
import numpy as np
import os
import pickle
import shutil
from subprocess import call
import tempfile
from warnings import warn

from hypnettorch.hpsearch import hpsearch
import hypnettorch.utils.misc as misc

def get_single_run_config(out_dir):
    """Load the config file from a specified experiment.

    Args:
        out_dir (str): The path to the experiment.

    Returns:
        The Namespace object containing argument names and values.
    """
    print('Loading the configuration of run: %s' % out_dir)

    if not os.path.exists(os.path.join(out_dir, "config.pickle")):
        # Currently, we can't read the config from the results csv files.
        raise NotImplementedError('The run "%s" does not contain a ' % out_dir +
            '"config.pickle" file.')

    with open(os.path.join(out_dir, "config.pickle"), "rb") as f:
        config = pickle.load(f)

    return config

def get_best_hpsearch_config(out_dir):
    """Load the config file from the best run of a hyperparameter search.

    This file loads the results of the hyperparameter search, and select the
    configuration that lead to the best performance score.

    Args:
        out_dir (str): The path to the hpsearch result folder.

    Returns:
        (tuple): Tuple containing:

        - **config**: The config of the best run.
        - **best_out_dir**: The path to the best run.
    """
    run_dirs = os.listdir(out_dir)
    if 'TO_BE_DELETED' in run_dirs:
        # Postprocessing has marked some runs for deletion!
        run_dirs.remove('TO_BE_DELETED')
        run_dirs.extend(os.listdir(os.path.join(out_dir, 'TO_BE_DELETED')))

    curr_best_dir = None
    curr_best_score = None

    for i, sim_dir in enumerate(run_dirs):
        sim_path = os.path.join(run_dirs, sim_dir)
        if not os.path.isdir(sim_path):
            continue

        if not os.path.exists(os.path.join(sim_path,
                                           hpsearch._SUMMARY_FILENAME)):
            # No result summary in folder.
            continue

        try:
            performance_dict = hpsearch._SUMMARY_PARSER_HANDLE(sim_path, i)
        except:
            #warn('Cannot read results from simulation "%s"!' % sim_dir)
            continue

        has_finished = int(performance_dict['finished'][0])
        if not has_finished:
            #warn('Simulation "%s" did not finish!' % sim_dir)
            continue

        curr_score = float(performance_dict[hpsearch._PERFORMANCE_KEY][0])

        if curr_best_dir is None:
            curr_best_dir = sim_path
            curr_best_score = curr_score
        elif hpsearch._PERFORMANCE_SORT_ASC:
            if curr_score < curr_best_score:
                curr_best_dir = sim_path
                curr_best_score = curr_score
        else:
            if curr_score > curr_best_score:
                curr_best_dir = sim_path
                curr_best_score = curr_score

    if curr_best_dir is None:
        raise RuntimeError('Did not find any finished run!')

    return get_single_run_config(curr_best_dir), curr_best_dir

def build_grid_and_conditions(cmd_args, config, seeds_list):
    """Build the hpconfig for the random seed gathering.

    Args:
        cmd_args: CLI arguments of this script.
        config: The config to be translated into a search grid.
        seeds_list (list): The random seeds to be gathered.

    (tuple): Tuple containing:

        - **grid** (dict): The search grid.
        - **conditions** (list): Constraints for the search grid.
    """
    grid = {}

    for k, v in vars(config).items():
        if isinstance(v, str):
            v = v.strip('"')
            v = '"' + v + '"'
        grid[k] = [v]
    grid['random_seed'] = seeds_list

    conditions = []
    if cmd_args.vary_data_seed:
        for s in seeds_list:
            conditions.append(
                (
                    {'random_seed': [s]},
                    {'data_random_seed': [s]}
                )
            )

    return grid, conditions

def get_hpsearch_call(cmd_args, num_seeds, grid_config, hpsearch_dir=None):
    """Generate the command line for the hpsearch.

    Args:
        cmd_args: The command line arguments.
        num_seeds (int): Number of searches.
        grid_config (str): Location of search grid.
        hpsearch_dir (str, optional): Where the hpsearch should write its
            results to.

    Returns:
        (str): The command line to be executed.

    """
    cluster_cmd_prefix = ''
    cluster_cmd_suffix = ''
    non_cluster_cmd_suffix = ''
    if cmd_args.run_cluster and cmd_args.scheduler == 'lsf':
        cluster_cmd_prefix = 'bsub -n 1 -W %s:00 ' % cmd_args.hps_num_hours + \
            '-e random_seeds.err -o random_seeds.out -R "%s" ' % \
            cmd_args.hps_resources.strip('"')

        cluster_cmd_suffix = ' --run_cluster ' + \
            '--scheduler=%s ' % cmd_args.scheduler +\
            '--num_jobs=%s ' % cmd_args.num_jobs +\
            '--num_hours=%s ' % cmd_args.num_hours + \
            '--resources="\\"%s\\"" ' % cmd_args.resources.strip('"') + \
            '--num_searches=%d ' % num_seeds
    elif cmd_args.run_cluster:
        assert cmd_args.scheduler == 'slurm'
        cluster_cmd_suffix = ' --run_cluster ' + \
            '--scheduler=%s ' % cmd_args.scheduler + \
            '--num_jobs=%s ' % cmd_args.num_jobs + \
            '--num_hours=%s ' % cmd_args.num_hours + \
            '--slurm_mem=%s ' % cmd_args.slurm_mem + \
            '--slurm_gres=%s ' % cmd_args.slurm_gres + \
            '--slurm_partition=%s ' % cmd_args.slurm_partition + \
            '--slurm_qos=%s ' % cmd_args.slurm_qos + \
            '--slurm_constraint=%s ' % cmd_args.slurm_constraint + \
            '--num_searches=%d ' % num_seeds
    else:
        non_cluster_cmd_suffix = \
            '--visible_gpus=%s ' % cmd_args.visible_gpus + \
            '--allowed_load=%f ' % cmd_args.allowed_load + \
            '--allowed_memory=%f ' % cmd_args.allowed_memory + \
            '--sim_startup_time=%d ' % cmd_args.sim_startup_time + \
            '--max_num_jobs_per_gpu=%d ' % cmd_args.max_num_jobs_per_gpu
    #cmd_str = 'TMP_CUR_DIR="$(pwd -P)" && pushd ../../hpsearch && ' + \
    cmd_str = cluster_cmd_prefix + \
        'python3 hpsearch.py --grid_module=%s '% cmd_args.grid_module + \
        '--grid_config=%s ' % grid_config + \
        '--run_cwd=%s ' % cmd_args.run_cwd #'--run_cwd=$TMP_CUR_DIR '
    if cmd_args.deterministic_search:
        cmd_str += '--deterministic_search '
    if cmd_args.dont_generate_full_grid:
        cmd_str += '--dont_generate_full_grid '
    if hpsearch_dir is not None:
        cmd_str += '--out_dir=%s --force_out_dir '% hpsearch_dir + \
    '--dont_force_new_dir '
    cmd_str += cluster_cmd_suffix + non_cluster_cmd_suffix #+ ' && popd'

    return cmd_str

def write_seeds_summary(results_dir, summary_keys, summary_sem, summary_precs,
                        ret_seeds=False, summary_fn=None,
                        seeds_summary_fn='seeds_summary_text.txt'):
    """Write the MEAN and STD (resp. SEM) while aggregating all seeds to text
    file.

    Args:
        results_dir (str): The results directory.
        summary_keys (list): See argument ``summary_keys`` of function
            :func:`run`.
        summary_sem (bool): See argument ``summary_sem`` of function
            :func:`run`.
        summary_precs (list or int, optional): See argument ``summary_precs`` of
            function :func:`run`.
        summary_fn (str, optional): If given, this will determine
            the name of the summary file within individual runs.
        seeds_summmary_fn (str, optional): The name to give to the summary
            file across all seeds.
        ret_seeds (bool, optional): If activated, the random seeds of all
            considered runs are returned as a list.
    """
    random_seeds = []

    if summary_precs is None:
        summary_precs = 2
    if isinstance(summary_precs, int):
        summary_precs = [summary_precs] * len(summary_keys)
    else:
        assert len(summary_keys) == len(summary_precs)

    # Iterate over all result folders.
    score_dict = {}
    n_scores = 0
    for k in summary_keys:
        score_dict[k] = []

    seed_dirs = []
    seed_dir_prefix = {}

    for i, sim_dir in enumerate(os.listdir(results_dir)):
        sim_path = os.path.join(results_dir, sim_dir)
        if not os.path.isdir(sim_path):
            continue

        try:
            performance_dict = hpsearch._SUMMARY_PARSER_HANDLE(sim_path, i,
                summary_fn=summary_fn)
        except:
            warn('Cannot read results from simulation "%s"!' % sim_dir)
            continue

        has_finished = int(performance_dict['finished'][0])
        if not has_finished:
            warn('Simulation "%s" did not finish!' % sim_dir)
            continue

        n_scores += 1
        for k in summary_keys:
            score_dict[k].append(float(performance_dict[k][0]))

        if ret_seeds:
            sim_config = get_single_run_config(sim_path)
            random_seeds.append(sim_config.random_seed)

        seed_dirs.append(sim_path)
        if sim_dir.count('_') == 2:
            prefix = sim_dir[:sim_dir.rfind('_')]
            if prefix not in seed_dir_prefix.keys():
                seed_dir_prefix[prefix] = [sim_path, 1]
            else:
                seed_dir_prefix[prefix][1] += 1
        else:
            seed_dir_prefix[sim_dir] = [sim_path, 1]

    # In case the gathering has been started from an existing seed, we can try
    # to determine its path. Note, this might turn out difficult if only single
    # seeds have been gathered or the seed gathering has been started multiple
    # times.
    original_seed_path = None
    nunique = 0
    for k, v in seed_dir_prefix.items():
        if v[1] == 1:
            original_seed_path = v[0]
            nunique += 1
    if nunique > 1:
        original_seed_path = None


    if n_scores == 0:
        raise RuntimeError('No results found!')

    score_means = {}
    score_devs = {}

    # Get averages across seeds.
    for k in summary_keys:
        score_means[k] = np.mean(score_dict[k])
        score_devs[k] = np.std(score_dict[k])
        if summary_sem:
            score_devs[k] /= np.sqrt(n_scores)

    # Write into a summary text file.
    filename = os.path.join(results_dir, seeds_summary_fn)
    with open(filename, "w") as f:
        for i, k in enumerate(summary_keys):
            p = summary_precs[i]
            f.write(('%s (mean +/- %s): %.'+str(p)+'f +- %.'+str(p)+'f\n') % \
                    (k, 'sem' if summary_sem else 'std',
                     score_means[k], score_devs[k]))
        f.write('Number of seeds: %i \n\n' % n_scores)
        f.write('Publication tables style: \n')
        f.write('%s \n' % summary_keys)
        tab_str = ''
        for i, k in enumerate(summary_keys):
            if i > 0:
                tab_str += ' & '
            p = summary_precs[i]
            tab_str += ('%.'+str(p)+'f $\pm$  %.'+str(p)+'f ') \
                % (score_means[k], score_devs[k])
        f.write('%s \n\n' % tab_str)

    return random_seeds if ret_seeds else None

def run(grid_module=None, results_dir='./out/random_seeds', config=None,
        ignore_kwds=None, forced_params=None, summary_keys=None,
        summary_sem=False, summary_precs=None, hpmod_path=None):
    """Run the script.

    Args:
        grid_module (str, optional): Name of the reference module which contains
            the hyperparameter search config that can be modified to gather
            random seeds.
        results_dir (str, optional): The path where the hpsearch should store
            its results.
        config: The Namespace object containing argument names and values.
            If provided, all random seeds will be gathered from zero, with no
            reference run.
        ignore_kwds (list, optional): A list of keywords in the config file
            to exclude from the grid.
        forced_params (dict, optional): Dict of key-value pairs specifying
            hyperparameter values that should be fixed across runs.
        summary_keys (list, optional): If provided, those mean and std of those
            summary keys will be written by function
            :func:`write_seeds_summary`. Otherwise, the performance key defined
            in ``grid_module`` will be used.
        summary_sem (bool): Whether SEM or SD should be calculated in function
            :func:`write_seeds_summary`.
        summary_precs (list or int, optional): The precision with which the
            summary statistics according to ``summary_keys`` should be listed.
        hpmod_path (str, optional): If the hpsearch doesn't reside in the same
            directory as the calling script, then we need to know from where to
            start the hpsearch.
    """
    if ignore_kwds is None:
        ignore_kwds = []
    if forced_params is None:
        forced_params = {}

    ### Parse the command-line arguments.
    parser = argparse.ArgumentParser(description= \
        'Gathering random seeds for the specified experiment.')
    parser.add_argument('--seeds_dir', type=str, default='',
                        help='If provided, all other arguments (except ' +
                             '"grid_module") are ignored! ' +
                             'This is supposed to be the output folder of a ' +
                             'random seed gathering experiment. If provided, ' +
                             'the results (for different seeds) within this ' +
                             'directory are gathered and written to a human-' +
                             'readible text file.')
    parser.add_argument('--run_dir', type=str, default='',
                        help='The output directory of a simulation or a ' +
                             'hyperparameter search. '
                             'For single runs, the configuration will be ' +
                             'loaded and run with different seeds.' +
                             'For multiple runs, i.e. results of ' +
                             'hyperparameter searches, the configuration ' +
                             'leading to the best performance will be ' +
                             'selected and run with different seeds.')
    parser.add_argument('--config_name', type=str,
                        default='hpsearch_random_seeds',
                        help='A name for this call of gathering random ' +
                             'seeds. As multiple gatherings might be running ' +
                             'in parallel, it is important that this name is ' +
                             'unique name for each experiment. ' +
                             'Default: %(default)s.')
    parser.add_argument('--grid_module', type=str, default=grid_module,
                        help='See CLI argument "grid_module" of ' +
                             'hyperparameter search script "hpsearch". ' +
                             ('Default: %(default)s.' \
                              if grid_module is not None else ''))
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='The number of different random seeds.')
    parser.add_argument('--seeds_list', type=str, default='',
                        help='The list of seeds to use. If specified, ' +
                             '"num_seeds" will be ignored.')
    parser.add_argument('--vary_data_seed', action='store_true',
                        help='If activated, "data_random_seed"s are set ' +
                             'equal to "random_seed"s. Otherwise only ' +
                             '"random_seed"s are varied.')
    parser.add_argument('--start_gathering', action='store_true',
                        help='If activated, the actual gathering of random ' +
                             'seeds is started via the "hpsearch.py" script.')

    # Arguments only required if `start_gathering`.
    hpgroup = parser.add_argument_group('Hpsearch call options')
    hpgroup.add_argument('--hps_num_hours', type=int, metavar='N', default=24,
                         help='If "run_cluster" is activated, then this ' +
                              'option determines the maximum number of hours ' +
                              'the entire search may run on the cluster. ' +
                              'Default: %(default)s.')
    hpgroup.add_argument('--hps_resources', type=str,
                         default='"rusage[mem=8000]"',
                         help='If "run_cluster" is activated and "scheduler" ' +
                              'is "lsf", then this option determines the ' +
                              'resources assigned to the entire ' +
                              'hyperparameter search (option -R of bsub). ' +
                              'Default: %(default)s.')
    hpgroup.add_argument('--hps_slurm_mem', type=str, default='8G',
                         help='See option "slum_mem". This argument effects ' +
                              'hyperparameter search itself. '
                              'Default: %(default)s.')

    rsgroup = parser.add_argument_group('Random seed hpsearch options')
    hpsearch.hpsearch_cli_arguments(rsgroup, show_out_dir=False,
                                    show_grid_module=False)
    cmd_args = parser.parse_args()

    grid_module = cmd_args.grid_module
    if grid_module is None:
        raise ValueError('"grid_module" needs to be specified.')
    grid_module = importlib.import_module(grid_module)
    hpsearch._read_config(grid_module, require_perf_eval_handle=True)

    if summary_keys is None:
        summary_keys = [hpsearch._PERFORMANCE_KEY]

    ####################################################
    ### Aggregate results of random seed experiments ###
    ####################################################
    if len(cmd_args.seeds_dir):
        print('Writing seed summary ...')
        write_seeds_summary(cmd_args.seeds_dir, summary_keys, summary_sem,
                            summary_precs)
        exit(0)

    #######################################################
    ### Create hp config grid for random seed gathering ###
    #######################################################
    if len(cmd_args.seeds_list) > 0:
        seeds_list = misc.str_to_ints(cmd_args.seeds_list)
        cmd_args.num_seeds = len(seeds_list)
    else:
        seeds_list = list(range(cmd_args.num_seeds))

    if config is not None and cmd_args.run_dir != '':
        raise ValueError('"run_dir" may not be specified if configuration ' +
                         'is provided directly.')

    # The directory in which the hpsearch results should be written. Will only
    # be specified if the `config` is read from a finished simulation.
    hpsearch_dir = None
    # Get config if not provided.
    if config is None:
        if not os.path.exists(cmd_args.run_dir):
            raise_error = True
            # FIXME hacky solution.
            if cmd_args.run_cwd != '':
                tmp_dir = os.path.join(cmd_args.run_cwd, cmd_args.run_dir)
                if os.path.exists(tmp_dir):
                    cmd_args.run_dir = tmp_dir
                    raise_error = False
            if raise_error:
                raise ValueError('Directory "%s" does not exist!' % \
                                 cmd_args.run_dir)

        # FIXME A bit of a shady decision.
        single_run = False
        if os.path.exists(os.path.join(cmd_args.run_dir, 'config.pickle')):
            single_run = True

        # Get the configuration.
        if single_run:
            config = get_single_run_config(cmd_args.run_dir)
            run_dir = cmd_args.run_dir
        else:
            config, run_dir = get_best_hpsearch_config(cmd_args.run_dir)

        # We should already have one random seed.
        try:
            performance_dict = hpsearch._SUMMARY_PARSER_HANDLE(run_dir, -1)
            has_finished = int(performance_dict['finished'][0])
            if not has_finished:
                raise Exception()

            use_run = True

        except:
            use_run = False

        if use_run:
            # The following ensures that we can safely use `basename` later on.
            run_dir = os.path.normpath(run_dir)

            if not os.path.isabs(results_dir):
                if os.path.isdir(cmd_args.run_cwd):
                    results_dir = os.path.join(cmd_args.run_cwd, results_dir)
            results_dir = os.path.abspath(results_dir)
            hpsearch_dir = os.path.join(results_dir, os.path.basename(run_dir))

            if os.path.exists(hpsearch_dir):
                # TODO attempt to write summary and exclude existing seeds.
                warn('Folder "%s" already exists.' % hpsearch_dir)
                print('Attempting to aggregate random seed results ...')

                gathered_seeds = write_seeds_summary(hpsearch_dir, summary_keys,
                    summary_sem, summary_precs, ret_seeds=True)

                if len(gathered_seeds) >= len(seeds_list):
                    print('Already enough seeds have been gathered!')
                    exit(0)

                for gs in gathered_seeds:
                    if gs in seeds_list:
                        seeds_list.remove(gs)
                    else:
                        ignored_seed = seeds_list.pop()
                        if len(cmd_args.seeds_list) > 0:
                            print('Seed %d is ignored as seed %d already ' \
                                  % (ignored_seed, gs) + 'exists.')

            else:
                os.makedirs(hpsearch_dir)
                # We utilize the already existing random seed.
                shutil.copytree(run_dir,
                    os.path.join(hpsearch_dir, os.path.basename(run_dir)))
                if config.random_seed in seeds_list:
                    seeds_list.remove(config.random_seed)
                else:
                    ignored_seed = seeds_list.pop()
                    if len(cmd_args.seeds_list) > 0:
                        print('Seed %d is ignored as seed %d already exists.' \
                              % (ignored_seed, config.random_seed))

    print('%d random seeds will be gathered!' % len(seeds_list))

    ### Which attributes of the `config` should be ignored?
    # We never set the ouput directory.
    if hpsearch._OUT_ARG not in ignore_kwds:
        ignore_kwds.append(hpsearch._OUT_ARG)

    for kwd in ignore_kwds:
        delattr(config, kwd)

    ### Replace config values provided via `forced_params`.
    if len(forced_params.keys()) > 0:
        for kwd, value in forced_params.items():
            setattr(config, kwd, value)

    ### Get a filename for where to store the search grid.
    config_dn, config_bn = os.path.split(cmd_args.config_name)
    if len(config_dn) == 0: # No relative path given, store only temporary.
        config_dn = tempfile.gettempdir()
    else:
        config_dn = os.path.abspath(config_dn)
    config_fn_prefix = os.path.splitext(config_bn)[0]
    config_name = os.path.join(config_dn, config_fn_prefix + '.pickle')
    if os.path.exists(config_name):
        if len(config_dn) > 0:
            overwrite = input('The config file "%s" ' % config_name + \
                'already exists! Do you want to overwrite the file? [y/n] ')
            if not overwrite in ['yes','y','Y']:
                exit(1)
        else: # Get random temporary filename.
            config_name_temp = tempfile.NamedTemporaryFile( \
                prefix=config_fn_prefix, suffix=".pickle")
            print('Search grid "%s" already exists, using name "%s" instead!' \
                  % (config_name, config_name_temp.name))
            config_name = config_name_temp.name
            config_name_temp.close()

    ### Build and store hpconfig for random seed gathering!
    grid, conditions = build_grid_and_conditions(cmd_args, config, seeds_list)

    rseed_config = {
        'grid': grid,
        'conditions': conditions
    }
    with open(config_name, 'wb') as f:
        pickle.dump(rseed_config, f)

    ### Gather random seeds.
    if cmd_args.start_gathering:
        
        cmd_str = get_hpsearch_call(cmd_args, len(seeds_list), config_name,
                                    hpsearch_dir=hpsearch_dir)
        print(cmd_str)

        ### Start hpsearch.
        if hpmod_path is not None:
            backup_curr_path = os.getcwd()
            os.chdir(hpmod_path)
        if cmd_args.run_cluster and cmd_args.scheduler == 'slurm':
            # FIXME hacky solution to write SLURM job script.
            # FIXME might be wrong to give the same `slurm_qos` to the hpsearch,
            # as the job might have to run much longer.
            job_script_fn = hpsearch._write_slurm_script(Namespace(**{
                    'num_hours': cmd_args.hps_num_hours,
                    'slurm_mem': cmd_args.hps_slurm_mem,
                    'slurm_gres': 'gpu:0',
                    'slurm_partition': cmd_args.slurm_partition,
                    'slurm_qos': cmd_args.slurm_qos,
                    'slurm_constraint': cmd_args.slurm_constraint,
                }),
                cmd_str, 'random_seeds')

            cmd_str = 'sbatch %s' % job_script_fn
            print('We will execute command "%s".' % cmd_str)

        # Execute the program.
        print('Starting gathering random seeds...')
        ret = call(cmd_str, shell=True,  executable='/bin/bash')
        print('Call finished with return code %d.' % ret)
        if hpmod_path is not None:
            os.chdir(backup_curr_path)

        # If we run the hpsearch on the cluster, then we just submitted a job
        # and the search didn't actually run yet.
        if not cmd_args.run_cluster and hpsearch_dir is not None:
            write_seeds_summary(hpsearch_dir, summary_keys, summary_sem,
                                summary_precs)

        print('Random seed gathering finished successfully!')
        exit(0)

    ### Random seeds not gathered yet - finalize program.
    print(hpsearch_dir is None)
    if hpsearch_dir is not None:
        print('IMPORTANT: At least one random seed has already been ' + \
              'gathered! Please ensure that the hpsearch forces the correct ' +
              'output path.')

    print('Below is a possible hpsearch call:')
    call_appendix = ''
    if hpsearch_dir is not None:
        call_appendix = '--force_out_dir --dont_force_new_dir ' + \
            '--out_dir=%s' % hpsearch_dir
    print()
    print('python3 hpsearch.py --grid_module=%s --grid_config=%s %s' % \
          (cmd_args.grid_module, config_name, call_appendix))
    print()

    # We print the individual paths to allow easy parsing via `awk` and `xargs`.
    if hpsearch_dir is None:
        print('Below is the "grid_module" name and the path to the ' +
              '"grid_config".')
        print(cmd_args.grid_module, config_name)
    else:
        print('Below is the "grid_module" name, the path to the ' +
              '"grid_config" and the output path that should be used for the ' +
              'hpsearch.')
        print(cmd_args.grid_module, config_name, hpsearch_dir)

if __name__ == '__main__':
    run()


