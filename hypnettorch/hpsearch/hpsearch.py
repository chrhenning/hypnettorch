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
# @title          :hpsearch/hpsearch.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :05/05/2019
# @version        :1.0
# @python_version :3.6.8
"""
Hyperparameter Search Script
----------------------------

A very simple hyperparameter search. The results will be gathered as a CSV file.

Here is an example on how to start an hyperparameter search on a cluster using
:code:`bsub`:

.. code-block:: console

   $ bsub -n 1 -W 48:00 -e hpsearch.err -o hpsearch.out \\
     -R "rusage[mem=8000]" \\
     python -m hypnettorch.hpsearch.hpsearch --run_cluster --num_jobs=20

For more demanding jobs (e.g., ImageNet), one may request more resources:

.. code-block:: console

   $ bsub -n 1 -W 96:00 -e hpsearch.err -o hpsearch.out \\
     -R "rusage[mem=16000]" \\
     python -m hypnettorch.hpsearch.hpsearch --run_cluster --num_jobs=20 \\
     --num_hours=48 --resources="\\"rusage[mem=8000, ngpus_excl_p=1]\\""

Please fill in the grid parameters in the corresponding config file (see
command line argument `grid_module`).
"""
import argparse
from collections import defaultdict
from datetime import datetime
import getpass
import glob
import importlib
import json
import numpy as np
import os
import pandas
import pickle
from queue import Queue, Empty
import random
import re
import shutil
import subprocess
import sys
import time
from threading  import Thread
import traceback
import warnings

try:
    import GPUtil
except ModuleNotFoundError:
    GPUtil = None
    warnings.warn('Package "GPUtil" could not be imported, but might be ' +
                  'needed for some functionalities of this script.')

try:
    from bsub import bsub
except ModuleNotFoundError:
    bsub = None
    warnings.warn('Package "bsub" could not be imported, but might be ' +
                  'needed for some functionalities of this script.')

from hypnettorch.utils import misc

# From which module to read the default grid.
_DEFAULT_GRID = 'classifier.imagenet.hpsearch_config_ilsvrc_cub'

### The following variables will be otherwritten in the main ###
################################################################
### See method `_read_config`.
# Name of the script that should be executed by the hyperparameter search.
# Note, the working directory is set seperately by the hyperparameter search
# script.
_SCRIPT_NAME = None # Has to be specified in helper module!
# This file is expected to reside in the output folder of the simulation.
_SUMMARY_FILENAME = None # Has to be specified in helper module!
# These are the keywords that are supposed to be in the summary file.
# A summary file always has to include the keyword "finished"!.
_SUMMARY_KEYWORDS = None # Has to be specified in helper module!
# The name of the command-line argument that determines the output folder
# of the simulation.
_OUT_ARG = 'out_dir' # Default value if attribute `_OUT_ARG` does not exist.
# Function handle to parser of performance summary file.
_SUMMARY_PARSER_HANDLE = None # Default parser `_get_performance_summary` used.
# A function handle, that is used to evaluate whether an output folder should
# be kept.
_PERFORMANCE_EVAL_HANDLE = None # Has to be set in config file.
# According to which keyword will the CSV be sorted.
_PERFORMANCE_KEY = None # First key in `_SUMMARY_KEYWORDS` will be used.
# Sort order.
_PERFORMANCE_SORT_ASC = False
# FIXME should be deleted soon.
_ARGPARSE_HANDLE = None
################################################################

# This will be a list of booleans, each representing whether a specific cmd has
# been executed.
_CMD_FINISHED = None

def _grid_to_commands(grid_dict):
    """Translate a dictionary of parameter values into a list of commands.

    The entire set of possible combinations is generated.

    Args:
        grid_dict: A dictionary of argument names to lists, where each list
            contains possible values for this argument.

    Returns:
        A list of dictionaries. Each key is an argument name that maps onto a
        single value.
    """
    # We build a list of dictionaries with key value pairs.
    commands = []

    # We need track of the index within each value array.
    gkeys = list(grid_dict.keys())
    indices = [0] * len(gkeys)

    stopping_criteria = False
    while not stopping_criteria:

        cmd = dict()
        for i, k in enumerate(gkeys):
            v = grid_dict[k][indices[i]]
            cmd[k] = v
        commands.append(cmd)
        
        for i in range(len(indices)-1,-1,-1):
            indices[i] = (indices[i] + 1) % len(grid_dict[gkeys[i]])
            if indices[i] == 0 and i == 0:
                stopping_criteria = True
            elif indices[i] != 0:
                break

    return commands

def _grid_to_commands_random_pick(grid_dict, num_commands):
    """Translate a dictionary of parameter values into a list of commands.

    The desired number of commands is generated by randomly picking the values
    of the arguments. Note that since conditions are applied afterwards, we
    generate here some extra commands, such that some can be removed because
    of the conditions. However, it is still possible that if many commands are
    rejected because of the conditions, the number of commands doesn't reach
    the number of searches specified by the user.

    Args:
        grid_dict: A dictionary of argument names to lists, where each list
            contains possible values for this argument.
        num_commands (int): The number of commands to be generated.

    Returns:
        A list of dictionaries. Each key is an argument name that maps onto a
        single value.
    """
    # We build a list of dictionaries with key value pairs.
    commands = []

    # FIXME
    # Because the conditions are only enforced afterwards, we generate some
    # extra commands, assuming less than 300 will be removed because of the
    # conditions.
    num_commands += 300

    gkeys = list(grid_dict.keys())
    for c in range(num_commands):

        cmd = dict()
        for k in gkeys:
            # Randomly select a value for the current key.
            index = np.random.randint(len(grid_dict[k]))
            v = grid_dict[k][index]
            cmd[k] = v
        commands.append(cmd)

    return commands

def _args_to_cmd_str(cmd_dict, out_dir=None):
    """Translate a dictionary of argument names to values into a string that
    can be typed into a console.

    Args:
        cmd_dict: Dictionary with argument names as keys, that map to a value.
        out_dir (optional): The output directory that should be passed to the
            command. No output directory will be passed if not specified.

    Returns:
        A string of the form:
            python3 train.py --out_dir=OUT_DIR --ARG1=VAL1 ...
    """
    cmd_str = 'python3 %s' % _SCRIPT_NAME

    if out_dir is not None:
        cmd_str += ' --%s=%s' % (_OUT_ARG, out_dir)

    for k, v in cmd_dict.items():
        if type(v) == bool:
            cmd_str += ' --%s' % k if v else ''
        else:
            cmd_str += ' --%s=%s' % (k, str(v))

    return cmd_str

def _get_performance_summary(out_dir, cmd_ident, summary_fn=None):
    """Parse the performance summary file of a simulation.

    This is a very primitive parser, that expects that each line of the
    result file :code:`os.path.join(out_dir, _SUMMARY_FILENAME)` is a
    keyword-value pair. The keyword is taken from the :code:`_SUMMARY_KEYWORDS`
    list. **They must appear in the correct order.**
    The value can either be a single number or a list of numbers. A list of
    numbers will be converted into a string, such that it appears in a single
    cell under the given keyword when opening the result CSV file with a
    spreadsheet.

    Args:
        out_dir: The output directory of the simulation.
        cmd_ident (int): Identifier of this command (needed for informative
            error messages).
        summary_fn (str, optional): The summary filename.

    Raises:
        IOError: If performance summary file does not exist.
        ValueError: If a summary key is not at the expected position in the
            result file.

    Returns:
        A dictionary containing strings as keywords. Note, the values may not be
        lists, and strings need to be wrapped into an extra layer of double
        quotes such that the spreadsheet interprets them as a single entity.
    """
    # Get training results.
    if summary_fn is None:
        summary_fn =_SUMMARY_FILENAME
    result_summary_fn = os.path.join(out_dir, summary_fn)
    if not os.path.exists(result_summary_fn):
        raise IOError('Training run %d did not finish. No results!' \
                      % (cmd_ident+1))

    with open(result_summary_fn, 'r') as f:
        result_summary = f.readlines()

    # Ensure downwards compatibility!
    summary_keys = _SUMMARY_KEYWORDS

    performance_dict = dict()
    for line, key in zip(result_summary, summary_keys):
        if not line.startswith(key):
            raise ValueError('Key %s does not appear in performance '
                             % (key) + 'summary where it is expected.')
        # Parse the lines of the result file.
        # Crop keyword to retrieve only the value.
        _, line = line.split(' ', maxsplit=1)
        # https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
        line_nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(line_nums) == 1: # Single number
            performance_dict[key] = [line_nums[0]]
        else: # List of numbers
            # Convert list to a string for the resulting CSV file. Note, the
            # quotes are needed that the list will be written into a single cell
            # when opening the csv file (note, every key can have exactly one
            # value).
            performance_dict[key] = \
                ['"' + misc.list_to_str(line_nums, delim=',') + '"']

    return performance_dict

def _write_slurm_script(args, cmd_str, cmd_folder_name):
    """Write a slurm job script for the given command string.

    The bash script will be dumped in the current folder.

    Args:
        args (argparse.Namespace): Command-line arguments.
        cmd_str (str): The actual command that should be executed by the job
            job scheduler (independent of slurm).
        cmd_folder_name (str): The folder name of the command ``cmd_str`` within
            the hpsearch output folder. This is used to determine a filename.

    Returns:
        (str):
            Bash script filename.
    """
    script_fn = '%s_script.sh' % cmd_folder_name
    with open(script_fn, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --job-name %s\n' % cmd_folder_name)
        f.write('#SBATCH --output %s_' % cmd_folder_name + '%j.out\n')
        f.write('#SBATCH --error %s_' % cmd_folder_name + '%j.err\n')
        f.write('#SBATCH --time %d:00:00\n' % args.num_hours)
        if len(args.slurm_mem) > 0:
            f.write('#SBATCH --mem %s\n' % args.slurm_mem)
        if len(args.slurm_gres) > 0:
            f.write('#SBATCH --gres %s\n' % args.slurm_gres)
        if len(args.slurm_partition) > 0:
            f.write('#SBATCH --partition %s\n' % args.slurm_partition)
        if len(args.slurm_qos) > 0:
            f.write('#SBATCH --qos %s\n' % args.slurm_qos)
        if len(args.slurm_constraint) > 0:
            f.write('#SBATCH --constraint %s\n' % args.slurm_constraint)
        f.write(cmd_str)

    return script_fn

def _slurm_check_running(job_ids):
    """Check whether jobs are still in the job queue (either pending or
    running).

    Args:
        job_ids (list): List of job IDs.

    Returns:
        (list): List of bool values, denoting whether the corresponding job in
        ``job_ids`` is still listed via ``squeue``. Returns ``None`` if the
        jobs status couldn't be checked.
    """
    # FIXME hacky way of getting the username. Are we sure, that the Slurm
    # username always is the same as the linux username?
    p = subprocess.Popen('squeue -u %s' % getpass.getuser(), shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p_out, p_err = p.communicate()

    if len(p_out) == 0:
        warnings.warn('Couldn\'t check whether jobs are running. "squeue" ' +
                      'returned error: < %s >.' % (p_err.decode('utf-8')))
        return None

    is_running = [False] * len(job_ids)
    try:
        qjobs = p_out.decode('utf-8').split('\n')
        assert len(qjobs) > 0
        # First line is table header.
        qjobs = qjobs[1:]
        for qjob in qjobs:
            qentries = qjob.split()
            if len(qentries) > 0:
                # FIXME Don't know why some job IDs have underscores (job
                # arrays?).
                qid = int(qentries[0].strip('_'))
                if qid in job_ids:
                    is_running[job_ids.index(qid)] = True
    except:
        traceback.print_exc(file=sys.stdout)
        warnings.warn('Could not assess whether jobs are still in the job ' +
                      'queue. Command "squeue" ended with error message: %s' \
                      % (p_err.decode('utf-8')))
        return None

    return is_running

def _get_available_gpus(args):
    """Get IDs of available GPUs.

    Args:
        (....): See function :func:`_run_cmds`.

    Returns:
        (list): List of GPU IDs. Contained are only IDs of visible GPUs that
        have enough resources and where we didn't start a job recently
        (according to the user defined warmup time).

        Returns ``None`` if no GPUs are visible to the program.
    """
    if args.visible_gpus == '-1':
        return None

    if '_VISIBLE_GPUS' in globals():
        visible_gpus = globals()['_VISIBLE_GPUS']
    else:
        visible_gpus = misc.str_to_ints(args.visible_gpus)
        if len(visible_gpus) == 0: # Use all GPUs
            visible_gpus = [gpu.id for gpu in GPUtil.getGPUs()]
        globals()['_VISIBLE_GPUS'] = visible_gpus

        print('The following GPUs are used:')
        for gpu in GPUtil.getGPUs():
            if gpu.id in visible_gpus:
                print('GPU %d: %s' % (gpu.id, gpu.name))

    # When was the last job scheduled on each GPU. Note, we are only allowed to
    # schedule new jobs after a certain warmup period has passed.
    if '_LAST_JOBS_TS' not in globals():
        globals()['_LAST_JOBS_TS'] = [None] * len(visible_gpus)

    if len(visible_gpus) == 0:
        return None

    visible_gpus_tmp = visible_gpus
    visible_gpus = []
    for i, last_ts in enumerate(globals()['_LAST_JOBS_TS']):
        if last_ts is None or (time.time()-last_ts) > args.sim_startup_time:
            visible_gpus.append(visible_gpus_tmp[i])

    gpus_to_exclude = []
    for gpu in GPUtil.getGPUs():
        if gpu.id not in visible_gpus:
            gpus_to_exclude.append(gpu.id)

    # FIXME We should ensure that the ressources are free over a sustained
    # period of time, as there are sometime sudden drops in GPU usage based on
    # the instantaneous computation done by the scripts running on them.
    return GPUtil.getAvailable(order='random', limit=len(visible_gpus),
        maxLoad=args.allowed_load, maxMemory=args.allowed_memory,
        excludeID=gpus_to_exclude)

def _check_running(args, out_dir, results_file, jobs):
    """Check whether jobs are still running.

    This function checks whether the jobs in ``jobs`` are still running. If they
    have finished, they will be removed from the list ``jobs`` and the results
    of the jobs will be collected and saved.

    Args:
        (....): See function :func:`_run_cmds`.
        jobs (list): List of tuples, containing all running jobs.

    Returns:
        (list): Updated list ``jobs``, where finished jobs have been deleted.
    """
    def _clear_pipe(out_file, out_queue):
        try:
            with open(out_file,'a') as f:
                while True:
                    line = out_queue.get_nowait()
                    f.write(line)
        except Empty:
            pass # Pipes are cleared again, no more lines in there

    if args.run_cluster and args.scheduler == 'lsf':
        try:
            rjobs = bsub.running_jobs()
        except:
            traceback.print_exc(file=sys.stdout)
            warnings.warn('Could not assess whether jobs are still in the ' +
                          'job queue. Assuming all jobs are still running.')
            rjobs = None
    elif args.run_cluster:
        assert args.scheduler == 'slurm'
        rjobs = _slurm_check_running([t[0] for t in jobs])

    tmp_jobs = jobs
    jobs = []
    for ii, job_tup in enumerate(tmp_jobs):
        job, cmd_dict, folder_name, ind, gpu_id, job_io = job_tup

        cmd_out_dir = cmd_dict[_OUT_ARG]

        if args.run_cluster and args.scheduler == 'lsf':
            if rjobs is None or job.job_id in rjobs:
                jobs.append((job, cmd_dict, folder_name, ind, gpu_id, job_io))
                continue
        elif args.run_cluster:
            assert args.scheduler == 'slurm'
            # FIXME If we couldn't check the queue, then we assume all jobs are
            # still running. Otherwise we might run in danger that we just
            # consider all jobs as finished while always scheduling more.
            if rjobs is None or rjobs[ii]:
                jobs.append((job, cmd_dict, folder_name, ind, gpu_id, job_io))
                continue
        else:
            job_name = 'job_%s' % folder_name
            job_out_file = os.path.join(cmd_out_dir, job_name + '.out')
            job_err_file = os.path.join(cmd_out_dir, job_name + '.err')

            if job.poll() == None:
                # Clear pipes, such that they don't fill up.
                if os.path.exists(cmd_out_dir) and job_io is not None:
                    _, q_out, _, q_err = job_io
                    _clear_pipe(job_out_file, q_out)
                    _clear_pipe(job_err_file, q_err)

                jobs.append((job, cmd_dict, folder_name, ind, gpu_id, job_io))
                continue

        print('Job %d finished.' % ind)

        try:
            # If the output folder doesn't exist yet, we still create it to
            # write the log files. An example scenario could be, that the user
            # provided invalid CLI arguments. He can figure this out, if we
            # write the error log into the corresponding result folder.
            # FIXME Could be, that just our check whether a job still exists
            # failed. Note, that simulations might not start if the output
            # folder already exists.
            if not os.path.exists(cmd_out_dir):
                # FIXME I deactivated the creation, as it causes sometimes
                # trouble.
                #warnings.warn('Output directory of run %d does not exist ' \
                #    % (ind+1) + 'and will be created to save log files.')
                #os.makedirs(cmd_out_dir)
                warnings.warn('Output directory of run %d does not exist.' % \
                              (ind+1))

            # We store the command used for execution. This might be helpful
            # for the user in case he wants to manually continue the
            # simulation.
            with open(os.path.join(cmd_out_dir, 'hpsearch_command.sh'),
                      'w') as f:
                f.write('#!/bin/sh\n')
                f.write('%s' % (_args_to_cmd_str(cmd_dict)))

            ### Save logs from run.
            if args.run_cluster and args.scheduler == 'lsf':
                # Move the output files written by LSF on the cluster in the
                # simulation output folder.
                job_out_file = glob.glob('job_%s*.out' % folder_name)
                job_err_file = glob.glob('job_%s*.err' % folder_name)
                assert len(job_out_file + job_err_file) <= 2
                for job_f in job_out_file + job_err_file:
                    os.rename(job_f, os.path.join(cmd_out_dir, job_f))
            elif args.run_cluster:
                assert args.scheduler == 'slurm'
                job_out_file = '%s_%d.out' % (folder_name, job)
                job_err_file = '%s_%d.err' % (folder_name, job)
                job_script_fn = '%s_script.sh' % (folder_name)

                for job_f in [job_out_file, job_err_file, job_script_fn]:
                    if os.path.exists(job_f):
                        os.rename(job_f, os.path.join(cmd_out_dir, job_f))
                    else:
                        warnings.warn('Could not find file %s.' % job_f)
            elif job_io is not None:
                _, q_out, _, q_err = job_io
                _clear_pipe(job_out_file, q_out)
                _clear_pipe(job_err_file, q_err)

            ### Save results.
            # Get training results.
            performance_dict = _SUMMARY_PARSER_HANDLE(cmd_out_dir, ind)
            for k, v in performance_dict.items():
                cmd_dict[k] = v

            # Create or update the CSV file summarizing all runs.
            panda_frame = pandas.DataFrame.from_dict(cmd_dict)
            if os.path.isfile(results_file):
                old_frame = pandas.read_csv(results_file, sep=';')
                panda_frame = pandas.concat([old_frame, panda_frame],
                                            sort=True)
            panda_frame.to_csv(results_file, sep=';', index=False)

            # Check whether simulation has finished successfully.
            has_finished = int(float(cmd_dict['finished'][0]))
            if has_finished == 1:
                _CMD_FINISHED[ind] = True
            else:
                _CMD_FINISHED[ind] = False

        except Exception:
            traceback.print_exc(file=sys.stdout)
            warnings.warn('Could not assess whether run %d has been ' \
                          % (ind+1) + 'completed.')

    return jobs

def _run_cmds(args, commands, out_dir, results_file):
    """Run all commands associated with the hpsearch.

    Depending on the CLI argument ``--run_cluster``, this function will either
    submit a certain number of jobs to an LSF cluster and wait for these jobs to
    complete before starting new jobs or it will send jobs to multiple visible
    GPUs (potentially multiple jobs per GPU).

    Args:
        args (argparse.Namespace): Command-line arguments.
        commands (list): List of command dictionaries.
        out_dir (str): Output directory.
        results_file (str): Path to CSV file to store hpsearch results.
    """
    num_cmds = len(commands)

    jobs = []
    i = -1
    while len(commands) > 0:
        ### Stall until resources are available ###
        jobs = _check_running(args, out_dir, results_file, jobs)
        if args.run_cluster:
            # On the cluster, we just need to check whether we can schedule
            # more jobs. The batch system is taking care of checking whether
            # ressources are available.
            while len(jobs) >= args.num_jobs:
                time.sleep(10)
                jobs = _check_running(args, out_dir, results_file, jobs)

        else:
            gpu_to_use = None
            while gpu_to_use is None:
                # On a machine without job scheduler, we have to figure out
                # which GPUs are available.
                available_gpus = _get_available_gpus(args)
                while available_gpus is not None and len(available_gpus) == 0:
                    time.sleep(10)
                    available_gpus = _get_available_gpus(args)

                if available_gpus is None:
                    warnings.warn('No GPUs visible to the hpsearch!')
                    gpu_to_use = -1
                    break

                # Check that there are not already too many jobs on the GPU.
                jobs = _check_running(args, out_dir, results_file, jobs)
                jobs_per_gpu = defaultdict(int)
                for _, _, _, _, job_gpu_id, _ in jobs:
                    jobs_per_gpu[job_gpu_id] += 1
                for agpu in available_gpus:
                    if jobs_per_gpu[agpu] < args.max_num_jobs_per_gpu:
                        gpu_to_use = agpu
                        break

                if gpu_to_use is None:
                    time.sleep(10)

        cmd_dict = commands.pop(0)
        i += 1

        cmd_out_dir = cmd_dict[_OUT_ARG]
        folder_name = os.path.basename(cmd_out_dir)

        cmd_str = _args_to_cmd_str(cmd_dict)

        # Execute the program.
        print('Starting training run %d/%d -- "%s"' % (i+1, num_cmds, cmd_str))

        job_name = 'job_%s' % folder_name

        if args.run_cluster and args.scheduler == 'lsf': # Schedule job.
            # FIXME the bsub module ignores the pathnames we set. Hence, all
            # output files are simply stored in the local directory. For now, we
            # will capture this when the run completed and move the file.
            job_error_file = os.path.join(cmd_out_dir, job_name + '.err')
            job_out_file = os.path.join(cmd_out_dir, job_name + '.out')
            sub = bsub(job_name, R=args.resources, n=1,
                       W='%d:00' % args.num_hours, e=job_error_file,
                       o=job_out_file, verbose=True)
            sub(cmd_str)
            jobs.append((sub, cmd_dict, folder_name, i, None, None))
        elif args.run_cluster:
            assert args.scheduler == 'slurm'
            script_name = _write_slurm_script(args, cmd_str, folder_name)
            p = subprocess.Popen('sbatch --parsable %s' % script_name,
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p_out, p_err = p.communicate()

            if p.returncode != 0:
                warnings.warn('Job submission ended with return code %d - %s.' \
                              % (p.returncode, p_err.decode('utf-8')))
            try:
                job_id = int(p_out.decode('utf-8').strip())
                jobs.append((job_id, cmd_dict, folder_name, i, None, None))
            except:
                traceback.print_exc(file=sys.stdout)
                warnings.warn('Could not assess whether run %d has been ' \
                              % (i+1) + 'submitted.')

        elif gpu_to_use == -1: # Start on CPU.
            # FIXME stdout and stderr is not logged and therefore can't be
            # written to file.
            p_cmd = subprocess.Popen(cmd_str, shell=True)
            # Wait for job to finish.
            p_cmd.communicate()
            jobs.append((p_cmd, cmd_dict, folder_name, i, None, None))

        else: # Start job on local GPU.
            print('Job will be scheduled on GPU %d.' % gpu_to_use)
            p_cmd = subprocess.Popen( \
                'CUDA_VISIBLE_DEVICES=%d ' % (gpu_to_use) + cmd_str, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Register start time of job, such that we can respect the warmup
            # time before sending the next job onto the GPU.
            gpu_ind = globals()['_VISIBLE_GPUS'].index(gpu_to_use)
            globals()['_LAST_JOBS_TS'][gpu_ind] = time.time()

            # We have to clear the stdout and stderr pipes regularly, as
            # they have limited size in linux and will stall the process
            # once they are full.
            q_out = Queue()
            q_err = Queue()
            t_out = Thread(target=_enqueue_pipe, args=(p_cmd.stdout, q_out))
            t_err = Thread(target=_enqueue_pipe, args=(p_cmd.stderr, q_err))
            # The threads should get killed ones a job ends. But if the hpsearch
            # ends, they also should get killed.
            t_out.daemon = True
            t_err.daemon = True
            t_out.start()
            t_err.start()

            job_io = (t_out, q_out, t_err, q_err)
            jobs.append((p_cmd, cmd_dict, folder_name, i, gpu_to_use, job_io))

    # Wait for all jobs to complete.
    while len(jobs) > 0:
        time.sleep(10)
        jobs = _check_running(args, out_dir, results_file, jobs)

def _enqueue_pipe(pipe, queue):
    # The code from this function and our solution for logging is inpired by the
    # following thread (accessed: 05/12/2020):
    #   https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python
    #
    # NOTE Our copyright and license does not apply for this function.
    # We use this code WITHOUT ANY WARRANTIES.
    #
    # Instead, the code in this method is licensed under CC BY-SA 3.0:
    #    https://creativecommons.org/licenses/by-sa/3.0/
    #
    # The code stems from an answer by user "jfs":
    #    https://stackoverflow.com/users/4279/jfs
    # and was edited by user "ankostis"
    #    https://stackoverflow.com/users/548792/ankostis
    for l in iter(pipe.readline, b''):
        queue.put(l.decode('utf-8'))
    pipe.close()

def _backup_commands(commands, out_dir):
    """Backup commands.

    This function will generate a bash script that resembles the order in
    which the individual commands have been executed. This is important, as the
    order might be random. This script is just another helper for the user to
    follow the execution order. Additionally, this file save the commands as
    pickle. This is a backup for future usage (i.e., maybe a continue search
    option will be build in at some point).

    Args:
        commands (list): List of command dictionaries.
        out_dir (str): Output directory.
    """
    fn_script = os.path.join(out_dir, 'commands.sh')
    fn_plain_script = os.path.join(out_dir, 'commands_wo_dirs.sh')
    fn_pickle = os.path.join(out_dir, 'commands.pickle')

    with open(fn_pickle, 'wb') as f:
        pickle.dump(commands, f)

    with open(fn_script, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('# This script contains all %d commands that are planned ' \
                % (len(commands)) + 'to be executed during this ' +
                'hyperparameter search. The order of execution is preserved ' +
                'in this script.\n\n')
        for cmd in commands:
            f.write('%s\n' % (_args_to_cmd_str(cmd)))

    with open(fn_plain_script, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('# This script contains all %d commands that are planned ' \
                % (len(commands)) + 'to be executed during this ' +
                'hyperparameter search. The order of execution is preserved ' +
                'in this script.\n')
        f.write('# Note, for visual clarity, the output directories have ' +
                'been omitted. See script "commands.sh" for the full ' +
                'commands.\n')
        for cmd in commands:
            cmd_wo_dir = dict(cmd)
            cmd_wo_dir.pop(_OUT_ARG, None)
            f.write('%s\n' % (_args_to_cmd_str(cmd_wo_dir)))

def _store_incomplete(commands, out_dir):
    """This function will pickle all command dictionaries of commands that have
    not been completed. This might be used to just continue an interrupted
    hyperparameter search.

    Args:
        commands: List of command dictionaries.
        out_dir: Output directory.
    """
    incomplete = []

    for i, cmd in enumerate(commands):
        if not _CMD_FINISHED[i]:
            incomplete.append(cmd)

    if len(incomplete) == 0:
        return

    warnings.warn('%d runs have not been completed.' % (len(incomplete)))
    fn_pickle = os.path.join(out_dir, 'not_completed.pickle')
    with open(fn_pickle, 'wb') as f:
        pickle.dump(incomplete, f)

def _read_config(config_mod, require_perf_eval_handle=False,
                 require_argparse_handle=False):
    """Parse the configuration module and check whether all attributes are set
    correctly.

    This function will set the corresponding global variables from this script
    appropriately.

    Args:
        config_mod: The implemented configuration template
            :mod:`hypnettorch.hpsearch.hpsearch_postprocessing`.
        require_perf_eval_handle: Whether :attr:`_PERFORMANCE_EVAL_HANDLE` has
            to be specified in the config file.
        require_argparse_handle: Whether :attr:`_ARGPARSE_HANDLE` has to be
            specified in the config file.
    """
    assert(hasattr(config_mod, '_SCRIPT_NAME'))
    assert(hasattr(config_mod, '_SUMMARY_FILENAME'))
    assert(hasattr(config_mod, '_SUMMARY_KEYWORDS') and \
           'finished' in config_mod._SUMMARY_KEYWORDS)
    globals()['_SCRIPT_NAME'] = config_mod._SCRIPT_NAME
    globals()['_SUMMARY_FILENAME'] = config_mod._SUMMARY_FILENAME
    globals()['_SUMMARY_KEYWORDS'] = config_mod._SUMMARY_KEYWORDS

    # Ensure downwards compatibility -- attributes did not exist previously.
    if hasattr(config_mod, '_OUT_ARG'):
        globals()['_OUT_ARG'] = config_mod._OUT_ARG

    if hasattr(config_mod, '_SUMMARY_PARSER_HANDLE') and \
            config_mod._SUMMARY_PARSER_HANDLE is not None:
        globals()['_SUMMARY_PARSER_HANDLE'] = config_mod._SUMMARY_PARSER_HANDLE
    else:
        globals()['_SUMMARY_PARSER_HANDLE'] = _get_performance_summary

    if require_perf_eval_handle:
        assert(hasattr(config_mod, '_PERFORMANCE_EVAL_HANDLE') and \
               config_mod._PERFORMANCE_EVAL_HANDLE is not None)
        globals()['_PERFORMANCE_EVAL_HANDLE'] = \
            config_mod._PERFORMANCE_EVAL_HANDLE
    else:
        if not hasattr(config_mod, '_PERFORMANCE_EVAL_HANDLE') or \
                config_mod._PERFORMANCE_EVAL_HANDLE is None:
            warnings.warn('Attribute "_PERFORMANCE_EVAL_HANDLE" not defined ' +
                          'in configuration file but might be required in ' +
                          'future releases.')

    if hasattr(config_mod, '_PERFORMANCE_KEY') and \
            config_mod._PERFORMANCE_KEY is not None:
        globals()['_PERFORMANCE_KEY'] = config_mod._PERFORMANCE_KEY
    else:
        globals()['_PERFORMANCE_KEY'] = config_mod._SUMMARY_KEYWORDS[0]

    if hasattr(config_mod, '_PERFORMANCE_SORT_ASC'):
        globals()['_PERFORMANCE_SORT_ASC'] = config_mod._PERFORMANCE_SORT_ASC

    if require_argparse_handle:
        assert(hasattr(config_mod, '_ARGPARSE_HANDLE') and \
               config_mod._ARGPARSE_HANDLE is not None)
        globals()['_ARGPARSE_HANDLE'] = config_mod._ARGPARSE_HANDLE

def hpsearch_cli_arguments(parser, show_num_searches=True, show_out_dir=True,
                           dout_dir='./out/hyperparam_search',
                           show_grid_module=True):
    """The CLI arguments of the hpsearch."""
    parser.add_argument('--deterministic_search', action='store_true',
                        help='If not selected, the order of configurations ' +
                             'is randomly picked.')
    if show_num_searches:
        parser.add_argument('--num_searches', type=int, metavar='N', default=-1,
                            help='If not -1, then the number of ' +
                                 'configurations that should be tested ' +
                                 'maximally. Default: %(default)s.')
    if show_out_dir:
        parser.add_argument('--out_dir', type=str, default=dout_dir,
                            help='Where should all the output files be ' +
                                 'written to? Note, a timestep is added to ' +
                                 'this path, except "force_out_dir" is set. ' +
                                 'Default: %(default)s.')
        parser.add_argument('--force_out_dir', action='store_true',
                            help='If enabled, the search will be stored in ' +
                                 'the exact location provided in "out_dir" ' +
                                 'and not a subfolder.')
        parser.add_argument('--dont_force_new_dir', action='store_true',
                            help='If enabled, the search can be stored in an '+
                                 'output folder that already exists. NOTE, ' +
                                 'this option is not a merging option. ' +
                                 'Previous hpsearch results will be ' +
                                 'overwritten if existing.')
    if show_grid_module:
        parser.add_argument('--grid_module', type=str, default=_DEFAULT_GRID,
                            help='Name of module to import from which to ' +
                                 'read the hyperparameter search grid. The ' +
                                 'module must define the two variables "grid" ' +
                                 'and "conditions". Default: %(default)s.')
        parser.add_argument('--grid_config', type=str, default='',
                            help='While a "grid_module" needs to be ' +
                                 'specified in order to have the hpsearch ' +
                                 'properly setup, the actual search "grid" ' +
                                 'and its "conditions" can be overwritten by ' +
                                 'providing the path to a pickle file here, ' +
                                 'containing "grid" and/or "conditions".')
    parser.add_argument('--dont_generate_full_grid', action='store_true',
                        help='If active, the full grid of possible ' +
                             'combinations will not be generated. Instead '+
                             'the desired number of commands are ' +
                             'generated by randomly picking values from ' +
                             'all provided command line arguments.')
    parser.add_argument('--run_cwd', type=str, default='.',
                        help='The working directory in which runs are ' +
                             'executed (in case the run script resides at a ' +
                             'different folder than this hpsearch script. ' +
                             'All outputs of this script will be relative to ' +
                             'this working directory (if output folder is ' +
                             'defined as relative folder). ' +
                             'Default: "%(default)s".')
    parser.add_argument('--run_cluster', action='store_true',
                        help='This option would produce jobs for a GPU ' +
                             'cluser running a job scheduler (see option ' +
                             '"scheduler".')
    parser.add_argument('--scheduler', type=str, default='lsf',
                        choices=['lsf', 'slurm'],
                        help='The job scheduler used on the cluster. ' +
                             'Default: %(default)s.')
    parser.add_argument('--num_jobs', type=int, metavar='N', default=8,
                        help='If "run_cluster" is activated, then this ' +
                             'option determines the maximum number of jobs ' +
                             'that can be submitted in parallel. ' +
                             'Default: %(default)s.')
    parser.add_argument('--num_hours', type=int, metavar='N', default=24,
                        help='If "run_cluster" is activated, then this ' +
                             'option determines the maximum number of hours ' +
                             'a job may run on the cluster. ' +
                             'Default: %(default)s.')
    parser.add_argument('--resources', type=str,
                        default='"rusage[mem=8000, ngpus_excl_p=1]"',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "lsf", then this option determines the ' +
                             'resources assigned to job in the ' +
                             'hyperparameter search (option -R of bsub). ' +
                             'Default: %(default)s.')
    parser.add_argument('--slurm_mem', type=str, default='8G',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "slurm", then this value will be passed as ' +
                             'argument "mem" of "sbatch". An empty string ' +
                             'means that "mem" will not be specified. ' +
                             'Default: %(default)s.')
    parser.add_argument('--slurm_gres', type=str, default='gpu:1',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "slurm", then this value will be passed as ' +
                             'argument "gres" of "sbatch". An empty string ' +
                             'means that "gres" will not be specified. ' +
                             'Default: %(default)s.')
    parser.add_argument('--slurm_partition', type=str, default='',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "slurm", then this value will be passed as ' +
                             'argument "partition" of "sbatch". An empty ' +
                             'string means that "partition" will not be ' +
                             'specified. Default: %(default)s.')
    parser.add_argument('--slurm_qos', type=str, default='',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "slurm", then this value will be passed as ' +
                             'argument "qos" of "sbatch". An empty string ' +
                             'means that "qos" will not be specified. ' +
                             'Default: %(default)s.')
    parser.add_argument('--slurm_constraint', type=str, default='',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "slurm", then this value will be passed as ' +
                             'argument "constraint" of "sbatch". An empty ' +
                             'string means that "constraint" will not be ' +
                             'specified. Default: %(default)s.')
    parser.add_argument('--visible_gpus', type=str, default='',
                        help='If "run_cluster" is NOT activated, then this ' +
                             'option determines the CUDA devices visible to ' +
                             'the hyperparameter search. A string of comma ' +
                             'separated integers is expected. If the list is ' +
                             'empty, then all GPUs of the machine are used. ' +
                             'The relative memory usage is specified, i.e., ' +
                             'a number between 0 and 1. If "-1" is given, ' +
                             'the jobs will be executed sequentially and not ' +
                             'assigned to a particular GPU. ' +
                             'Default: %(default)s.')
    parser.add_argument('--allowed_load', type=float, default=0.5,
                        help='If "run_cluster" is NOT activated, then this ' +
                             'option determines the maximum load a GPU may ' +
                             'have such that another process may start on ' +
                             'it. The relative load is specified, i.e., a ' +
                             'number between 0 and 1. Default: %(default)s.')
    parser.add_argument('--allowed_memory', type=float, default=0.5,
                        help='If "run_cluster" is NOT activated, then this ' +
                             'option determines the maximum memory usage a ' +
                             'GPU may have such that another process may ' +
                             'start on it. Default: %(default)s.')
    parser.add_argument('--sim_startup_time', type=int, metavar='N', default=60,
                        help='If "run_cluster" is NOT activated, then this ' +
                             'option determines the startup time of ' +
                             'simulations. If a job was assigned to a GPU, ' +
                             'then this time (in seconds) has to pass before ' +
                             'options "allowed_load" and "allowed_memory" ' +
                             'are checked to decide whether a new process ' +
                             'can be send to a GPU.Default: %(default)s.')
    parser.add_argument('--max_num_jobs_per_gpu', type=int, metavar='N',
                        default=1,
                        help='If "run_cluster" is NOT activated, then this ' +
                             'option determines the maximum number of jobs ' +
                             'per GPU that can be submitted in parallel. ' +
                             'Note, this script does not validate whether ' +
                             'other processes are already assigned to a GPU. ' +
                             'Default: %(default)s.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=42,
                        help='Random seed. Default: %(default)s.')

def run(argv=None, dout_dir='./out/hyperparam_search'):
    """Run the hyperparameter search script.

    Args:
        argv (list, optional): If provided, it will be treated as a list of
            command-line argument that is passed to the parser in place of
            :code:`sys.argv`.
        dout_dir (str, optional): The default value of command-line option
            ``--out_dir``.

    Returns:
        (str): The path to the CSV file containing the results of this search.
    """
    parser = argparse.ArgumentParser(description= \
        'hpsearch - Automatic Parameter Search -- ' +
        'Note, that the search values are defined in the source code of the ' +
        'accompanied configuration file!')
    hpsearch_cli_arguments(parser, dout_dir=dout_dir)
    # TODO build in "continue" option to finish incomplete commands.
    args = None
    if argv is not None:
        args = argv
    args = parser.parse_args(args=args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.run_cluster and args.scheduler == 'lsf':
        if bsub is None:
            raise ImportError('Package "bsub" is required for running a ' +
                'hyperparameter-search on the cluster using the LSF job ' +
                'scheduler. Please install it via ' +
                '"pip install -U --user bsub".')
    elif not args.run_cluster:
        if args.visible_gpus != '-1' and GPUtil is None:
            raise ImportError('Package "GPUtil" is required for this  hyper-' +
                'parameter search if option "--run_cluster" is not used. ' +
                'please install via "pip install -U --user gputil" as ' +
                'explained here: https://github.com/anderskm/gputil.')

    ### Get hyperparameter search grid from specified module.
    grid_module = importlib.import_module(args.grid_module)
    print('Loaded hp config from %s.' % grid_module.__file__)
    assert hasattr(grid_module, 'grid') and hasattr(grid_module, 'conditions')
    grid = grid_module.grid
    conditions = grid_module.conditions

    grid_config_provided = len(args.grid_config) > 0
    if grid_config_provided:
        grid_config_path = os.path.abspath(args.grid_config)
        print('Loading hp search grid and/or conditions from %s.' \
              % grid_config_path)
        with open(grid_config_path, "rb") as f:
            grid_config = pickle.load(f)
        assert isinstance(grid_config, dict)
        if 'grid' in grid_config.keys():
            grid = grid_config['grid']
        if 'conditions' in grid_config.keys():
            conditions = grid_config['conditions']

    assert(len(grid) > 0)

    _read_config(grid_module)

    print('### Running Hyperparameter Search ...')

    if len(conditions) > 0:
        print('Note, %d conditions have been defined and will be enforced!' % \
              len(conditions))

    if args.run_cwd != '.':
        os.chdir(args.run_cwd)
        print('Current working directory: %s.' % os.path.abspath(os.curdir))

    ### Output directory creation.
    # FIXME we should build in possibilities to merge with previous searches.
    hpsearch_dt = datetime.now()
    if args.force_out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(args.out_dir,
            'search_' + hpsearch_dt.strftime('%Y-%m-%d_%H-%M-%S'))
        # Sometimes on the cluster, hpsearches get scheduled simultaneously and
        # this is a simple way to assert that they don't collide.
        while os.path.exists(out_dir):
            time.sleep(1)
            hpsearch_dt = datetime.now()
            out_dir = os.path.join(args.out_dir,
                'search_' + hpsearch_dt.strftime('%Y-%m-%d_%H-%M-%S'))

    print('Results will be stored in %s.' % os.path.abspath(out_dir))

    if not args.dont_force_new_dir and os.path.exists(out_dir):
        raise RuntimeError('Output directory %s already exists!' % out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ### Backup hpsearch config.
    shutil.copyfile(grid_module.__file__, os.path.join(out_dir,
        os.path.basename(grid_module.__file__)))
    # NOTE We may not use the grid specified in the module, hence we better
    # backup the full grid in a human-readible form.
    if grid_config_provided:
        grid_config_bn = os.path.basename(grid_config_path)
        shutil.copyfile(grid_config_path, os.path.join(out_dir, grid_config_bn))
        gc_name, gc_ext = os.path.splitext(grid_config_bn)
        if gc_ext != 'json':
            with open(os.path.join(out_dir, gc_name+'.json'), "w") as f:
                json.dump({'grid': grid, 'conditions': conditions}, f)

    ### Build the grid.
    # We build a list of dictionaries with key value pairs.
    if args.dont_generate_full_grid:
        commands = _grid_to_commands_random_pick(grid, args.num_searches)
    else:
        commands = _grid_to_commands(grid)

    # Ensure, that all conditions can be enforced.
    orig_conditions = conditions
    conditions = []
    for i, cond in enumerate(orig_conditions):
        assert len(cond) == 2 and isinstance(cond[0], dict) \
               and isinstance(cond[1], dict)
        valid = True
        for k in cond[0].keys():
            if k not in grid.keys():
                warnings.warn('Condition %d can not be enforced. ' % (i) +
                    'Key "%s" is not specified in grid -- %s.' % (k, str(cond)))
                valid = False
        if valid:
            conditions.append(cond)

    # Now, we have the commands according to the grid, but we still need to
    # enforce the conditions.
    # This list will keep track of the conditions each command is affected.
    # FIXME We enforce conditions sequentially. But it could be, that the user
    # specifies conflicting conditions. E.g., condition 2 modifies commands such
    # that condition 1 would fire again.
    for i, cond_tup in enumerate(conditions):
        cond, action = cond_tup
        cond_keys = list(cond.keys())

        affected = [False] * len(commands)
        new_commands = []
        for j, command in enumerate(commands):
            # Figure out, whether condition i is satisfied for command j.
            comm_keys = command.keys()
            key_satisfied = [False] * len(cond_keys)
            for l, cond_key in enumerate(cond_keys):
                if cond_key in comm_keys:
                    cond_vals = cond[cond_key]
                    if command[cond_key] in cond_vals:
                        key_satisfied[l] = True

            if np.all(key_satisfied):
                affected[j] = True
            else:
                continue

            # Generate a set of replacement commands for command j, such that
            # condition i is satisfied.
            cmds = _grid_to_commands(action)
            for l, cmd in enumerate(cmds):
                for k in comm_keys:
                    if k not in cmd.keys():
                        cmds[l][k] = command[k]
            new_commands.extend(cmds)

        # Remove all commands affected by this condition and insert the new
        # ones.
        old_cmds = commands
        commands = []
        for j, cmd in enumerate(old_cmds):
            if not affected[j]:
                commands.append(cmd)
        commands.extend(new_commands)

    # Note, the way we enforced conditions above may result in duplicates.
    # We need to remove them now.
    old_cmds = [(hash(frozenset(cmd.items())), cmd) for cmd in commands]
    # Sort commands according to their hash value.
    old_cmds = sorted(old_cmds, key=lambda tup: tup[0])
    commands = []
    i = 0
    while i < len(old_cmds):
        hash_i, cmd_i = old_cmds[i]

        # Check if current command has duplicate.
        if i < len(old_cmds)-1:
            hash_next, cmd_next = old_cmds[i+1]
            if hash_i == hash_next:
                warnings.warn('Command duplicate found! The following ' +
                    'commands have been identified as duplicates. The first ' +
                    'one will be removed.\n--- %s\n--- %s' \
                    % (_args_to_cmd_str(cmd_i), _args_to_cmd_str(cmd_next)))
                i += 1
                continue

        commands.append(cmd_i)
        i += 1

    ### Random shuffling of command execution order.
    if not args.deterministic_search:
        random.shuffle(commands)

    ### Consider the maximum number of commands we may execute.
    if args.num_searches != -1 and len(commands) > args.num_searches:
        print('Only %d of %d configurations will be tested!' % \
              (args.num_searches, len(commands)))
        commands = commands[:args.num_searches]

    ### Print all commands to user to allow visual verification.
    print('\n### List of all commands. Please verify carefully. ###\n')
    for cmd in commands:
        print(_args_to_cmd_str(cmd))
    print('\nThe %d command(s) above will be executed.' % len(commands))
    _CMD_FINISHED = [False] * len(commands)

    ### Assign an output directory to each command.
    # We do this after the shuffling to make sure the folder are displayed in
    # their execution order.
    # We also do it after the printing above to avoid visual clutter.

    # Identifier of current hpsearch. Why do we add such a cryptic number to the
    # simulation output folders? We need the folder names of different
    # hpsearches to be different, as we use the folder names to name the job
    # files (.out and .err) that are written by the LSF Batch system on the
    # cluster. Those files are all stored in the same folder, even if coming
    # from different hpsearches.
    # FIXME There is for sure a better solution.
    hpsearch_ident = hpsearch_dt.strftime("%Y%m%d%H%M%S")
    num_cmds = len(commands)
    n_digits = int(np.floor(np.log10(num_cmds))) + 1
    for i, cmd in enumerate(commands):
        assert _OUT_ARG not in cmd.keys()
        folder_name = 'sim_%s_%s' % (hpsearch_ident, str(i+1).zfill(n_digits))
        cmd[_OUT_ARG] = os.path.join(out_dir, folder_name)

    # The list of command strings will be dumped into a file, such that the
    # user sees their order.
    _backup_commands(commands, out_dir)

    ### Hyperparameter Search
    # Where do we summarize the results?
    results_file = os.path.join(out_dir, 'search_results.csv')

    try:
        _run_cmds(args, commands, out_dir, results_file)
    except:
        traceback.print_exc(file=sys.stdout)
        warnings.warn('An error occurred during the hyperparameter search.')

    _store_incomplete(commands, out_dir)

    ### Sort CSV file according to performance key.
    try:
        csv_file_content = pandas.read_csv(results_file, sep=';')
        csv_file_content = csv_file_content.sort_values(_PERFORMANCE_KEY,
            ascending=_PERFORMANCE_SORT_ASC)
        csv_file_content.to_csv(results_file, sep=';', index=False)
    except:
        traceback.print_exc(file=sys.stdout)
        warnings.warn('No results have been gathered during this hpsearch.')

    print('### Running Hyperparameter Search ... Done')

    return results_file

if __name__ == '__main__':
    run()
