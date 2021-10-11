# Copyright ©2016.  The University of Chicago (“Chicago”). All Rights Reserved.
# Created by Nicolas Y. Masse, Gregory D. Grant, David J. Freedman, University
# of Chicago
# Retrieved from: https://github.com/nmasse/Context-Dependent-Gating/blob/master/parameters.py
# The license can be obtained from: https://github.com/nmasse/Context-Dependent-Gating/blob/master/LICENSE
# A copy of the license is provided below:
#
# Permission to use, copy, modify, and distribute this software, including all
# object code and source code, and any accompanying documentation (together the
# “Program”) for educational and not-for-profit research purposes, without fee
# and without a signed licensing agreement, is hereby granted, provided that the
# above copyright notice, this paragraph and the following three paragraphs
# appear in all copies, modifications, and distributions. For the avoidance of
# doubt, educational and not-for-profit research purposes excludes any service
# or part of selling a service that uses the Program. To obtain a commercial
# license for the Program, contact the Technology Commercialization and
# Licensing, Polsky Center for Entrepreneurship and Innovation, University of
# Chicago, 1452 East 53rd Street, 2nd floor, Chicago, IL 60615.
#
# Created by Nicolas Y. Masse, Gregory D. Grant, David J. Freedman, University
# of Chicago
#
# The Program is copyrighted by Chicago. The Program is supplied "as is",
# without any accompanying services from Chicago. Chicago does not warrant that
# the operation of the Program will be uninterrupted or error-free. The end-user
# understands that the Program was developed for research purposes and is
# advised not to rely exclusively on the Program for any reason.
#
# IN NO EVENT SHALL CHICAGO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
# SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING
# OUT OF THE USE OF THE PROGRAM, EVEN IF CHICAGO HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE. CHICAGO SPECIFICALLY DISCLAIMS ANY WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE.  THE PROGRAM PROVIDED HEREUNDER IS PROVIDED
# "AS IS".  CHICAGO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
# ENHANCEMENTS, OR MODIFICATIONS.

import numpy as np

#print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par
par = {
    # Setup parameters
    'save_dir'              : 'C:/Users/Benjamin Ehret/Documents/cognet/', 
    'stabilization'         : 'pathint',    # None or 'pathint' (Zenke method)
    'save_analysis'         : False,
    'reset_weights'         : False,        # reset weights between tasks

    # Network configuration
    'synapse_config'        : 'std_stf',    # Full is 'std_stf', otherwise None
    'exc_inh_prop'          : 0.8,          # Literature 0.8, for EI off 1
    'var_delay'             : False,
    'training_method'       : 'RL',         # 'SL', 'RL'
    'architecture'          : 'BIO',        # 'BIO', 'LSTM'

    # Network shape
    'num_motion_tuned'      : 64,
    'num_fix_tuned'         : 4,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 256,
    'n_val'                 : 1,
    'include_rule_signal'   : True,

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 1e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 1.0,
    'discount_rate'         : 0.,

    # Variance values
    'clip_max_grad_val'     : 1.0,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.0,
    'noise_rnn_sd'          : 0.05,

    # Task specs
    'task'                  : 'multistim',  # See stimulus file for more options
    'n_tasks'               : 20,
    'multistim_trial_length': 2000,
    'mask_duration'         : 0,
    'dead_time'             : 200,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4.0,          # von Mises magnitude scaling factor

    # Cost values
    'spike_cost'            : 1e-7,
    'weight_cost'           : 0.,
    'entropy_cost'          : 0.001,
    'val_cost'              : 0.01,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_size'            : 256,
    'n_train_batches'       : 5000, #50000,

    # Omega parameters
    'omega_c'               : 2.,
    'omega_xi'              : 0.001,

    # Gating parameters
    'gating_type'           : None, # 'XdG', 'split', or None
    'gate_pct'              : 0.8,  # Num. gated hidden units for 'XdG' only
    'n_subnetworks'         : 4,    # Num. subnetworks for 'split' only

    # Stimulus parameters
    'fix_break_penalty'     : -1.,
    'wrong_choice_penalty'  : -0.01,
    'correct_choice_reward' : 1.,

}


############################
### Dependent parameters ###
############################


def update_parameters(updates):
    """ Updates parameters based on a provided
        dictionary, then updates dependencies """
    
    for (key, val) in updates.items():
        par[key] = val
        print('Updating: {:<24} --> {}'.format(key, val))
        
    update_dependencies()


def update_dependencies():
    """ Updates all parameter dependencies """

    ###
    ### Putting together network structure
    ###

    # Turn excitatory-inhibitory settings on or off
    if par['architecture'] == 'BIO':
        par['EI'] = True if par['exc_inh_prop'] < 1 else False
    elif par['architecture'] == 'LSTM':
        print('Using LSTM networks; setting to EI to False')
        par['EI'] = False
        par['exc_inh_prop'] = 1.
        par['synapse_config'] = None
        par['spike_cost'] = 0.

    # Generate EI matrix
    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']
    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    if par['EI']:
        n = par['n_hidden']//par['num_inh_units']
        par['ind_inh'] = np.arange(n-1,par['n_hidden'],n)
        par['EI_list'][par['ind_inh']] = -1.
    par['EI_matrix'] = np.diag(par['EI_list'])

    # Number of output neurons
    par['n_output'] = par['num_motion_dirs'] + 1
    par['n_pol'] = par['num_motion_dirs'] + 1

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']

    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])

    # Specify time step in seconds and neuron time constant
    par['dt_sec'] = par['dt']/1000
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']

    # Generate noise deviations
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd']

    # Set trial step length
    par['num_time_steps'] = par['multistim_trial_length']//par['dt']

    # Set up gating vectors for hidden layer
    gen_gating()

    ###
    ### Setting up weights, biases, masks, etc.
    ###

    # Specify initial RNN state
    par['h_init'] = 0.1*np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)

    # Initialize weights
    c = 0.05

    par['W_in_init'] = c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_input'], par['n_hidden']]))
    par['W_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_output']]))

    if par['EI']:
        par['W_rnn_init'] = c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], par['n_hidden']]))
        par['W_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
        par['W_rnn_init'] *= par['W_rnn_mask']
    else:
        par['W_rnn_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['W_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32)

    # Initialize biases
    par['b_rnn_init'] = np.zeros((1,par['n_hidden']), dtype = np.float32)
    par['b_out_init'] = np.zeros((1,par['n_output']), dtype = np.float32)

    # Specify masks
    par['W_out_mask'] = np.ones((par['n_hidden'], par['n_output']), dtype=np.float32)
    par['W_in_mask'] = np.ones((par['n_input'], par['n_hidden']), dtype=np.float32)
    if par['EI']:
        par['W_out_init'][par['ind_inh'], :] = 0
        par['W_out_mask'][par['ind_inh'], :] = 0

    # Initialize RL-specific weights
    par['W_pol_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_pol']]))
    par['b_pol_out_init'] = np.zeros((1,par['n_pol']), dtype = np.float32)

    par['W_val_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_val']]))
    par['b_val_out_init'] = np.zeros((1,par['n_val']), dtype = np.float32)

    ###
    ### Setting up LSTM weights and biases, if required
    ###

    if par['architecture'] == 'LSTM':
        c = 0.05
        par['Wf_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))
        par['Wi_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))
        par['Wo_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))
        par['Wc_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))

        par['Uf_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Ui_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Uo_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Uc_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))


        par['bf_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bi_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bo_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bc_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)

    ###
    ### Setting up synaptic plasticity parameters
    ###

    """
    0 = static
    1 = facilitating
    2 = depressing
    """

    par['synapse_type'] = np.zeros(par['n_hidden'], dtype=np.int8)

    # only facilitating synapses
    if par['synapse_config'] == 'stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)

    # only depressing synapses
    elif par['synapse_config'] == 'std':
        par['synapse_type'] = 2*np.ones(par['n_hidden'], dtype=np.int8)

    # even numbers facilitating, odd numbers depressing
    elif par['synapse_config'] == 'std_stf':
        par['synapse_type'] = 2*np.ones(par['n_hidden'], dtype=np.int8)
        ind = range(1,par['n_hidden'],2)
        #par['synapse_type'][par['ind_inh']] = 1
        par['synapse_type'][ind] = 1

    par['alpha_stf'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['alpha_std'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['U'] = np.ones((par['n_hidden'], 1), dtype=np.float32)

    # initial synaptic values
    par['syn_x_init'] = np.zeros((par['n_hidden'], par['batch_size']), dtype=np.float32)
    par['syn_u_init'] = np.zeros((par['n_hidden'], par['batch_size']), dtype=np.float32)

    for i in range(par['n_hidden']):
        if par['synapse_type'][i] == 1:
            par['alpha_stf'][i,0] = par['dt']/par['tau_slow']
            par['alpha_std'][i,0] = par['dt']/par['tau_fast']
            par['U'][i,0] = 0.15
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

        elif par['synapse_type'][i] == 2:
            par['alpha_stf'][i,0] = par['dt']/par['tau_fast']
            par['alpha_std'][i,0] = par['dt']/par['tau_slow']
            par['U'][i,0] = 0.45
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

    par['alpha_stf'] = np.transpose(par['alpha_stf'])
    par['alpha_std'] = np.transpose(par['alpha_std'])
    par['U'] = np.transpose(par['U'])
    par['syn_x_init'] = np.transpose(par['syn_x_init'])
    par['syn_u_init'] = np.transpose(par['syn_u_init'])


def gen_gating():
    """
    Generate the gating signal to applied to all hidden units
    """
    par['gating'] = []

    for t in range(par['n_tasks']):
        gating_task = np.zeros(par['n_hidden'], dtype=np.float32)
        for i in range(par['n_hidden']):

            if par['gating_type'] == 'XdG':
                if np.random.rand() < 1-par['gate_pct']:
                    gating_task[i] = 1

            elif par['gating_type'] == 'split':
                if t%par['n_subnetworks'] == i%par['n_subnetworks']:
                    gating_layer[i] = 1

            elif par['gating_type'] is None:
                gating_task[i] = 1

        par['gating'].append(gating_task)


def initialize_weight(dims, connection_prob):
    w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


update_dependencies()
#print("--> Parameters successfully loaded.\n")
