# Copyright ©2016.  The University of Chicago (“Chicago”). All Rights Reserved.
# Created by Nicolas Y. Masse, Gregory D. Grant, David J. Freedman, University
# of Chicago
# Retrieved from: https://github.com/nmasse/Context-Dependent-Gating/blob/master/stimulus.py
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
from hypnettorch.data.timeseries.cognitive_tasks.parameters import par

#print('Using \'Multistim\' stimulus file.')

class MultiStimulus:

    def __init__(self,rstate):

        
        # set random state
        self.rstate = rstate
        
        # Shape configuration
        self.input_shape    = [par['num_time_steps'], par['batch_size'],par['n_input'] ]
        self.output_shape   = [par['num_time_steps'], par['batch_size'],par['n_output'] ]
        self.stimulus_shape = [par['num_time_steps'], par['batch_size'],par['num_motion_tuned'] ]
        self.response_shape = [par['num_time_steps'], par['batch_size'],par['num_motion_dirs'] ]
        self.fixation_shape = [par['num_time_steps'], par['batch_size'],par['num_fix_tuned'] ]
        self.rule_shape = [par['num_time_steps'], par['batch_size'],par['num_rule_tuned'] ]
        self.mask_shape     = [par['num_time_steps'], par['batch_size']]

        # Motion and stimulus configuration
        self.motion_dirs        = np.linspace(0,2*np.pi-2*np.pi/par['num_motion_dirs'],par['num_motion_dirs'])
        self.stimulus_dirs      = np.linspace(0,2*np.pi-2*np.pi/(par['num_motion_tuned']//2),(par['num_motion_tuned']//2))
        self.pref_motion_dirs   = np.reshape(np.linspace(0,2*np.pi-2*np.pi/(par['num_motion_tuned']//2), \
            (par['num_motion_tuned']//2)), (par['num_motion_tuned']//2,1))
        self.modality_size      = (par['num_motion_tuned'])//2

        self.fix_time = 400

        # Go task stuff
        self.go_delay = np.array([200, 400, 800])//par['dt']

        # DM task stuff
        self.dm_c_set = np.array([-0.4, -0.2, -0.1, 0.1, 0.2, 0.4])
        self.dm_stim_lengths = np.array([200,400,800])//par['dt']

        # DM Dly task stuff
        self.dm_dly_c_set = np.array([-0.4, -0.2, -0.1, 0.1, 0.2, 0.4])
        self.dm_dly_delay = np.array([200, 400, 800])//par['dt']

        # Matching task stuff
        self.match_delay = np.array([200, 400, 800])//par['dt']

        # Initialize task interface
        self.get_tasks()
        self.task_id = 0
        self.task_order = np.arange(len(self.task_types))

        self.rule_signal_factor = 1. if par['include_rule_signal'] else 0.


    def circ_tuning(self, theta):

        ang_dist = np.angle(np.exp(1j*theta - 1j*self.pref_motion_dirs))
        return par['tuning_height']*np.exp(-0.5*(8*ang_dist/np.pi)**2)


    def get_tasks(self):
        if par['task'] == 'multistim':
            self.task_types = [
                [self.task_go, 'go', 0],
                [self.task_go, 'rt_go', 0],
                [self.task_go, 'dly_go', 0],

                [self.task_go, 'go', np.pi],
                [self.task_go, 'rt_go', np.pi],
                [self.task_go, 'dly_go', np.pi],

                [self.task_dm, 'dm1'],
                [self.task_dm, 'dm2'],
                [self.task_dm, 'ctx_dm1'],
                [self.task_dm, 'ctx_dm2'],
                [self.task_dm, 'multsen_dm'],

                [self.task_dm_dly, 'dm1_dly'],
                [self.task_dm_dly, 'dm2_dly'],
                [self.task_dm_dly, 'ctx_dm1_dly'],
                [self.task_dm_dly, 'ctx_dm2_dly'],
                [self.task_dm_dly, 'multsen_dm_dly'],

                [self.task_matching, 'dms'],
                [self.task_matching, 'dmc'],
                [self.task_matching, 'dnms'],
                [self.task_matching, 'dnmc']
            ]

        elif par['task'] == 'twelvestim':
            self.task_types = [
                [self.task_go, 'go', 0],
                [self.task_go, 'dly_go', 0],

                [self.task_dm, 'dm1'],
                [self.task_dm, 'dm2'],
                [self.task_dm, 'ctx_dm1'],
                [self.task_dm, 'ctx_dm2'],
                [self.task_dm, 'multsen_dm'],

                [self.task_dm_dly, 'dm1_dly'],
                [self.task_dm_dly, 'dm2_dly'],
                [self.task_dm_dly, 'ctx_dm1_dly'],
                [self.task_dm_dly, 'ctx_dm2_dly'],
                [self.task_dm_dly, 'multsen_dm_dly']
            ]
        else:
            raise Exception('Multistimulus task type \'{}\' not yet implemented.'.format(par['task']))

        return self.task_types


    def generate_trial(self, current_task):

        self.trial_info = {
            'neural_input'   : self.rstate.normal(par['input_mean'], par['noise_in'], size=self.input_shape),
            'desired_output' : np.zeros(self.output_shape, dtype=np.float32),
            'reward_data'    : np.zeros(self.output_shape, dtype=np.float32),
            'train_mask'     : np.ones(self.mask_shape, dtype=np.float32)}

        self.trial_info['train_mask'][:par['dead_time']//par['dt'], :] = 0

        if par['num_rule_tuned'] > 0:
            rule_signal = np.zeros((1,1,par['num_rule_tuned']))
            rule_signal[0,0,current_task] = par['tuning_height']
            self.trial_info['neural_input'][:, :, -par['num_rule_tuned']:] += rule_signal*self.rule_signal_factor

        task = self.task_types[current_task]    # Selects a task from the list
        task[0](*task[1:])                      # Generates that task into trial_info

        # Apply reinforcement learning task specifications
        if par['training_method'] == 'RL':

            # Iterate over batches
            for b in range(par['batch_size']):

                # Designate timings
                respond_time    = np.where(np.sum(self.trial_info['desired_output'][:,b,:-1], axis=1) > 0)[0]

                fix_time        = list(range(respond_time[0])) if len(respond_time) > 0 else [-1]
                respond_time    = respond_time if len(respond_time) > 0 else [-1]

                # Designate responses
                correct_response    = np.where(self.trial_info['desired_output'][respond_time[0],b,:]==1)[0]
                incorrect_response  = np.where(self.trial_info['desired_output'][respond_time[0],b,:-1]==0)[0]

                # Build reward data
                self.trial_info['reward_data'][fix_time,b,:-1] = par['fix_break_penalty']
                self.trial_info['reward_data'][respond_time,b,correct_response] = par['correct_choice_reward']
                for i in incorrect_response:
                    self.trial_info['reward_data'][respond_time,b,i] = par['wrong_choice_penalty']

                # Penalize fixating throughout entire trial if response was required
                if not self.trial_info['desired_output'][-1,b,-1] == 1:
                    self.trial_info['reward_data'][-1,b,-1] = par['fix_break_penalty']
                else:
                    self.trial_info['reward_data'][-1,b,-1] = par['correct_choice_reward']

        # Returns the task name and trial info
        return task[1], self.trial_info['neural_input'], self.trial_info['desired_output'], \
            self.trial_info['train_mask'], self.trial_info['reward_data']


    def task_go(self, variant='go', offset=0):

        # Task parameters
        if variant == 'go':
            stim_onset = self.rstate.randint(self.fix_time, self.fix_time+1000, par['batch_size'])//par['dt']
            stim_off = -1
            fixation_end = np.ones(par['batch_size'], dtype=np.int16)*(self.fix_time+1000)//par['dt']
            resp_onset = fixation_end
        elif variant == 'rt_go':
            stim_onset = self.rstate.randint(self.fix_time, self.fix_time+1000, par['batch_size'])//par['dt']
            stim_off = -1
            fixation_end = np.ones(par['batch_size'],dtype=np.int16)*par['num_time_steps']
            resp_onset = stim_onset
        elif variant == 'dly_go':
            stim_onset = self.fix_time//par['dt']*np.ones((par['batch_size']),dtype=np.int16)
            stim_off = (self.fix_time+300)//par['dt']
            fixation_end = stim_off + self.rstate.choice(self.go_delay, size=par['batch_size'])
            resp_onset = fixation_end
        else:
            raise Exception('Bad task variant.')

        # Need dead time
        self.trial_info['train_mask'][:par['dead_time']//par['dt'], :] = 0

        for b in range(par['batch_size']):

            # Input neurons index above par['num_motion_tuned'] encode fixation
            self.trial_info['neural_input'][:fixation_end[b], b, par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned']] \
                += par['tuning_height']
            """
            self.trial_info['neural_input'][:fixation_end[b], b, par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned']] \
                += par['tuning_height']
            """
            modality   = self.rstate.randint(2)
            neuron_ind = range(self.modality_size*modality, self.modality_size*(1+modality))
            stim_dir   = self.rstate.choice(self.motion_dirs)
            target_ind = int(np.round(par['num_motion_dirs']*(stim_dir+offset)/(2*np.pi))%par['num_motion_dirs'])

            self.trial_info['neural_input'][stim_onset[b]:stim_off, b, neuron_ind] += np.reshape(self.circ_tuning(stim_dir),(1,-1))
            self.trial_info['desired_output'][resp_onset[b]:, b, target_ind] = 1
            self.trial_info['desired_output'][:resp_onset[b], b, -1] = 1

            self.trial_info['train_mask'][resp_onset[b]:resp_onset[b]+par['mask_duration']//par['dt'], b] = 0

        return self.trial_info


    def task_dm(self, variant='dm1'):

        # Create trial stimuli
        stim_dir1 = self.rstate.choice(self.motion_dirs, [1, par['batch_size']])
        stim_dir2 = (stim_dir1 + np.pi/2 + self.rstate.choice(self.motion_dirs[::2], [1, par['batch_size']])/2)%(2*np.pi)

        stim1 = self.circ_tuning(stim_dir1)
        stim2 = self.circ_tuning(stim_dir2)

        # Determine the strengths of the stimuli in each modality
        c_mod1 = self.rstate.choice(self.dm_c_set, [1, par['batch_size']])
        c_mod2 = self.rstate.choice(self.dm_c_set, [1, par['batch_size']])
        mean_gamma = 0.8 + 0.4*self.rstate.rand(1, par['batch_size'])
        gamma_s1_m1 = mean_gamma + c_mod1
        gamma_s2_m1 = mean_gamma - c_mod1
        gamma_s1_m2 = mean_gamma + c_mod2
        gamma_s2_m2 = mean_gamma - c_mod2

        # Determine response directions and convert to output indices
        resp_dir_mod1 = np.where(gamma_s1_m1 > gamma_s2_m1, stim_dir1, stim_dir2)
        resp_dir_mod2 = np.where(gamma_s1_m2 > gamma_s2_m2, stim_dir1, stim_dir2)
        resp_dir_sum  = np.where(gamma_s1_m1 + gamma_s1_m2 > gamma_s2_m1 + gamma_s2_m2, stim_dir1, stim_dir2)

        resp_dir_mod1 = np.round(par['num_motion_dirs']*resp_dir_mod1/(2*np.pi))
        resp_dir_mod2 = np.round(par['num_motion_dirs']*resp_dir_mod2/(2*np.pi))
        resp_dir_sum  = np.round(par['num_motion_dirs']*resp_dir_sum/(2*np.pi))

        # Apply stimuli to modalities and build appropriate response
        if variant == 'dm1':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = np.zeros_like(stim1)
            resp_dirs = resp_dir_mod1
        elif variant == 'dm2':
            modality1 = np.zeros_like(stim1)
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'ctx_dm1':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod1
        elif variant == 'ctx_dm2':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'multsen_dm':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_sum
        else:
            raise Exception('Bad task variant.')

        resp = np.zeros([par['num_motion_dirs'], par['batch_size']])
        for b in range(par['batch_size']):
            resp[np.int16(resp_dirs[0,b]%par['num_motion_dirs']),b] = 1

        # Setting up arrays
        fixation = np.zeros(self.fixation_shape)
        response = np.zeros(self.response_shape)
        stimulus = np.zeros(self.stimulus_shape)
        mask     = np.ones(self.mask_shape)
        mask[:par['dead_time']//par['dt'],:] = 0
        resp_fix  = np.copy(fixation[:,:,0:1])

        # Identify stimulus onset for each trial and build each trial from there
        stim_onset = self.fix_time//par['dt']
        stim_off   = stim_onset + self.rstate.choice(self.dm_stim_lengths, par['batch_size'])
        for b in range(par['batch_size']):
            fixation[:stim_off[b],b,:] = par['tuning_height']
            resp_fix[:stim_off[b],b] = 1
            stimulus[stim_onset:stim_off[b],b,:] = np.transpose(np.concatenate([modality1[:,b], modality2[:,b]], axis=0)[:,np.newaxis])
            response[stim_off[b]:,b,:] = np.transpose(resp[:,b,np.newaxis])
            mask[stim_off[b]:stim_off[b]+par['mask_duration']//par['dt'],b] = 0

        # Merge activies and fixations into single vector
        stimulus = np.concatenate([stimulus, fixation], axis=2)
        response = np.concatenate([response, resp_fix], axis=2)

        self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
        self.trial_info['desired_output'] = response
        self.trial_info['train_mask'] = mask

        return self.trial_info


    def task_dm_dly(self, variant='dm1'):

        # Create trial stimuli
        stim_dir1 = 2*np.pi*self.rstate.rand(1, par['batch_size'])
        stim_dir2 = (stim_dir1 + np.pi/2 + np.pi*self.rstate.rand(1, par['batch_size']))%(2*np.pi)
        stim1 = self.circ_tuning(stim_dir1)
        stim2 = self.circ_tuning(stim_dir2)

        # Determine the strengths of the stimuli in each modality
        c_mod1 = self.rstate.choice(self.dm_dly_c_set, [1, par['batch_size']])
        c_mod2 = self.rstate.choice(self.dm_dly_c_set, [1, par['batch_size']])
        mean_gamma = 0.8 + 0.4*self.rstate.rand(1, par['batch_size'])
        gamma_s1_m1 = mean_gamma + c_mod1
        gamma_s2_m1 = mean_gamma - c_mod1
        gamma_s1_m2 = mean_gamma + c_mod2
        gamma_s2_m2 = mean_gamma - c_mod2

        # Determine the delay for each trial
        delay = self.rstate.choice(self.dm_dly_delay, [1, par['batch_size']])

        # Determine response directions and convert to output indices
        resp_dir_mod1 = np.where(gamma_s1_m1 > gamma_s2_m1, stim_dir1, stim_dir2)
        resp_dir_mod2 = np.where(gamma_s1_m2 > gamma_s2_m2, stim_dir1, stim_dir2)
        resp_dir_sum  = np.where(gamma_s1_m1 + gamma_s1_m2 > gamma_s2_m1 + gamma_s2_m2, stim_dir1, stim_dir2)

        resp_dir_mod1 = np.round(par['num_motion_dirs']*resp_dir_mod1/(2*np.pi))
        resp_dir_mod2 = np.round(par['num_motion_dirs']*resp_dir_mod2/(2*np.pi))
        resp_dir_sum  = np.round(par['num_motion_dirs']*resp_dir_sum/(2*np.pi))

        # Apply stimuli to modalities and build appropriate response
        if variant == 'dm1_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = np.zeros_like(stim1)
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = np.zeros_like(stim2)
            resp_dirs = resp_dir_mod1
        elif variant == 'dm2_dly':
            modality1_t1 = np.zeros_like(stim1)
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = np.zeros_like(stim2)
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'ctx_dm1_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod1
        elif variant == 'ctx_dm2_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'multsen_dm_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_sum
        else:
            raise Exception('Bad task variant.')

        resp = np.zeros([par['num_motion_dirs'], par['batch_size']])
        for b in range(par['batch_size']):
            resp[np.int16(resp_dirs[0,b]%par['num_motion_dirs']),b] = 1

        # Setting up arrays
        fixation = np.zeros(self.fixation_shape)
        response = np.zeros(self.response_shape)
        stimulus = np.zeros(self.stimulus_shape)
        mask     = np.ones(self.mask_shape)
        mask[:par['dead_time']//par['dt'],:] = 0
        resp_fix  = np.copy(fixation[:,:,0:1])

        # Identify stimulus onset for each trial and build each trial from there
        stim_on1   = self.fix_time//par['dt']
        stim_off1  = (self.fix_time+300)//par['dt']
        stim_on2   = delay + stim_off1
        stim_off2  = stim_on2 + 300//par['dt']
        resp_time  = stim_off2 + 0//par['dt']
        for b in range(par['batch_size']):
            fixation[:resp_time[0,b],b,:] = par['tuning_height']
            resp_fix[:resp_time[0,b],b] = 1
            stimulus[stim_on1:stim_off1,b,:] = np.concatenate([modality1_t1[:,b], modality2_t1[:,b]], axis=0)[np.newaxis,:]
            stimulus[stim_on2[0,b]:stim_off2[0,b],b] = np.concatenate([modality1_t2[:,b], modality2_t2[:,b]], axis=0)[np.newaxis,:]
            response[resp_time[0,b]:,b,:] = resp[np.newaxis,:,b]
            mask[resp_time[0,b]:resp_time[0,b]+par['mask_duration'],b] = 0

        # Merge activies and fixations into single vectors
        stimulus = np.concatenate([stimulus, fixation], axis=2)
        response = np.concatenate([response, resp_fix], axis=2)    # Duplicates starting fixation on output

        self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
        self.trial_info['desired_output'] = response
        self.trial_info['train_mask'] = mask

        return self.trial_info


    def task_matching(self, variant='dms'):

        # Determine matches, and get stimuli
        if variant in ['dms', 'dnms']:
            stim1 = self.rstate.choice(self.motion_dirs, par['batch_size'])
            nonmatch = (stim1 + self.rstate.choice(self.motion_dirs[1:], par['batch_size']))%(2*np.pi)

            match = self.rstate.choice(np.array([True, False]), par['batch_size'])
            stim2 = np.where(match, stim1, nonmatch)

        elif variant in ['dmc', 'dnmc']:
            stim1 = self.rstate.choice(self.motion_dirs, par['batch_size'])
            stim2 = self.rstate.choice(self.motion_dirs, par['batch_size'])

            stim1_cat = np.logical_and(np.less(-1e-3, stim1), np.less(stim1, np.pi))
            stim2_cat = np.logical_and(np.less(-1e-3, stim2), np.less(stim2, np.pi))
            match = np.logical_not(np.logical_xor(stim1_cat, stim2_cat))
        else:
            raise Exception('Bad variant.')

        # Establishing stimuli
        stimulus1 = self.circ_tuning(stim1)
        stimulus2 = self.circ_tuning(stim2)

        # Convert to response
        stim1_int = np.round(par['num_motion_dirs']*stim1/(2*np.pi))
        stim2_int = np.round(par['num_motion_dirs']*stim2/(2*np.pi))

        if variant in ['dms', 'dmc']:
            resp = np.where(match, stim1_int, -1)
        elif variant in ['dnms', 'dnmc']:
            resp = np.where(match, -1, stim2_int)
        else:
            raise Exception('Bad variant.')

        # Setting up arrays
        modality_choice = self.rstate.choice(np.array([0,1], dtype=np.int16), [2, par['batch_size']])
        modalities = np.zeros([2, par['num_time_steps'], par['batch_size'], par['num_motion_tuned']//2])
        fixation = np.zeros(self.fixation_shape)
        response = np.zeros(self.response_shape)
        stimulus = np.zeros(self.stimulus_shape)
        mask     = np.ones(self.mask_shape)
        mask[:par['dead_time']//par['dt'],:] = 0

        # Decide timings and build each trial
        stim1_on  = self.fix_time//par['dt']
        stim1_off = (self.fix_time+300)//par['dt']
        stim2_on  = stim1_off + self.rstate.choice(self.match_delay, par['batch_size'])
        stim2_off = stim2_on + 300//par['dt']
        resp_time = stim2_off
        resp_fix  = np.copy(fixation[:,:,0:1])

        for b in range(par['batch_size']):
            fixation[:resp_time[b],b,:] = par['tuning_height']

            # Ensuring that sample and test stimuli are in same modality (RF)
            modalities[modality_choice[0,b],stim1_on:stim1_off,b,:] = stimulus1[np.newaxis,:,b]
            modalities[modality_choice[0,b],stim2_on[b]:stim2_off[b],b,:] = stimulus2[np.newaxis,:,b]

            mask[resp_time[b]:resp_time[b]+par['mask_duration']//par['dt'],b] = 0
            if not resp[b] == -1:
                response[resp_time[b]:,b,int(resp[b])] = 1
                resp_fix[:resp_time[b],b] = 1
            else:
                resp_fix[:,b,:] = 1

        # Merge activies and fixations into single vectors)
        stimulus = np.concatenate([modalities[0], modalities[1], fixation], axis=2)
        response = np.concatenate([response, resp_fix], axis=2)

        self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
        self.trial_info['desired_output'] = response
        self.trial_info['train_mask'] = mask

        return self.trial_info
