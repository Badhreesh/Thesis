import os.path as osp
import time
import joblib #  set of tools to provide lightweight pipelining
import numpy as np
import tensorflow as tf
from random import random
from baselines import logger
		
from baselines.common import set_global_seeds, explained_variance
from baselines.common.runners import AbstractEnvRunner
from baselines.common import tf_util
from baselines.common.schedules import LinearSchedule

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse, huber_loss

class Model(object):
    '''
    Used to initialize the step_model(sampling) and train_model(training)
    '''
    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,max_grad_norm=0.5,
     lr=7e-4,alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'): # The epsilion and alpha mentioned here is for RMSProp
    											

        sess = tf_util.make_session()
        nbatch = nenvs*nsteps # 16*20 nsteps set in learn()
        print('nbatch defined and size is ', nbatch)

        A = tf.placeholder(tf.int32, [nbatch])

        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        #print('Calling chosen policy()')

        step_model = policy(sess, ob_space, ac_space, nbatch=nenvs*1, nsteps=1, reuse=False) # nenvs*nsteps = nbatch, 1 = nsteps , model for generating data, Take 1 step for each env
        train_model = policy(sess, ob_space, ac_space, nbatch=nenvs*nsteps, nsteps=nsteps, reuse=True) # model for training using collected data

        Qf_loss = tf.reduce_sum(huber_loss(tf.squeeze(train_model.Qf) - R)) # This is your TD Error

        loss = Qf_loss
        
        params = find_trainable_variables("model") # Returns a list of variable objects
        grads = tf.gradients(loss, params) #Calculate gradients of loss wrt params.Returns a list of tensor of length len(params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params)) # grads is a list of (gradient, variable) pairs 
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads) # Returns an operation that applies the specified gradients.  
        
        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule) # Learning Rate Scheduling

        def train(obs, rewards, actions):
            #advs = rewards - values
            for step in range(len(obs)): # len(obs) = 320
                cur_lr = lr.value()
            #td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            td_map = {train_model.X:obs,R:rewards, LR:cur_lr, train_model.A:actions}
            #if states is not None:
            #    td_map[train_model.S] = states
            #    td_map[train_model.M] = masks
            action_value_loss, _ = sess.run(
                [Qf_loss, _train],
                td_map
            )
            return action_value_loss

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path) #Persist an arbitrary Python object into one file. Here, store the object "ps" to save_path

        def load(load_path):
            loaded_params = joblib.load(load_path) #Reconstruct a Python object from a file persisted with joblib.dump.
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        #self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        self.get_copy_weights_operator = step_model.get_copy_weights_operator
        tf.global_variables_initializer().run(session=sess)

class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps=5, gamma=0.99): # Additional args: num_envs: already there in AbstractEnvRunner????
        super().__init__(env=env, model=model, nsteps=nsteps)
        print('nsteps is',nsteps)
        self.gamma = gamma
    
    '''
    Each time run() executes, 
    '''

    def run(self, greedy_epsilon): # Additional args: greedy_epsilon (This value comes from learn(), before run() is called)
    	
        mb_obs, mb_rewards, mb_actions, mb_dones = [],[],[],[]
        #runtime_actions_loop = 0
        #runtime_rollouts = 0
        #runtime_reward_loop = 0
        
        #t_start_actions = time.time()
        for n in range(self.nsteps):
            '''
            With each step, 16 actions r computed, 1 for each env
            These actions r calculated by taking argmax of Q values from network (Refer CnnPolicy)
            '''
            #t_start = time.time()
            actions = self.model.step(self.obs) # Forward Step, np.shape(actions) = (16,) -> bcus u have 16 envs, as defined in step_model
            #t_end = time.time()
            #runtime = t_end - t_start
            '''
            For each env in a step, replace some actions with random actions based on epsilon greedy
            '''
            for i in range(len(actions)): # 16 actions
            	if random() < greedy_epsilon:
            		actions[i] = self.actionspace.sample()

            '''
            actions is initially defined as a (16,) tensor 
            mb_actions is initially (1,16) shape. Once all steps have been taken, mb_actions will be (20, 16)
            '''             		
            mb_actions.append(actions)
            
            ''' self.obs is initially a (16, 84, 84, 1) tensor of zeros'''
            mb_obs.append(np.copy(self.obs))
            
            '''
            self.dones is initially defined as a (16,) tensor of False 
            mb_dones is initially (1,16) shape. Once all steps have been taken, mb_dones will be (20, 16)
            '''             		
            mb_dones.append(self.dones)
            
            ''' Agent takes action in env to get new obs and reward'''	
            obs, rewards, dones, _ = self.env.step(actions) # Environment Step, _ is for info. Shapes are: (16,84,84,1), (16,) , (16,) (False, if not terminal)-> 16 refers to nenvs
            
            '''Update self.dones'''
            self.dones = dones
            
            '''If a terminal state for an env is reached, set the observation for that env as a tensor of zeros '''
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0 # each self.obs[n] and self.obs[n]*0 is (84, 84, 1)
            self.obs = obs
            mb_rewards.append(rewards) # Last line of for loop
        
        #runtime_actions = runtime_actions/self.nsteps

        
        '''
        Once the for loop is run for 20 steps:
        np.shape(mb_obs) -> (20, 16, 84, 84 ,1)
        np.shape(mb_rewards) -> (20, 16)
        np.shape(mb_actions) -> (20, 16)
        np.shape(mb_dones) -> (20, 16)
        '''    
        
        #t_start = time.time()
        mb_dones.append(self.dones) # shape updated to (21, 16) to account for last self.dones update in loop
        #runtime_actions_loop = time.time() - t_start_actions
        
        #batch of steps to batch of rollouts
        #t_start_rollouts = time.time()
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape) # shape (16*20, 84, 84, 1) 
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0) # shape (16, 20)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0) # shape (16, 20)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0) # shape (16, 21)
        mb_dones = mb_dones[:, 1:] # shape (16, 20) to remove the initial value of self.dones ( The tensor containing all False, defined in runners.py)
        last_values = self.model.value(self.obs).tolist() # This is the only value thats needed ( last Q value w/0 gamma, for every env) shape (16,)
        #runtime_rollouts = time.time() - t_start_rollouts
        
        #discount/bootstrap off value fn
        
        #t_start_rewards = time.time()
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist() # shape (20,) for each n
            dones = dones.tolist() # shape (20,) for each n
            if dones[-1] == 0: # If not terminal state, dones[-1] is False, then 
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1] # Everything except last item. To keep size of rewards the same as nsteps
            else: # If terminal state has been reached, done=1, so discount_with_dones returns only reward
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        
        mb_rewards = mb_rewards.flatten() # Before: (16,20) After: (320,)
        mb_actions = mb_actions.flatten() # Before: (16,20) After: (320,)
        #runtime_reward_loop = time.time()- t_start_rewards
        
        return mb_obs, mb_rewards, mb_actions #, runtime_actions_loop, runtime_rollouts, runtime_reward_loop 

'''
Learn a "policy" for the "env" for "total_timesteps with "lrschedule"
'''
def learn(policy, env, seed,  total_timesteps=int(80e6),lrschedule='linear',nsteps=20,
 max_grad_norm=None, lr=1e-4,  epsilon=0.1, alpha=0.99, gamma=0.99, log_interval=1000, #alpha and epsilon for RMSprop used in Model()
 exploration_fraction=0.8, exploration_final_eps=0.01, target_network_update_freq=10000): # Additional arguments for epsilon greedy

    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs # 16, from train() in run_atari.py -> used by env, which is a parameter in learn()
    ob_space = env.observation_space # (84,84,1)
    ac_space = env.action_space # Discrete(6)

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs*nsteps #

    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
    							 initial_p=1.0, final_p=exploration_final_eps)

    model.get_copy_weights_operator()
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1): # For 100k steps, loop is from 1 to 313 -> runs 312 updates     
        update_eps = exploration.value(update*nbatch) 
        obs, rewards, actions = runner.run(update_eps) # Performs 1 update step. For 16 envs nd nstep = 20, shapes r: (16*20,84,84,1), (320,), (320,) resp
        action_value_loss = model.train(obs, rewards, actions) # Computes TD Error
        nseconds = time.time()-tstart

        fps = int((update*nbatch)/nseconds)
        # Update target network every 10k steps
        if update % 31 == 0:
            #print('Target Network Updated')
            model.get_copy_weights_operator()
        if update % log_interval == 0 or update == 1:

            logger.record_tabular("epsilon", update_eps)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("action_value_loss", float(action_value_loss))
            logger.record_tabular("time_elapsed", nseconds)
            logger.dump_tabular()
    #env.close()
    return model
    
