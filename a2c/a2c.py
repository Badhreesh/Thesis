import os.path as osp
import time
import joblib #  set of tools to provide lightweight pipelining
import numpy as np
import tensorflow as tf
from random import random
from baselines import logger
import matplotlib.pyplot as plt
		
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

        #A = tf.placeholder(tf.int32, [nbatch])

        R = tf.placeholder(tf.float32, [nbatch]) # This is your TD Target
        LR = tf.placeholder(tf.float32, [])


        step_model = policy(sess, ob_space, ac_space, nbatch=nenvs*1, nsteps=1, reuse=False) # nbatch = nenvs*nsteps, model for generating data, Take 1 step for each env
        train_model = policy(sess, ob_space, ac_space, nbatch=nenvs*nsteps, nsteps=nsteps, reuse=True) # model for training using collected data
        print('Qf:', train_model.Qf.get_shape())
        print('R:', R.get_shape())
        
        loss = tf.reduce_sum(huber_loss(train_model.Qf - R)) # This is your TD Error (Prediction - TD Target)
        # Add both mapping and adversarial losses here. They will be the same
        params = find_trainable_variables("model") # Returns a list of variable objects
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(params)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        grads = tf.gradients(loss, params) #Calculate gradients of loss wrt params.Returns a list of sum(d_loss/d_param) for each param in params
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_vars = list(zip(grads, params)) # grads_and_vars is a list of (gradient, variable) pairs 
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads_and_vars) # Returns an operation that applies the specified gradients.  
        
        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule) # Learning Rate Scheduling

        def train(obs, rewards, actions):
            for step in range(len(obs)): # len(obs) = 320
                cur_lr = lr.value()

            td_map = {train_model.X:obs,R:rewards, LR:cur_lr, train_model.A:actions}

            action_value_loss, _ = sess.run(
                [loss, _train],
                td_map
            )
            # run DA losses in a session here. Start with running them after every update step. Later, condsider running after every 10 steps
            return action_value_loss, cur_lr

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
        
        saver = tf.train.Saver()
        #part_vars_names = ('model/c1/b','model/c1/w','model/c2/b','model/c2/w','model/c3/b','model/c3/w','model/fc1/b','model/fc1/w')
        part_vars_names = ('model/c1/b','model/c1/w','model/c2/b','model/c2/w','model/c3/b','model/c3/w')
        part_vars = [var for var in params if var.name[:-2] in part_vars_names]
        print(part_vars)
        saver_adda = tf.train.Saver(part_vars)
        
        def save_model():
            saver.save(sess, './hg_normal_many_textures/hg_normal_many_textures_model')   
        
        def load_model(snapshot, seed, adda_mode=False):

            # Load the saved parameters of the graph
            if snapshot == 0:
                saver.restore(sess, './hg_normal/hg_normal_model')
                #saver.restore(sess, './hg_normal_many_textures/hg_normal_many_textures_model')

            if snapshot > 0 and adda_mode:
            	#variables_can_be_restored = list(set(tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)).intersection(tf.train.list_variables('./adda_doom_DA/adda_doom_DA-10')))
            	#print(variables_can_be_restored)
            	
            	saver_adda.restore(sess, './adda_doom_DA/hg_normal no fc snapshots/Seed {}/adda_doom_DA-{}'.format(seed, snapshot))
            	#saver_adda.restore(sess, './adda_doom_DA/hg_normal_snapshots for different batch sizes/{}/adda_doom_DA-{}'.format(bsize, snapshot))
            #saver.restore(sess, './hg_normal_many_textures/hg_normal_many_textures_model')
            #print(sess.run('model/c1/b:0'))

        copy_op = step_model.get_copy_weights_operator()
        
        def update_target():
            sess.run(copy_op)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.save = save
        self.load = load
        self.save_model = save_model
        self.load_model = load_model
        self.update_target = update_target
        tf.global_variables_initializer().run(session=sess)
        
        #var = [var for var in tf.global_variables() if var.op.name=="model/Qf/b"][0]
        #var_tar = [var for var in tf.global_variables() if var.op.name=="target_model/Qf_target/b"][0]

        def print_var():
            print(sess.run(var))
            print(sess.run(var_tar))
            
        self.print_var = print_var
            

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

        for n in range(self.nsteps):
            '''
            With each step, 16 actions r computed, 1 for each env
            These actions r calculated by taking argmax of Q values from prediction network (Refer CnnPolicy)
            '''

            actions = self.model.step(self.obs) # Forward Step, np.shape(actions) = (16,) -> bcus u have 16 envs, as defined in step_model
            #print(actions)
            '''
            For some envs, replace chosen action with random actions based on epsilon greedy
            '''
            for i in range(len(actions)): # 16 actions
            	if random() < greedy_epsilon:
            		actions[i] = self.actionspace.sample()
            #print(actions)
            #import sys; sys.exit()
            '''
            actions is initially defined as a (16,) tensor 
            mb_actions is initially (1,16) shape. Once all steps have been taken, mb_actions will be (20, 16)
            '''             		
            mb_actions.append(actions)
            
            ''' self.obs is initially a (16, 84, 84, 1) tensor of zeros. Once env.reset() is done for each env (in runners.py), u will get a state'''
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
            
            '''If a terminal state for an env is reached, set the observation for that env as a tensor of zeros 
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0 # each self.obs[n] and self.obs[n]*0 is (84, 84, 1) ########## New State(obs) should be zeros if terminal, not previous state(self.obs)... ########
                    obs[n] = obs[n]*0
            '''
            self.obs = obs 
            mb_rewards.append(rewards) # Last line of for loop
        
        '''
        Once the for loop is run for 20 steps:
        np.shape(mb_obs) -> (20, 16, 84, 84 ,1)
        np.shape(mb_rewards) -> (20, 16)
        np.shape(mb_actions) -> (20, 16)
        np.shape(mb_dones) -> (20, 16)
        '''    
        
        mb_dones.append(self.dones) # shape updated to (21, 16) to account for last self.dones update in loop
        
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape) # shape (16*20, 84, 84, 1) 
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0) # shape (16, 20)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0) # shape (16, 20)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0) # shape (16, 21)
        mb_dones = mb_dones[:, 1:] # shape (16, 20) to remove the initial value of self.dones ( The tensor containing all False, defined in runners.py)
        last_values = self.model.value(self.obs).tolist() # last Q value w/0 gamma, for every env, from your target network with shape (16,)
        
        #discount/bootstrap off value fn
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
        
        return mb_obs, mb_rewards, mb_actions
    
    # Funcn to return optimal actions from trained model
    def runner_eval(self, num_episodes):
        all_episodes_rwd = []
        #episode_health = []
        for i in range(num_episodes):
            episode_rwd = []
            #total_health = []
            done = False
           
            while not done:
                
                action = self.model.step(self.obs) # (1,)
                
                obs, reward, done, info = self.env.step(action) # obs: (1, 84, 84, 1), rwd: (1,), done: (1,)->[False] if not terminal
                #print(type(done))
                
                #total_health.append(info[0]['evaluation_reward'])
                episode_rwd.append(reward)

                if done:
                    #print(done)
                    #print('eps done')
                    break
                else:
                    #print(done)	
                    #print('new obs')
                    self.obs = obs
            #print(np.shape(episode_rwd)) # (x,1) -> example: [[1],[1],[1]] is (3,1) 1 is num_envs
            #episode_health.append(np.sum(total_health))
            all_episodes_rwd.append(np.sum(episode_rwd))

        mean_reward = np.mean(np.array(all_episodes_rwd))
        #mean_health = np.mean(np.array(episode_health))

        #std_reward = np.std(np.array(episode_rwd))
        #std_health = np.std(np.array(episode_health))
        #print('Reward after episode ends is {}'.format(episode_rwd))
        #print('Mean Reward is {}'.format(mean_reward))
        #print('Standard Deviation of Reward is {}'.format(std_reward))
        
        #print('\n')
        #print('Health after episode ends is {}'.format(episode_health))
        #print('Mean Health is {}'.format(mean_health))
        #print('Standard Deviation of Health is {}'.format(std_health))
        return mean_reward#, mean_health ################################### Mean rwd of episodes run for one snapshot

    def runner_eval_parallel(self, num_episodes, num_envs):

        all_episodes_rwd = []
        env_rwd = np.zeros([num_envs])

        while len(all_episodes_rwd) < num_episodes:
            
            actions = self.model.step(self.obs) # (2,)
  
            obs, rewards, dones, infos = self.env.step(actions) # obs: (2, 84, 84, 1), rwd: (2,), done: (2,)->[False, False] if not terminal

            for i in range(num_envs):
                env_rwd[i] += rewards[i]
                
                if dones[i]:
                    all_episodes_rwd.append(env_rwd[i])
                    env_rwd[i] = 0
                
            self.obs = obs

        #print('All Episodes Reward', all_episodes_rwd)

        mean_reward = np.mean(all_episodes_rwd)

        return mean_reward ################################### Mean rwd of episodes run for one snapshot

'''
Learn a "policy" for the "env" for "total_timesteps with "lrschedule"
Update on 20/2/19; Added training boolean to switch between training and evaluation. Set to False in run_doom.py to perform evaluation
'''
def learn(policy, env, seed,training,  total_timesteps=int(80e6),lrschedule='linear',nsteps=20,
 max_grad_norm=None, lr=7e-4,  epsilon=0.1, alpha=0.99, gamma=0.99, log_interval=1000, #alpha and epsilon for RMSprop used in Model()
 exploration_fraction=0.8, exploration_final_eps=0.001, target_network_update_freq=10000): # Additional arguments for epsilon greedy

    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs # 16, from train() in run_doom.py -> used by env, which is a parameter in learn()
    print('Num Envs {}'.format(nenvs))
    ob_space = env.observation_space # (84,84,1)
    ac_space = env.action_space # Discrete(6)

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    
    if training:
        nbatch = nenvs*nsteps # 16*20

        exploration = LinearSchedule(schedule_timesteps=50000000 , initial_p=1.0, final_p=exploration_final_eps) # U want to hit lowest epsilon value in 50e6 steps
        model.update_target()
        tstart = time.time()
        for update in range(1, total_timesteps//nbatch+1): # For 100k steps, loop is from 1 to 313 -> runs 312 updates     
            update_eps = exploration.value(update*nbatch) 
            # Performs 1 update step (320 total_timesteps). For 16 envs nd nstep = 20, shapes r: (16*20,84,84,1), (320,), (320,) resp
            obs, rewards, actions = runner.run(update_eps) 
            action_value_loss, cur_lr = model.train(obs, rewards, actions) # Computes TD Error
            nseconds = time.time()-tstart

            fps = int((update*nbatch)/nseconds)
        
            # Save model every 2e6 steps (each iteration of loop makes 320 steps. To make target network update ard 10k steps, 320*6250 = 2e6 steps. So update % 6250)
            if update % 6250 == 0:
                model.save_model()
                #print('Model Saved')
            # Update target network every 10k steps 
            if update % 31 == 0:
                #print('Target Network Updated')
                model.update_target()
            if update % log_interval == 0 or update == 1:
                logger.record_tabular("learning rate", cur_lr)
                logger.record_tabular("epsilon", update_eps)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update*nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("action_value_loss", float(action_value_loss))
                logger.record_tabular("time_elapsed", nseconds)
                logger.dump_tabular()
    else:

        snapshots = 2
        seeds = 3
        snapshot_rewards = np.zeros(shape=(seeds, snapshots+1))
        seed_list = ['Seed 0', 'Seed 1', 'Seed 2']

        for seed in range(seeds):
            snapshot_reward = []
            #snapshot_health = []
            print('################### Seed {}!!! ###################'.format(seed))
            tstart = time.time()
            for snapshot in range(snapshots+1):
                model.load_model(snapshot, seed, adda_mode=True)
                #print('##################################################')
                print('Evaluating snapshot {}!!!'.format(snapshot))
                reward = runner.runner_eval_parallel(num_episodes=1000, num_envs=nenvs)
                #reward = runner.runner_eval(num_episodes=1000)
                snapshot_reward.append(reward)
                #snapshot_health.append(health)
                
            print('Mean Reward of every snapshot: ', snapshot_reward)
            snapshot_rewards[seed] = snapshot_reward
            #print('Mean Health of every snapshot: ', snapshot_health)
            print('##################################################')
            nseconds = time.time() - tstart
            print('\n')
            print('Time Elapsed:', nseconds)
            epochs = np.arange(0, snapshots+1)
            
            #plt.figure()
            plt.plot(epochs, np.array(snapshot_reward), '-o', label = seed_list[seed])
            plt.legend(loc = 'lower right')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Reward after 1000 episodes')
            plt.savefig('Mean Reward.png')
        
        # Create plot of mean reward of all seed values with std devns
        mean = []
        std = []
        for x, y, z in zip(snapshot_rewards[0],snapshot_rewards[1],snapshot_rewards[2]):
            mean_val = np.mean([x,y,z])
            std_val = np.std([x,y,z])
            mean.append(mean_val)
            std.append(std_val) 
        epochs = np.arange(0, snapshots+1)
        lower = np.array(mean)-np.array(std)
        upper = np.array(mean)+np.array(std)
        plt.figure()
        plt.plot(epochs, np.array(mean), 'k-')
        plt.fill_between(epochs, lower, upper, alpha = 0.75)
        #plt.title('Mean Reward of 3 seed values')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Reward after 1000 episodes')
        plt.savefig('Mean Reward of 3 seeds with std.png')


