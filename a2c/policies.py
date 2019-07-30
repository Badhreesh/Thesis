import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype

from tensorflow.contrib import slim
from contextlib import ExitStack

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255. 
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), # 8x8 filter size is common on the very 1st conv layer, looking at the input image
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

#################################### Define discriminitor network here
def discriminator(net, layers):#, scope='adversary'):
        
    with ExitStack() as stack:
        #stack.enter_context(tf.variable_scope(scope))
        stack.enter_context(
            slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(2.5e-5)))
        for dim in layers:
            net = slim.fully_connected(net, dim)
        net = slim.fully_connected(net, 2, activation_fn=None)
    return net

'''
class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        self.pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value
'''
class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, adda_batch, seed, nbatch, nsteps, reuse=False, dueling=True, use_adda=False, **conv_kwargs): #nbatch= nenvs*nsteps
 
        num_actions = ac_space.n
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc) # (320, 84, 84, 1)
        
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        A = tf.placeholder(tf.int32, nbatch) # 16*1 for step_model, 16*20 for train_model

        one_hot_A = tf.one_hot(A, num_actions)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X, **conv_kwargs)
            Qf, a0 = self.get_q_and_a(h, one_hot_A, num_actions, dueling=dueling)

        one_hot_normal_net_a = tf.one_hot(a0, num_actions) # Target N/W uses best action computed from DQN (action with max Q value) to compute the Q_target
        with tf.variable_scope("target_model", reuse=reuse):
            h = nature_cnn(X, **conv_kwargs)
            q_target = self.get_ddqn_q_target(h, one_hot_normal_net_a, num_actions, dueling=dueling)
        
        ################################################### ADDA ###################################################
        if use_adda:
        
            ################## create dataset ##################
            # new placeholder (imgs) for dataset images. Add dataset
            dataset_imgs = tf.placeholder(tf.uint8, shape=[None, 84, 84, 1])
        
            print('Datasets loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('ADDA SEED: ',seed)
            # Make dataset object using placeholder
            source_dataset = tf.data.Dataset.from_tensor_slices(dataset_imgs).shuffle(buffer_size=100000, seed=seed).batch(adda_batch).repeat()
            target_dataset = tf.data.Dataset.from_tensor_slices(dataset_imgs).shuffle(buffer_size=100000, seed=seed).batch(adda_batch).repeat()

            # Create generic iterator of correct shape and type
            src_iter = tf.data.Iterator.from_structure(source_dataset.output_types, source_dataset.output_shapes)
            tgt_iter = tf.data.Iterator.from_structure(target_dataset.output_types, target_dataset.output_shapes)
    
            src_imgs = src_iter.get_next()
            tgt_imgs = tgt_iter.get_next()

            # Create initialisation operations
            source_iter_op = src_iter.make_initializer(source_dataset)
            target_iter_op = tgt_iter.make_initializer(target_dataset)
            ####################################################

            with tf.variable_scope("model", reuse=True):
                h_src = nature_cnn(src_imgs, **conv_kwargs)
                h_tgt = nature_cnn(tgt_imgs, **conv_kwargs)
        
            #concat o/ps
            # Step 7: Concat model o/p's to form feature i/p to discriminator()
            adversary_ft = tf.concat([h_src, h_tgt], 0) 

            #create labels
            source_adversary_label = tf.zeros([adda_batch], tf.int32) 
            target_adversary_label = tf.ones([adda_batch], tf.int32)
            adversary_labels = tf.concat([source_adversary_label, target_adversary_label], 0)
        
            with tf.variable_scope('adversary', reuse=reuse):
                adversary_logits = discriminator(net=adversary_ft, layers=[512, 512])
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',adversary_logits.get_shape())
            

            self.dataset_imgs = dataset_imgs
            self.adversary_logits = adversary_logits
            self.adversary_labels = adversary_labels
            self.source_iter_op = source_iter_op
            self.target_iter_op = target_iter_op
        ###############################################################################################################





        '''
        step() called by self.model.step(self.obs) in run(). step_model is used to compute actions for each env, using self.obs (16, 84, 84, 1)
        '''
        
        
        def step(ob, *_args, **_kwargs): # Uses DQN N/W to select best action to take for the next state 
            a = sess.run(a0, {X:ob})
            return a 
	
        def value(ob, *_args, **_kwargs): # Uses Target N/W to calculate bootstraped target Q Value of DQN's best action at the next state
            Q = sess.run(q_target, {X:ob})
            return Q 

        self.X = X
        self.A = A
        

        self.Qf = Qf
        # Should have lists of variables for mapping and adversarial losses (self.adversay_vars and self.target_vars) -> need them for when u minimize the losses in a2c.py

        self.step = step
        self.value = value # Use this to compute bootstrapped value for ur Q Target
    
    @staticmethod
    # Returns the action and the Q value for each action
    def get_q_and_a(fc_layer, one_hot_A, num_actions, dueling):
        if not dueling:
            q = fc(fc_layer, 'Qf', num_actions) # (nbatch,nactions) --> (16,6)
            
            a = tf.argmax(q, axis=1) #  axis = 1, meaning argmax taken along rows. step_model: a is (16,). 
            
            q_a = tf.reduce_sum(tf.multiply(q, one_hot_A), 1)
            
            return q_a, a
        else:
            h = tf.nn.relu(fc(fc_layer, 'fc2_Q_adv', 512))
            q_adv = fc(h, 'Q_adv', num_actions)
            
            a = tf.argmax(q_adv, axis=1)
            
            q_adv = q_adv - tf.reduce_mean(q_adv, reduction_indices=1, keep_dims=True)
            q_adv_a = tf.reduce_sum(q_adv * one_hot_A, 1)
            
            h = tf.nn.relu(fc(fc_layer, 'fc2_V', 512))
            v = fc(h, 'V', 1)
            v = tf.reshape(v, [-1])
            
            q_a = q_adv_a + v
            return q_a, a
          
    @staticmethod
    def get_ddqn_q_target(fc_layer, one_hot_normal_net_action, num_actions, dueling):
        if not dueling:
            q = fc(fc_layer, 'Qf', num_actions) # (nbatch,nactions) --> (16,6)
            
            q_target = tf.reduce_sum(tf.multiply(q, one_hot_normal_net_action), 1) # In Vanilla DQN, q_target = tf.reduce_max(q, axis=1) 
            # q_target has max q values for each env, bcus one_hot_normal_net_action is built from actions that have max q values from
            # the DQN 
            return q_target # This is only the bootstrapped part of the TD Target 
        else:
            h = tf.nn.relu(fc(fc_layer, 'fc2_Q_adv', 512))
            q_adv = fc(h, 'Q_adv', num_actions)
            
            q_adv = q_adv - tf.reduce_mean(q_adv, reduction_indices=1, keep_dims=True)
            q_adv_target = tf.reduce_sum(q_adv * one_hot_normal_net_action, 1)
            
            h = tf.nn.relu(fc(fc_layer, 'fc2_V', 512))
            v = fc(h, 'V', 1)
            v = tf.reshape(v, [-1])
            
            q_target = q_adv_target + v
            return q_target
          
          
    @staticmethod      
    def get_copy_weights_operator():
        assign_ops = []
        # Get parameters of prediction network
        vars_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")
        # Get parameters of target network
        vars_v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_model")
        
        # for i in range(len(vars_v)):
        #     assign_ops.append(tf.assign(vars_v[i], vars_c[i])) # Update vars_v[i] by assigning vars_c[i] to it
        # assign_all = tf.group(*assign_ops)
        
        for var_c, var_v in zip(vars_c, vars_v):
            assign_ops.append(var_v.assign(var_c))
        
        return assign_ops

'''
class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        self.pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            flatten = tf.layers.flatten
            pi_h1 = activ(fc(flatten(X), 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(flatten(X), 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)


        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
'''
