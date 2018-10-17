import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) # / 255. # Rmoving normalisation here bcus its already done in the doom envs in doom_env.pu
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

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

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #nbatch= nenvs*nsteps
 
        num_actions = ac_space.n
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc) # (320, 84, 84, 1)
        
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        A = tf.placeholder(tf.int32, nbatch) # 16*1 for step_model, 16*20 for train_model
        one_hot_A = tf.one_hot(A, 6)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X, **conv_kwargs)
            Qf = fc(h, 'Qf', ac_space.n) # (nbatch,nactions) --> (16,6)

        with tf.variable_scope("target_model", reuse=reuse):
            h = nature_cnn(X, **conv_kwargs)
            Qf_target = fc(h, 'Qf_target', ac_space.n) # (nbatch,nactions)
            

        '''
        step() called by self.model.step(self.obs) in run(). step_model is used to compute actions for each env, using self.obs (16, 84, 84, 1)
        '''
        a0 = tf.argmax(Qf,axis=1) #  axis = 1, meaning argmax taken along rows. step_model: a0 is (16,)

        Q0 = tf.reduce_max(Qf_target,axis=1)
        
        def step(ob, *_args, **_kwargs):
            a = sess.run(a0, {X:ob})
            return a 
	
        def value(ob, *_args, **_kwargs):
            Q = sess.run(Q0, {X:ob})
            return Q 

        # Target Network update function
        def get_copy_weights_operator(scope_c = "model", scope_v = "target_model"):
            """
            copy weights from scope_c to scope_v

            Args:
            scope_c: target scope
            scope_v: source scope
            """
            assign_ops = []
            # Get parameters of prediction network
            vars_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_c)
            # Get parameters of target network
            vars_v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_v)
            '''
            for i in range(len(vars_v)):
                assign_ops.append(tf.assign(vars_v[i], vars_c[i])) # Update vars_v[i] by assigning vars_c[i] to it
            assign_all = tf.group(*assign_ops)
            '''
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            for var_c,var_v in zip(vars_c, vars_v):
                assign_ops.append(var_v.assign(var_c))
                #print(var_c, var_v)
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            return assign_ops

        self.X = X
        self.A = A
 
        self.Qf = tf.reduce_sum(tf.multiply(Qf, one_hot_A), 1)
        print('$$$$$$$$$$$')
        print(self.Qf.get_shape())
        print('$$$$$$$$$$$')
        #self.Qf = Qf
        self.step = step
        self.value = value # Use this to compute bootstrapped value for ur Q Target
        self.get_copy_weights_operator = get_copy_weights_operator

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
