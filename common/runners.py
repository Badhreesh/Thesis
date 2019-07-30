import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = env.num_envs 
        self.num_envs = env.num_envs
        self.actionspace = env.action_space # My addition
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape # (16*20,) + (84,84,1) = (16*20,84,84,1)
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name) # ((16), (84,84,1))
        #print(type(self.env))
        #import sys; sys.exit()
        self.obs[:] = env.reset() # Start each env. Depends on nenv, in our case, 16
        self.nsteps = nsteps
        #self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod 
    def run(self):
        raise NotImplementedError
