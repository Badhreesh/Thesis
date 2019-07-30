"""
Helpers for scripts like run_atari.py.
"""

import os
import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym_vizdoom_control # responsible for calling register_doom_env() in __init__.py, which uses DoomEnv() class 
'''
def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id) # Make doom env instead # Just call gym.make(env_id)
            env.seed(seed + rank) # Same
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank))) # Same
            return wrap_deepmind(env, **wrapper_kwargs) # Needs to go. THis does preprocessing, like stacking frames.
                                                        # Not needed for doom Look into wrapper_kwargs, what it does. Return env.
        return _thunk
    set_global_seeds(seed) # Same
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)]) # NEcessary for creating multiple envs
'''
def make_doom_env(doom_lvl, num_env, seed, start_index=0):
    """
    Create a monitored SubprocVecEnv for Doom. 
    """
    def make_env(rank):
        def _thunk():
            env = gym.make(doom_lvl) # Make doom env instead # Just call gym.make(env_id)
            env.seed(seed + rank) # Same
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank))) # Same
            return env                                           
        return _thunk
    set_global_seeds(seed) # Same
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])        
'''
def make_mujoco_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env
'''
def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
'''
def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    return parser
'''
def doom_arg_parser():
    """
    Create an argparse.ArgumentParser for run_doom.py.
    """
    tasks = (
    "navigation","hg_simple", "battle", "battle_2",
    "hg_normal", "hg_normal_target", "hg_sparse", "hg_very_sparse",
    "hg_normal_health_reward", "hg_normal_many_textures",
    "hg_delay_2", "hg_delay_4", "hg_delay_8", "hg_delay_16", "hg_delay_32",
    "hg_terminal_health_m_1", "hg_terminal_health_m_2", "hg_terminal_health_m_3"
    )
    gym_tasks = ["Doom-" + t + "-v0" for t in tasks]
    seed = int.from_bytes(os.urandom(4), byteorder='big')

    parser = arg_parser()
    parser.add_argument('--doom-lvl', type=str, help='Doom Level', choices=gym_tasks, default='Doom-hg_normal_target-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=seed)
    parser.add_argument('--num-timesteps', type=int, default=int(60000000))
    return parser    
'''
def mujoco_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser
'''
