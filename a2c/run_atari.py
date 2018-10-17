#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env):
    print('train() called')
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4) # Make "num_env" environments 

    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)  # Learn 
    env.close()

def main():
    parser = atari_arg_parser() # Create an argparse.ArgumentParser for run_atari.py.(contains env_id, seed, num_timesteps args for
                                # train() )
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    # Maybe add argument for epsilon greedy here???? -- No, doesnt make sense here!!!
    args = parser.parse_args()
    logger.configure()
    # Train "num_env" envs, each running "env_id" for "num_timesteps" timesteps with a policy architecture "policy"
    # with a Learning rate schedule "lrschedule"
    train(env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=1) # 1) Train 16 envs 

if __name__ == '__main__':
    main()
