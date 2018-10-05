 #!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_doom_env, doom_arg_parser
#from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
from baselines.ppo2.policies import CnnPolicy

'''
create 16 envs to train a particular doom level for set no of time steps with specified model(cnn) and learning rate schedule(linear)
'''
def train(doom_lvl, num_timesteps, seed, policy, lrschedule, num_env):
    print('train() called')
    if policy == 'cnn':
        policy_fn = CnnPolicy

    env = make_doom_env(doom_lvl, num_env, seed) # Make "num_env" environments 

    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)  # Learn 
    env.close()

def main():
    parser = doom_arg_parser() # Create an argparse.ArgumentParser for doom_atari.py.(contains env_id, seed, num_timesteps args for
                                # train() )
    parser.add_argument('--policy', help='Policy architecture', default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='linear')
    
    args = parser.parse_args()
    logger.configure('/misc/student/raob/Tests/DummyTests/Exp_1')
    # Train "num_env" envs, each running "env_id" for "num_timesteps" timesteps with a policy architecture "policy"
    # with a Learning rate schedule "lrschedule"
    train(doom_lvl=args.doom_lvl, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=16) # 1) Train 16 envs 

if __name__ == '__main__':
    main()
