 #!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_doom_env, doom_arg_parser
#from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
from baselines.ppo2.policies import CnnPolicy

'''
create 16 envs to train a particular doom level for set no of time steps with specified model(cnn) and learning rate schedule(linear)
'''
def train(doom_lvl, num_timesteps, seed, policy, lrschedule, num_env, adda_lr, adda_batch, training, use_adda):
    print('train() called')
    if policy == 'cnn':
        policy_fn = CnnPolicy

    env = make_doom_env(doom_lvl, num_env, seed) # Make "num_env" environments 

    learn(policy_fn, env, seed, training, use_adda, adda_lr, adda_batch, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)  
    env.close()

def main():
    parser = doom_arg_parser() # Create an argparse.ArgumentParser for doom_atari.py.(contains env_id, seed, num_timesteps args for
                                # train() )
    parser.add_argument('--policy', help='Policy architecture', default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='linear')

    parser.add_argument('--adda_lr', help='ADDA Learning Rate', default=0.0002)
    parser.add_argument('--adda_batch', help='ADDA Batch Size', default=100)
    
    args = parser.parse_args()
    logger.configure('/misc/student/raob/Tests/60e6_Steps/Exp_hg_normal_with_ADDA')
    # Train "num_env" envs, each running "doom_lvl" for "num_timesteps" timesteps with a policy architecture "policy"
    # with a Learning rate schedule "lrschedule"
    train(doom_lvl=args.doom_lvl, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, adda_lr=args.adda_lr, adda_batch=args.adda_batch, num_env=16, training=True, use_adda=True)

if __name__ == '__main__':
    main()
