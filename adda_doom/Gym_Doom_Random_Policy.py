import gym_vizdoom_control
import numpy as np
import random
import gym
from scipy.misc import imsave


#Source_Env = 'Doom-hg_normal-v0'
#Target_Env = 'Doom-hg_normal_target-v0'



def gym_env(num_episodes, num_timesteps, doom_level):
    print('gym_env() called')
    imgs = []
    env = gym.make(doom_level)
    
    for _ in range(num_episodes):
        env.reset()

        for step in range(num_timesteps):
            env.render()
            action = env.action_space.sample()
            obs, rwds, done, _ = env.step(action)
            imgs.append(obs)
            #print('Step: {}, Reward: {}'.format(step, rwds))
            if done:
                print('Episode Done')
                break
        
    
    env.close()

    imgs = np.array(imgs)
    imgs.resize(len(imgs), 84, 84, 1)
    print('Dimensions after reshape:', np.shape(imgs))
    np.save('many_textures_dataset', imgs)

def doom_arg_parser():
	import argparse
	tasks = ('hg_normal', 'hg_normal_target', 'hg_normal_many_textures')
	gym_tasks = ["Doom-" + t + "-v0" for t in tasks]
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--num_episodes', help='How many episodes you want the agent to run', type=int, default=100)
	parser.add_argument('--num_steps', help='How many steps per episode', type=int, default=100)
	parser.add_argument('--env', help='Doom Level', type=str, default='Doom-hg_normal_many_textures-v0')
	return parser

def main():

    parser = doom_arg_parser()
    args = parser.parse_args()
    gym_env(num_episodes=args.num_episodes, num_timesteps=args.num_steps, doom_level=args.env)


if __name__ == '__main__':
	main()

dataset = np.load('many_textures_dataset.npy')
formatted = np.squeeze(dataset[5])
imsave('many_textures.png', formatted)

