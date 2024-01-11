import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from stable_baselines3.common.env_checker import check_env
from CasaGymEnv import CasaGymEnv

SEED = 33

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test Unity Environment')
    parser.add_argument('--env', default='CasaEntrenoRL.x86_64', help='Path to the Unity executable to test')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--render', type=bool, default=False, help='Display the Unity environment')

    # Get arguments
    args = parser.parse_args()
    env_name = args.env
    seed = args.seed
    render = args.render

    # Load Unity environment
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(height=1024, width=1024)
    unity_env = UnityEnvironment(file_name=env_name, seed=seed, no_graphics=not render, side_channels=[])
    # Wrap Unity environment in a gym wrapper
    env = CasaGymEnv(unity_env=unity_env, seed=seed)
    # Check environment
    #check_env(env)
    
    obs, _ = env.reset()
    print(obs.shape)
    
    for i in range(2000):
        env.reset()
        done = False
        t = 0
        while not done or t<100:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            t+=1
            print(action)
            print(reward)
            print(t)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    

if __name__ == '__main__':
    main()