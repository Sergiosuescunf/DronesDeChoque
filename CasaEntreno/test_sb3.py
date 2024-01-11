import numpy as np
import sys
import argparse
import time

# MLAgents imports
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# RL Imports
import gymnasium as gym
from CasaGymEnv import CasaGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import torch

SEED = 33

def make_env(unity_env, seed):
    return CasaGymEnv(unity_env, seed)

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Test Unity Environment')
    parser.add_argument('--env', default='CasaEntrenoRL.x86_64', help='Path to the Unity executable to test')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device for running training (cpu, cuda or auto)')
    parser.add_argument('--n_test_episodes', type=str, default=10, help='Number of episodes for testing')
    parser.add_argument('--saved_model_path', type=str, required=True, help='Path to a trained model')
    # Get arguments
    args = parser.parse_args()
    env_name = args.env
    seed = args.seed
    render = True
    n_envs = 1
    n_test_episodes = args.n_test_episodes
    model_path = args.saved_model_path

    # Load Unity Environment
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(height=1024, width=1024)
    unity_env = UnityEnvironment(file_name=env_name, seed=seed, no_graphics=not render, side_channels=[])
    unity_env.reset()
    time.sleep(5)
    # Wrap Unity environment in a gym wrapper
    # env = CasaGymEnv(unity_env=unity_env, seed=seed)
    vec_env = make_vec_env(lambda: make_env(unity_env, seed), n_envs=n_envs)

    # Load model
    model = PPO.load(model_path, vec_env)

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=n_test_episodes)
    print('Mean reward: ' + str(mean_reward))
    print('Std reward: ' + str(std_reward))

    vec_env.close()

    


if __name__ == '__main__':
    main()
