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
    parser.add_argument('--render', type=bool, default=True, help='Display the Unity environment')
    parser.add_argument('--n_envs', type=int, default=1, help='Number of parallel environments for training')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of learning epochs')
    parser.add_argument('--device', type=str, default='auto', help='Device for running training (cpu, cuda or auto)')
    # Get arguments
    args = parser.parse_args()
    env_name = args.env
    seed = args.seed
    render = args.render
    if not render:
        n_envs = args.n_envs
    else:
        n_envs = 1
    n_epochs = args.n_epochs
    device=args.device

    # Load Unity Environment
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(height=1024, width=1024)
    unity_env = UnityEnvironment(file_name=env_name, seed=seed, no_graphics=not render, side_channels=[])
    unity_env.reset()
    time.sleep(5)
    # Wrap Unity environment in a gym wrapper
    env = CasaGymEnv(unity_env=unity_env, seed=seed)
    vec_env = make_vec_env(lambda: make_env(unity_env, seed), n_envs=n_envs)

    # Define the RL model
    policy_kwargs = dict(activation_fn = torch.nn.ReLU,
                         net_arch=dict(pi=[32,16], vf=[32,16]))
    model = PPO("MlpPolicy", vec_env, n_steps=2000, batch_size=100, policy_kwargs=policy_kwargs, verbose=1, device=device)

    # Train the model
    model.learn(total_timesteps=n_epochs*model.n_steps*vec_env.num_envs, progress_bar=True)
    print("Training completed!")

    # Evaluate model
    print('Model evaluation starting...')
    mean_rew, std_rew = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print('Model evaluation completed!')
    print('Mean reward: ' + str(mean_rew))
    print('Reward std: ' + str(std_rew))

    # Save the trained model
    saving_path = env_name + '_' + 'PPO'
    print('Saving model at: ' + saving_path)
    model.save(saving_path)

if __name__ == '__main__':
    main()
