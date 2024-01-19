import numpy as np
import sys
import argparse
import time
import datetime

# MLAgents imports
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# RL Imports
import gymnasium as gym
from CasaGymEnv import CasaGymEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.evaluation import evaluate_policy
import torch

SEED = 33

def make_env(unity_env, agent_id, seed):
    if unity_env is not None:
        return CasaGymEnv(unity_env, drone_id=agent_id, seed=seed)
    else:
        pass

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Test Unity Environment')
    parser.add_argument('--env', type=str, default='CasaEntrenoRL.x86_64', help='Path to the Unity executable to test')
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
    n_envs = args.n_envs
    n_epochs = args.n_epochs
    device = args.device

    # Load Unity Environment
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(height=1024, width=1024, time_scale=5.0)
    
    unity_env = UnityEnvironment(file_name=env_name, seed=seed, no_graphics=not render, side_channels=[channel])
    unity_env.reset()
    time.sleep(5)
    # Wrap Unity environment in a gym wrapper
    env = CasaGymEnv(unity_env=unity_env, n_drones=n_envs, seed=seed)

    # Define the RL model
    save_dir = './runs/' + datetime.time().strftime("%Y%m%d-%H%M%S") + '/'

    policy_kwargs = dict(activation_fn = torch.nn.ReLU,
                         net_arch=dict(pi=[32,16], vf=[64,64]))
    model = PPO("MlpPolicy", env, n_steps=2000, batch_size=2000, tensorboard_log=save_dir + 'logs/' ,policy_kwargs=policy_kwargs, verbose=1, device=device)
    '''
    policy_kwargs = dict(net_arch=dict(pi=[32,16], qf=[256,256]))
    model = SAC("MlpPolicy", env, train_freq=2000, batch_size=2000, policy_kwargs=policy_kwargs, verbose=1, device=device)
    '''

    # Train the model
    # Custom training loop replicating the one in SB3. See: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py

    iteration = 0
    log_interval = 1

    total_timesteps, callback = model._setup_learn(
        total_timesteps=n_epochs*model.n_steps*n_envs,
        callback=None,
        tb_log_name=datetime.time().strftime("%Y%m%d-%H%M%S"),
        reset_num_timesteps=True,
        progress_bar=True,
    )

    callback.on_training_start(locals(), globals())
    assert model.env is not None
    # Main training loop
    while model.num_timesteps < total_timesteps:    
        print('-----------------------------------')
        print('Epoch: ' + str(iteration))
        assert model._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        model.policy.set_training_mode(False)

        n_steps = 0
        model.rollout_buffer.reset()
        if model.use_sde:
            model.policy.reset_noise(model.env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        
        while n_steps < model.n_steps:
            if model.use_sde and model.sde_sample_freq > 0 and n_steps % model.sde_sample_freq == 0:
                # Sample a new noise matrix
                model.policy.reset_noise(model.env.num_envs)

            # Predict next action
            with torch.no_grad():
                obs_t = obs_as_tensor(model._last_obs, model.device)
                actions, values, log_probs = model.policy(obs_t)
            actions = actions.cpu().numpy()

            #Rescale and perform actions
            clipped_actions = actions

            if model.policy.squash_output:
                clipped_actions = model.policy.unscale_action(clipped_actions)
            else:
                clipped_actions = np.clip(actions, model.action_space.low, model.action_space.high)
            
            # Execute next action
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            model.num_timesteps += model.env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                continue_training = False
                break
            
            model._update_info_buffer(infos)
            n_steps += 1
            # Handle timeouts by boostrapping with value function
            for idx, done in enumerate(dones):
                if(
                    done
                    and infos[idx].get("terminal_observation") is not None
                ):
                    terminal_obs = model.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = model.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += model.gamma * terminal_value

            # Store data in buffer
            model.rollout_buffer.add(
                model._last_obs,
                actions,
                rewards,
                model._last_episode_starts,
                values,
                log_probs,
            )
            model._last_obs = new_obs
            model._last_episode_starts = dones

            if np.sum(dones) == n_envs: # All drones crashed
                env.reset()
        
        with torch.no_grad():
            # Compute value for the last timestep
            values = model.policy.predict_values(obs_as_tensor(new_obs, model.device))
        
        model.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()

        # Update info
        iteration += 1
        model._update_current_progress_remaining(model.num_timesteps, total_timesteps)

        if log_interval is not None and iteration % log_interval == 0:
            assert model.ep_info_buffer is not None
            time_elapsed = max((time.time_ns() - model.start_time) /1e9, sys.float_info.epsilon)
            fps = int((model.num_timesteps - model._num_timesteps_at_start) / time_elapsed)
            model.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(model.ep_info_buffer) > 0 and len(model.ep_info_buffer[0]) > 0:
                model.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer]))
                model.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in model.ep_info_buffer]))
            model.logger.record("time/fps", fps)
            model.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            model.logger.record("time/total timesteps", model.num_timesteps, exclude="tensorboard")
            model.logger.dump(step=model.num_timesteps)

        # Update model
        model.train()

    callback.on_training_end()
    print("Training completed!")

    # Evaluate model
    print('Model evaluation starting...')
    eval_env = CasaGymEnv(unity_env=unity_env, n_drones=1, seed=seed)
    mean_rew, std_rew = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, return_episode_rewards=True)
    print('Model evaluation completed!')
    print('Mean reward: ' + str(mean_rew))
    print('Reward std: ' + str(std_rew))

    # Save the trained model
    saving_path = save_dir + env_name + '_' + 'PPO'
    print('Saving model at: ' + saving_path)
    model.save(saving_path)

if __name__ == '__main__':
    main()
