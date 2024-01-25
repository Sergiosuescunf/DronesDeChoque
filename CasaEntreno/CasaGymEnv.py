import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Any, Dict
import numpy as np
from numpy.typing import NDArray
import time
from mlagents_envs.environment import UnityEnvironment

from grid import Grid


class CasaGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, unity_env, n_drones=1, seed=33):
        super(CasaGymEnv, self).__init__()
        self.seed = seed
        self.n_drones = n_drones
        # Obtain list with used drone ids
        assert self.n_drones >= 1, "Number of drones must be greater than 0"
        self.drone_ids = list(range(n_drones))
        
        # Save Unity environment
        self.unity_env = unity_env
        # Init Unity environment
        self.unity_env.reset()

        # Get info from Unity environment
        self.behavior_name = list(self.unity_env.behavior_specs)[0]
        self.behavior_spec = self.unity_env.behavior_specs[self.behavior_name]
        self.action = self.behavior_spec.action_spec.random_action(n_agents=1) # Init action variable

        if self.behavior_spec.action_spec.continuous_size > 0:
            self.action_space = spaces.Box(low=np.array([-1., 0.]), high=np.array([1., 1.]), shape=(self.behavior_spec.action_spec.continuous_size-1,), dtype=np.float32)
        elif self.behavior_spec.action_spec.discrete_size > 0:
            self.action_space = spaces.Discrete(self.behavior_spec.action_spec.discrete_size)        
        
        self.n_obs = 78
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(1, self.n_obs))

        # Define different zones
        self.zones = []
        self.zones.append([-13, -6, -2, 0])
        self.zones.append([-13, -14, -2, -6])
        self.zones.append([-2, -14, 15, 0])
        self.zones.append([-13, 0, 15, 13])

        # Create utility variables for reward computation
        self.grids = []
        self.explored_zones = []
        for id in self.drone_ids:
            self.grids.append(Grid(-14, -16, 16, 14, 10))
            self.explored_zones.append([])

        self.max_distance = 0.55 #TODO: Change this to a parameter

        # Create utility variables for info
        self.steps = 0
        self.action_buffer = []
        self.obs_buffer = []
        self.rew_buffer = []
        self.last_actions = np.zeros((len(self.drone_ids), 3), dtype=np.float32)
        self.last_observations = np.zeros((self.n_drones, self.n_obs), dtype=np.float32)
        self.last_rewards = np.zeros(self.n_drones, dtype=np.float32)

        self.last_actions[:, 2] = 0.1 # Set all drones to be alive by default

        # Create utility variables for termination
        self.states = np.zeros((self.n_drones, 4), dtype=np.float32)
        self.max_steps = 2000 #TODO: Change this to a parameter
        self.dones = np.array([False for _ in range(len(self.drone_ids))], dtype=np.bool)
        self.crashed = np.array([False for _ in range(len(self.drone_ids))], dtype=np.bool)


    def reset(self, seed=33):
        # Reset Unity environment
        self.unity_env.reset()
        self.dones[:] = False
        self.crashed[:] = False
        self.states = np.zeros((self.n_drones, 4), dtype=np.float32)
        # Reset utility variables for reward computation
        for id in self.drone_ids:
            self.grids[id].clean_grid()
            self.explored_zones[id] = []
        # Reset utility variables for info
        self.steps = 0
        self.action_buffer = []
        self.obs_buffer = []
        self.rew_buffer = []
        # Reset last X buffers
        self.last_actions = np.zeros((len(self.drone_ids), 3), dtype=np.float32)
        self.last_actions[:, 2] = 0.1
        self.last_observations = np.zeros((self.n_drones, self.n_obs), dtype=np.float32)
        self.last_rewards = np.zeros(self.n_drones, dtype=np.float32)
        self.action = self.behavior_spec.action_spec.random_action(n_agents=1)
        # Return observation and info
        for id in self.drone_ids:
            self.last_observations[id] = self._get_obs_id(id)
        infos = self._get_info()

        return self.last_observations[id], infos[id]

    def step(self, dummy_action: NDArray[np.float32]) -> Tuple[NDArray[np.float32], float, bool, Dict[str, Any]]:
        '''
        This methods assumes you already called set_action_id() for each drone, so the action is already set in the Unity environment
        if you want to use this method, you should call set_action_id() for each drone before calling step()
        dummy_action is not used, it is only here to comply with the gym interface

        It returns the observation, reward, done and info for the first drone to comply with the gym interface
        nonetheless, the observations, rewards, dones and infos for all drones are updated and can be accessed through the class variables
        '''

        # Step all active drones in Unity environment     
        self.unity_env.step()              

        # Get observations from Unity environment
        for id in self.drone_ids:
            self.last_observations[id] = self._get_obs_id(id)
            self.dones[id] = self._check_termination(id)
            self.last_rewards[id] = self._compute_reward(id)

        # Get info
        infos = self._get_info(self.last_observations)

        return self.last_observations, self.last_rewards, self.dones, infos
    
    def set_action_id(self, id, action):
        # Add drone state 0.2 = Alive, 0.1 = Dead
        if self.crashed[id]:
            action = np.append(action, [0.1])
        else: 
            action = np.append(action, [0.2])
        
        self.last_actions[id] = action        
        # Pass action to Unity environment
        self.action.add_continuous(action.reshape((1,3)))
        self.unity_env.set_action_for_agent(self.behavior_name, id, self.action)
        

    def render(self, mode='human') -> NDArray[np.float32]:
        pass

    def close(self):
        self.unity_env.close()
    
    def compute_reward(self):
        # Weights for rewards and penalties
        w_forward = 1.0
        w_explore = 100.0
        w_grid = 1.0
        w_distance = 20.0
        w_crashed = 50.0

        # Init rewards vector
        rewards = np.zeros(self.n_drones, dtype=np.float32)
        for id in self.drone_ids:

            # Reward for moving forward
            forward_rew = w_forward * self.last_actions[id][1]

            # Update explored zones
            zone = self._compute_current_zone(self.states[id][0], self.states[id][2]) # PosX, PosZ
            if zone not in self.explored_zones[id]:
                self.explored_zones[id].append(zone)
            
            # Compute exploration reward
            explore_rew = w_explore * (len(self.explored_zones[id]) - 1)

            # Compute grid reward
            self.grids[id].update(self.states[id][0], self.states[id][2])
            grid_rew = w_grid * (self.grids[id].puntuation() / self.grids[id].cell_count())

            # Get front distance to obstacles
            dist_center = self.decision_steps[id][0][8][1]

            # Compute distance penalty
            dist_penalty = self._distance_penalty(dist_center, w_proximity_penalty=w_distance)
            crash_penalty = w_crashed * self.crashed[id]

            # Compute reward
            reward = forward_rew + explore_rew + grid_rew
            # Compute penalty
            penalty = dist_penalty + crash_penalty
            # Compute final reward
            rewards[id] = reward - penalty

        return rewards
    
    def _compute_reward(self, id=0):
        # Weights for rewards and penalties
        w_forward = 10.0
        w_explore = 200.0
        w_grid = 1.0
        w_distance = 20.0
        w_crashed = 50.0

        # Reward for moving forward
        forward_rew = w_forward * self.last_actions[id][1]

        # Update explored zones
        zone = self._compute_current_zone(self.states[id][0], self.states[id][2]) # PosX, PosZ
        if zone not in self.explored_zones[id]:
            self.explored_zones[id].append(zone)
            
        # Compute exploration reward
        explore_rew = w_explore * (len(self.explored_zones[id]) - 1)

        # Compute grid reward
        self.grids[id].update(self.states[id][0], self.states[id][2])
        grid_rew = w_grid * (self.grids[id].puntuation())

        # Get front distance to obstacles
        dist_center = self.decision_steps[id][0][8][1]

        # Compute distance penalty
        dist_penalty = self._distance_penalty(dist_center, w_proximity_penalty=w_distance)
        crash_penalty = w_crashed * self.crashed[id]

        # Compute reward
        reward = forward_rew + explore_rew + grid_rew
        # Compute penalty
        penalty = dist_penalty + crash_penalty
        # Compute final reward
        reward_id = reward - penalty
        self.last_rewards[id] = reward_id

        return reward_id   
    

    
    def _check_termination(self, id=0):
        if self.states[id][3] == 0: # Has crashed
            self.dones[id] = True
            self.crashed[id] = True
        if self.steps >= self.max_steps:
            self.dones[id] = True
        
        return self.dones[id]
    
    def _get_obs(self):
        # Get decision steps from Unity environment
        self.decision_steps, self.terminal_steps = self.unity_env.get_steps(self.behavior_name)
        observation = np.zeros((self.n_drones, self.n_obs), dtype=np.float32)
        
        # Get each drone observation (laser measures)
        for id in self.drone_ids:
            self.states[id] = self.decision_steps[id][0][9]
                    
            # Get laser info from Unity environment
            laser_measures = np.atleast_2d([])
            for i in range(7):
                laser = self.decision_steps[id][0][i]
                reading = np.atleast_2d([laser[21], laser[17], laser[13], laser[9], laser[5], laser[1], laser[3], laser[7], laser[11], laser[15], laser[19]])
                laser_measures = np.concatenate((laser_measures, reading), axis=1)
                
            height_measure = np.atleast_2d([self.decision_steps[id][0][7][1]])
            #laser_measures = self._normalize_laser_measures(laser_measures)
            laser_measures = np.concatenate((laser_measures, height_measure), axis=1)

            # Create observation vector
            obs_id = (laser_measures.flatten()).reshape(1, -1)
            observation[id,:] = obs_id

        return observation
    
    def _get_obs_id(self, id):
        # Get decision steps from Unity environment
        self.decision_steps, self.terminal_steps = self.unity_env.get_steps(self.behavior_name)
        
        # Get each drone observation (laser measures)
        self.states[id] = self.decision_steps[id][0][9]
                    
        # Get laser info from Unity environment
        laser_measures = np.atleast_2d([])
        for i in range(7):
            laser = self.decision_steps[id][0][i]
            reading = np.atleast_2d([laser[21], laser[17], laser[13], laser[9], laser[5], laser[1], laser[3], laser[7], laser[11], laser[15], laser[19]])
            laser_measures = np.concatenate((laser_measures, reading), axis=1)
                
        height_measure = np.atleast_2d([self.decision_steps[id][0][7][1]])
        #laser_measures = self._normalize_laser_measures(laser_measures)
        laser_measures = np.concatenate((laser_measures, height_measure), axis=1)

        # Create observation vector
        obs_id = (laser_measures.flatten()).reshape(1, -1)
        # Log last observation for this id
        self.last_observations[id,:] = obs_id

        return obs_id

    
    def _get_info(self, obs = None):
        # Log steps
        self.steps += 1
        infos = []
        for id in self.drone_ids:
            if (self.crashed[id] or self.dones[id]) and obs is not None:
                terminal_observation = obs[id].reshape(1, -1)
            else:
                terminal_observation = None
            info_dict = {
                'steps': self.steps,
                'terminal_observation': terminal_observation,
            }
            infos.append(info_dict)
        return infos
    
    def _normalize_laser_measures(self, laser_measures):
        min_val = 1.0
        max_val = 0.0

        for i in range(len(laser_measures[0])):
            if laser_measures[0][i] < min_val:
                min_val = laser_measures[0][i]
            if laser_measures[0][i] > max_val:
                max_val = laser_measures[0][i]
        
        diff = max_val - min_val

        for i in range(len(laser_measures[0])):
            laser_measures[0][i] = (laser_measures[0][i] - min_val) / diff
        
        return laser_measures
    
    def _compute_current_zone(self, x, z) -> int:
        
        for i in range(len(self.zones)):
            x0 = self.zones[i][0]
            z0 = self.zones[i][1]
            x1 = self.zones[i][2]
            z1 = self.zones[i][3]

            if (x > x0 and x < x1 and z > z0 and z < z1):
                return i
            
            return 0
        
    def _distance_penalty(self, distance, w_proximity_penalty) -> float:
        if distance <= self.max_distance:
            return w_proximity_penalty * (self.max_distance - distance) / self.max_distance
        else:
            return 0.0
