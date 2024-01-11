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

    def __init__(self, unity_env, seed=33):
        super(CasaGymEnv, self).__init__()
        self.seed = seed
        
        # Save Unity environment
        self.unity_env = unity_env
        # Init Unity environment
        self.unity_env.reset()

        # Get info from Unity environment
        self.behavior_name = list(self.unity_env.behavior_specs)[0]
        self.behavior_spec = self.unity_env.behavior_specs[self.behavior_name]
        self.last_action = self.behavior_spec.action_spec.random_action(n_agents=1) # Init action variable

        if self.behavior_spec.action_spec.continuous_size > 0:
            self.action_space = spaces.Box(low=0., high=1., shape=(self.behavior_spec.action_spec.continuous_size-1,), dtype=np.float32)
        elif self.behavior_spec.action_spec.discrete_size > 0:
            self.action_space = spaces.Discrete(self.behavior_spec.action_spec.discrete_size)        
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(1,78))

        # Define different zones
        self.zones = []
        self.zones.append([-13, -6, -2, 0])
        self.zones.append([-13, -14, -2, -6])
        self.zones.append([-2, -14, 15, 0])
        self.zones.append([-13, 0, 15, 13])

        # Create utility variables for reward computation
        self.grid = Grid(-14, -16, 16, 14, 1)
        self.explored_zones = []
        self.max_distance = 0.55 #TODO: Change this to a parameter
        self.proximity_penalty = 20 #TODO: Change this to a parameter
        self.crashed_penalty = 150

        # Create utility variables for info
        self.steps = 0
        self.action_buffer = []
        self.obs_buffer = []
        self.rew_buffer = []

        # Create utility variables for termination
        self.max_steps = 2000 #TODO: Change this to a parameter
        self.done = False
        self.crashed = False


    def reset(self, seed=33):
        self.unity_env.reset()
        self.done = False
        self.crashed = False
        # Get observation from Unity environment
        observation = self._get_obs()
        # Reset utility variables for reward computation
        self.grid.clean_grid()
        self.explored_zones = []
        # Reset utility variables for info
        self.steps = 0
        self.action_buffer = []
        self.obs_buffer = []
        self.rew_buffer = []
        # Reset last action
        self.last_action = self.behavior_spec.action_spec.random_action(n_agents=1)
        # Return observation and info
        info = self._get_info()

        return observation, info

    def step(self, action: NDArray[np.float32]) -> Tuple[NDArray[np.float32], float, bool, Dict[str, Any]]:

        # Add drone state 0.2 = Alive, 0.1 = Dead
        if self.crashed:
            action = np.append(action, [0.1])
        else: 
            action = np.append(action, [0.2])
        
        # Pass action to Unity environment
        self.last_action.add_continuous(action.reshape((1,3)))
        self.unity_env.set_actions(self.behavior_name, self.last_action)        
        self.unity_env.step()              

        # Get observation from Unity environment
        obs = self._get_obs()
        # Check termination criteria
        done = self.check_termination()
        # Compute reward
        reward = self.compute_reward()
        # Get info
        info = self._get_info()

        return obs, reward, done, done, info

    def render(self, mode='human') -> NDArray[np.float32]:
        pass

    def close(self):
        self.unity_env.close()
    
    def compute_reward(self) -> float:
        # Update explored zones
        zone = self._compute_current_zone(self.state[0], self.state[2]) # PosX, PosZ
        if zone not in self.explored_zones:
            self.explored_zones.append(zone)

        # Update exploration grid
        self.grid.update(self.state[0], self.state[2]) # PosX, PosZ


        # Get distance to obstacles
        dist_center = self.decision_steps[0][0][8][1]
        dist_left = self.decision_steps[0][0][8][3]
        dist_right = self.decision_steps[0][0][8][5]

        # Compute distance penalty
        penalty = self._distance_penalty(dist_center) + self._distance_penalty(dist_left) + self._distance_penalty(dist_right)
        penalty += self.crashed_penalty * self.crashed

        # Compute exploration reward
        reward = self.grid.puntuation() + (len(self.explored_zones)-1) * 100

        return reward-penalty
    
    def check_termination(self) -> bool:
        if self.state[3] == 0: # Has crashed
            self.done = True
            self.crashed = True
        if self.steps >= self.max_steps:
            self.done = True
        
        return self.done
    
    def _get_obs(self):
        # Get decision steps from Unity environment
        self.decision_steps, self.terminal_steps = self.unity_env.get_steps(self.behavior_name)
        self.state = self.decision_steps[0][0][9]
                
        # Get laser info from Unity environment
        for id in self.decision_steps.agent_id:
            laser_measures = np.atleast_2d([])
            for i in range(7):
                laser = self.decision_steps[id][0][i]
                reading = np.atleast_2d([laser[21], laser[17], laser[13], laser[9], laser[5], laser[1], laser[3], laser[7], laser[11], laser[15], laser[19]])
                laser_measures = np.concatenate((laser_measures, reading), axis=1)
            
            height_measure = np.atleast_2d([self.decision_steps[id][0][7][1]])
            laser_measures = self._normalize_laser_measures(laser_measures)
            laser_measures = np.concatenate((laser_measures, height_measure), axis=1)

        # Create observation vector
        observation = (laser_measures.flatten()).reshape(1, -1)


        return observation
    
    def _get_info(self):
        # Log steps
        self.steps += 1
        info_dict = {
            'steps': self.steps,
        }
        return info_dict
    
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
    def _distance_penalty(self, distance) -> float:
        if distance <= self.max_distance:
            return self.proximity_penalty * (self.max_distance - distance) / self.max_distance
        else:
            return 0.0
