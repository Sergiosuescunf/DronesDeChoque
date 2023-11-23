import random
import numpy as np
import mlagents
import keyboard 

import math
import os

from mlagents_envs.environment import UnityEnvironment

import cv2
import numpy as np
import time

env = UnityEnvironment(file_name="DronPruebas", seed=1, side_channels=[])
env.reset()

behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]
print("Number of observations : ", len(spec.observation_specs))

if spec.action_spec.continuous_size > 0:
    print(f"There are {spec.action_spec.continuous_size} continuous actions")
if spec.action_spec.is_discrete():
    print(f"There are {spec.action_spec.discrete_size} discrete actions")

decision_steps, terminal_steps = env.get_steps(behavior_name)
action = spec.action_spec.random_action(len(decision_steps))

for_back_velocity = 0
left_right_velocity = 0
up_down_velocity = 0
yaw_velocity = 0
speed = 1
rotSpeed = 1

aux = False

def MostrarInfo(Laseres, altura):
    
    os.system('cls')
    
    for i in range(7):
        for j in range(11):
            print(round(Laseres[0][i*11+j], 2), end=" ")
        print("\n")
        
    print("\nAltura: " + str(altura))    
    
def CalcularLaseres(decision_steps):
    
    Laseres = np.atleast_2d([])
    altura = 0
    
    for i in range(7):
        laser = decision_steps[0][0][i+1]
        lectura = np.atleast_2d([laser[21], laser[17], laser[13], laser[9], laser[5], laser[1], laser[3], laser[7], laser[11], laser[15], laser[19]])
        Laseres  = np.concatenate((Laseres, lectura), axis=1)
        
    altura = decision_steps[0][0][8][1]

    return Laseres, altura    

while True:
    
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    action = spec.action_spec.random_action(len(decision_steps))
    
    if(aux):
        Laseres, altura = CalcularLaseres(decision_steps)
        MostrarInfo(Laseres, altura)
    
    for_back_velocity = 0
    left_right_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0
    
    if keyboard.is_pressed('o'):
        aux = True
    if keyboard.is_pressed('p'):
        aux = False
        
    if keyboard.is_pressed('q'):
        break
    if keyboard.is_pressed('r'):
        env.reset()
        
    if keyboard.is_pressed('w'):
        for_back_velocity = speed
    elif keyboard.is_pressed('s'):
        for_back_velocity = -speed
        
    if keyboard.is_pressed('d'):
        left_right_velocity = speed
    elif keyboard.is_pressed('a'):
        left_right_velocity = -speed
        
    if keyboard.is_pressed('i'):
        up_down_velocity = speed
    elif keyboard.is_pressed('k'):
        up_down_velocity = -speed
        
    if keyboard.is_pressed('l'):
        yaw_velocity = rotSpeed
    elif keyboard.is_pressed('j'):
        yaw_velocity = -rotSpeed
        
    pred = np.array([[up_down_velocity, yaw_velocity, for_back_velocity, left_right_velocity, 0]], dtype = np.float32)
    
    print(pred)
    
    action.add_continuous(pred)
    env.set_actions(behavior_name, action)
    env.step()
    
env.close()
