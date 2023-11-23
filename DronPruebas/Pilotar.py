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
import torch

#MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transfrom = midas_transforms.small_transform

#Unity Enviroment
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

S = 60
FPS = 120

for_back_velocity = 0
left_right_velocity = 0
up_down_velocity = 0
yaw_velocity = 0
speed = 2
rotSpeed = 2

def Normalizar(Laseres):
    
    min = 1.0
    max = 0.0
    
    for i in range(len(Laseres[0])):
        if Laseres[0][i] < min:
            min = Laseres[0][i]
        if Laseres[0][i] > max:
            max = Laseres[0][i]
    
    diff = max - min
    
    for i in range(len(Laseres[0])):
        Laseres[0][i] = (Laseres[0][i] - min) / diff
        
    return Laseres

def Mostrar(Laseres):
    
    for i in range(7):
        for j in range(11):
            aux = i * 11 + j
            print(round(Laseres[0][aux], 4), end=' ')
        print('')

#Main loop
while True:
    
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    """
    img = decision_steps[0][0][0]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', img_rgb)
    cv2.waitKey(1)
    """
    
    Laseres = np.atleast_2d([])
    
    os.system('cls')
    for i in range(7):
        laser = decision_steps[0][0][i+1]
        lectura = np.atleast_2d([laser[21], laser[17], laser[13], laser[9], laser[5], laser[1], laser[3], laser[7], laser[11], laser[15], laser[19]])
        Laseres  = np.concatenate((Laseres, lectura), axis=1)
    
    Laseres = Normalizar(Laseres)
    Mostrar(Laseres)
    altura = decision_steps[0][0][8][1]
    print(altura)
    
    for_back_velocity = 0
    left_right_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0

    if keyboard.is_pressed('q'):
        break
    if keyboard.is_pressed('r'):
        env.reset()
        
    if keyboard.is_pressed('i'):
        for_back_velocity = speed
    elif keyboard.is_pressed('k'):
        for_back_velocity = -speed
        
    if keyboard.is_pressed('l'):
        left_right_velocity = speed
    elif keyboard.is_pressed('j'):
        left_right_velocity = -speed
        
    if keyboard.is_pressed('w'):
        up_down_velocity = speed
    elif keyboard.is_pressed('s'):
        up_down_velocity = -speed
        
    if keyboard.is_pressed('d'):
        yaw_velocity = rotSpeed
    elif keyboard.is_pressed('a'):
        yaw_velocity = -rotSpeed

    #print("for_back_velocity:", for_back_velocity, "- left_right_velocity:", left_right_velocity, "- up_down_velocity:", up_down_velocity, "- yaw_velocity:", yaw_velocity)
    pred = np.array([[up_down_velocity, yaw_velocity, for_back_velocity, left_right_velocity, 0]], dtype = np.float32)
    action.add_continuous(pred)
    env.set_actions(behavior_name, action)
    env.step()
    
env.close()