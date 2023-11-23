from djitellopy import Tello
import cv2, math, time, keyboard, os

import torch
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import models, layers

#MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transfrom = midas_transforms.small_transform

input()

#Tello
tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

for_back_velocity = 0
left_right_velocity = 0
up_down_velocity = 0
yaw_velocity = 0

speed = 50
rotSpeed = 100

#Modelo
NumLaserX = 11
NumLaserY = 7

def modelo():
    
    bias_init = tf.keras.initializers.he_uniform()
    funct_atc = tf.nn.relu
  
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape = (78,), bias_initializer=bias_init, activation = funct_atc))
    model.add(layers.Dense(8, bias_initializer=bias_init, activation = funct_atc))
    model.add(layers.Dense(4, bias_initializer=bias_init, activation = tf.nn.tanh))

    return model

def CargarModelo(model):

    path_model = 'Pesos.h5'
    model.load_weights(path_model)
    print('Pesos cargados')

def Mostrar(Laseres, altura, pred):
    
    os.system('cls')
    
    for i in range(7):
        for j in range(11):
            aux = i * 11 + j
            print(round(Laseres[0][aux], 2), end=' ')
        print()
        
    print(altura)
    
    print(round(float(pred[0][3]), 2), end = " ")
    print(round(float(pred[0][2]), 2), end = " ")
    print(round(float(pred[0][0]), 2), end = " ")
    print(round(float(pred[0][1]), 2), end = "\n")

    
def CalcularInput(img):
    
    distX = int(960/NumLaserX)
    distY = int(720/NumLaserY)
    
    mediaX = int(distX/2)
    mediaY = int(distY/2)
    
    
    Laseres = np.atleast_2d([])
    
    for y in range(NumLaserY):
        medidas = [0 for x in range(NumLaserX)]
        for x in range(NumLaserX):
            medidas[x] = 1 - img[distY*y + mediaY][distX*x + mediaX]
                    
        lectura = np.atleast_2d([medidas])
        Laseres = np.concatenate((Laseres, lectura), axis=1)
        
    return Laseres
            
model = modelo()
CargarModelo(model)
manual = True

#Main loop
while True:
    
    img = frame_read.frame
    input_batch = transfrom(img).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
        
    depth_map = prediction.cpu().numpy()
    depth_map_norm = cv2.normalize(depth_map, None, 0, 1, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    color_map = (depth_map_norm*255).astype(np.uint8)
    color_map_magma = cv2.applyColorMap(color_map, cv2.COLORMAP_MAGMA)
    
    cv2.imshow("drone", img)
    cv2.imshow('Depth', color_map_magma)
    cv2.waitKey(1)
    
    Laseres = CalcularInput(depth_map_norm)
    altura = np.atleast_2d([tello.get_distance_tof()/300])
    Laseres = np.concatenate((Laseres, altura), axis=1)
    
    Tensor = tf.constant(Laseres)
    pred = model.call(Tensor, training=None, mask=None)
    
    Mostrar(Laseres, altura, pred)
    
    for_back_velocity = 0
    left_right_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0

    if keyboard.is_pressed('m'):
        manual = True
    elif keyboard.is_pressed('n'):
        manual = False
    
    if(manual):
        if keyboard.is_pressed('q'):
            break
        if keyboard.is_pressed('b'):
            tello.land()
        if keyboard.is_pressed('v'):
            tello.takeoff()
                    
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
                
    else:
        up_down_velocity = int(pred[0][0] * 20)
        yaw_velocity = int((pred[0][1]) * 100)
        for_back_velocity = int(pred[0][2] * 20)
        left_right_velocity = int((pred[0][3]) * 20)
        
    tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)
    
tello.land()