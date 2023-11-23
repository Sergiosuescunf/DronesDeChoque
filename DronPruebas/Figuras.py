from djitellopy import Tello
import cv2, math, time, keyboard, os

import torch
import numpy as np

for_back_velocity = 0
left_right_velocity = 0
up_down_velocity = 0
yaw_velocity = 0

speed = 40
speedFigura = 20
rotSpeed = 100

cont = 0

#Tello
tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

def Figura():
    
    global for_back_velocity
    global left_right_velocity
    global up_down_velocity
    global yaw_velocity
    
    global cont
    
    for_back_velocity = 0
    left_right_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0
    
    if(cont < 1000):
        up_down_velocity = 40 
    elif(cont < 2000): 
        left_right_velocity = -20
    elif(cont < 3000):
        up_down_velocity = -40
    else:
        left_right_velocity = 20
        
    cont += 1
    if(cont > 4000):
        cont = 0

def Main():
    
    global for_back_velocity
    global left_right_velocity
    global up_down_velocity
    global yaw_velocity

    global cont
    
    manual = True
    
    #Main loop
    while True:
        
        img = frame_read.frame
        
        cv2.imshow("drone", img)

        cv2.waitKey(1)
        
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
            Figura()
            
        tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)
        
    tello.land()

if __name__ == '__main__':
    Main()