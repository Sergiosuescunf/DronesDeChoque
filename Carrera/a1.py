import random
import numpy as np

import tensorflow as tf
import math
import os

from tensorflow import keras
from keras import models, layers

def modelo():
  
  bias_init = tf.keras.initializers.glorot_uniform()
  
  model = models.Sequential()
  model.add(layers.Dense(8, input_shape = (5,), bias_initializer=bias_init, activation = tf.nn.relu))
  model.add(layers.Dense(6, bias_initializer=bias_init, activation = tf.nn.relu))
  model.add(layers.Dense(2, bias_initializer=bias_init, activation = tf.nn.relu))

  return model
        
def mutarPesos(pesos):
  for i in range(len(pesos)):
    if(i%2 == 0): #Pesos
      for j in range(len(pesos[i])):
        for k in range(len(pesos[i][j])):
          if(random.uniform(0,1) > 0.75):
              cambio = random.uniform(- 0.1, 0.1)
              pesos[i][j][k] += cambio
    else: #Bias
      for j in range(len(pesos[i])):
        if(random.uniform(0,1) > 0.75):
          cambio = random.uniform(- 0.1, 0.1)
          pesos[i][j] += cambio
          
  return pesos

#Dado dos modelos crea uno nuevo mezclando sus pesos y aplicando mutarPesos()
def cruceModelos(padre, madre):

  pesos1 = padre.get_weights()
  pesos2 = madre.get_weights()

  for i in range(len(pesos1)):
    if(i%2 == 0):
      for k in range(len(pesos1[i][0])):
        if(random.randint(0,1) == 0):
          for j in range(len(pesos1[i])):
            pesos1[i][j][k] = pesos2[i][j][k] #Pesos
          pesos1[i+1][k] = pesos2[i+1][k] #Bias

  nuevoModelo = modelo()
  pesos1 = mutarPesos(pesos1)
  nuevoModelo.set_weights(pesos1)

  return nuevoModelo
  
def PrintModel(model):
  
  print(model.get_weights()[2])
  print("\n")
  print(model.get_weights()[3])
  print("\n")

if __name__ == '__main__':
  
  padre = modelo()
  PrintModel(padre)
  
  madre = modelo()
  PrintModel(madre)
  
  hijo = cruceModelos(padre, madre)
  PrintModel(hijo)
  