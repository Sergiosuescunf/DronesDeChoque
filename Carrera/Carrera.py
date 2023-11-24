import random
import numpy as np
import mlagents

import tensorflow as tf
import math
import os
import time

from mlagents_envs.environment import UnityEnvironment
from tensorflow import keras
from keras import models, layers

#Atrivutos de puntuacion
NumSectores = 7 
SectorDist = [] 
Marcas = [] 

#Variiables Poblacion
TamPoblacion = 100 
TamElite = 10
Epoca = 0
EpocaPartida = 0
MaxEpocas = 100

#Variables de Mutacion
AjustMutacion = False
ProbMuta = .15
IndiceMuta = .1

IncrPM = 0.1 #Incremento de la probabilidad de mutacion
IncrIM = 0.2 #Incremento del índice de mutacion
MaxPM = .25
MaxIM = 0.2

#Atrivutos Poblacion
Modelos = []
Elite = []
Puntuaciones = [] 
PuntActual = 0
PuntPasada = 0
Tiempos = [] 
Chocados = []
Sectores = []

#Directorio de guardado
directorio = ""

def modelo():
    
    bias_init = tf.keras.initializers.he_uniform()
    funct_atc = tf.nn.tanh
  
    model = models.Sequential()
    model.add(layers.Dense(8, input_shape = (5,), bias_initializer=bias_init, activation = funct_atc))
    model.add(layers.Dense(6, bias_initializer=bias_init, activation = funct_atc))
    model.add(layers.Dense(2, bias_initializer=bias_init, activation = funct_atc))

    return model

def distEuclidea(p1, p2):
  return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def EstablecerMarcas():
    
    global Marcas
    
    Marcas.clear()
    Marcas.append((-90,50))
    Marcas.append((30,90))
    Marcas.append((90,30))
    Marcas.append((50,-90))
    Marcas.append((0,20))
    Marcas.append((-20,-60))
    Marcas.append((-90,-60))
  
def EstablecerDistancias():
    
    global SectorDist
    
    SectorDist.clear()
    for x in range(NumSectores):
        aux = distEuclidea(Marcas[(6+x)%7], Marcas[x])
        SectorDist.append(aux)

def CalcularSector(x, z, sector):

    if(x < 30 and z > 50 and sector%7 == 0):
        sector = sector + 1
    elif(x > 30 and z > 30 and sector%7 == 1):
        sector = sector + 1
    elif(x > 50 and z < 30 and sector%7 == 2):
        sector = sector + 1
    elif(x > 0 and x < 50 and z < 30 and sector%7 == 3):
        sector = sector + 1
    elif(x < 0  and x > -60 and z < 30 and z > -60 and sector%7 == 4):
        sector = sector + 1
    elif(x < -10 and z < -60 and sector%7 == 5):
        sector = sector + 1
    elif(x < -80 and z > -60 and z < 50 and sector%7 == 6):
        sector = sector + 1 

    return sector

def CalcularPunt(x, z, sector):    
    sector = CalcularSector(x, z, sector)
    marca = Marcas[sector%7]
    dist = distEuclidea(marca, (x, z))/10
    punt = 0

    for x in range(sector+1):
        punt += SectorDist[x%7]/10

    punt -= dist

    return punt, sector

#Genera una nueva poblacion desde 0
def NuevaPoblacion():
    
    Modelos.clear()
    Puntuaciones.clear()
    Tiempos.clear()
    Sectores.clear()
    Chocados.clear()
    
    for i in range(TamPoblacion):
        model = modelo()
        Modelos.append(model)
        Puntuaciones.append(0.0)
        Tiempos.append(0)
        Sectores.append(0)
        Chocados.append(1)
        
def ReiniciarCoches():
    for i in range(TamPoblacion):
        Sectores[i] = 0
        Chocados[i] = 1
        
#Selecciona a los mejores individuos de la generacion como Elite
def SelecElite():
    
    global PuntActual
    
    Elite.clear()
    PuntActual = 0

    pos = 0
    aux = []
    for x in range(TamElite):
        mejor = -9999
        for i in range(len(Modelos)):
            if(Puntuaciones[i] > mejor and not (i in aux)):
                mejor = Puntuaciones[i]
                pos = i
        aux.append(pos)
        
    os.system('cls')
            
    for x in aux:
        print("Coche:", x, end=" ")
        print("Tiempo:", Tiempos[x], end=" ")
        print("Punt:", "%.2f" % Puntuaciones[x])
        
        PuntActual += Puntuaciones[x]
        Elite.append(Modelos[x])

#Dado unos pesos aplica a cada uno una probabilidad de mutar(ProbMuta) un cierto indice(IndiceMuta)
def mutarPesos(pesos):
    for i in range(len(pesos)):
        if(i%2 == 0):
            for j in range(len(pesos[i])):
                for k in range(len(pesos[i][j])):
                    if(random.uniform(0,1) < ProbMuta):
                        cambio = random.uniform(- IndiceMuta, IndiceMuta)
                        pesos[i][j][k] += cambio
        else: #Bias
            for j in range(len(pesos[i])):
                if(random.uniform(0,1) < ProbMuta):
                    cambio = random.uniform(- IndiceMuta, IndiceMuta)
                    pesos[i][j] += cambio
    return pesos

#Dado dos modelos crea uno nuevo mezclando sus pesos y aplicando mutarPesos()
def cruceModelos(padre1, padre2):

    pesos1 = padre1.get_weights()
    pesos2 = padre2.get_weights()

    for i in range(len(pesos1)):
        if(i%2 == 0):
            for k in range(len(pesos1[i][0])):
                if(random.randint(0,1) == 0):
                    for j in range(len(pesos1[i])):
                        pesos1[i][j][k] = pesos2[i][j][k]
                    pesos1[i+1][k] = pesos2[i+1][k]      

    nuevoModelo = modelo()
    pesos1 = mutarPesos(pesos1)
    nuevoModelo.set_weights(pesos1)

    return nuevoModelo

#Genera una poblacion nueva a partir de la Elite
def NuevaGeneracion():

    Modelos.clear()

    for x in Elite:
        Modelos.append(x)

    for x in range(TamPoblacion - TamElite):
        p1 = random.randint(0,9)
        p2 = random.randint(0,9)
        while(p1 == p2):
            p2 = random.randint(0,9)
    
        nuevoHijo = cruceModelos(Elite[p1], Elite[p2])
        Modelos.append(nuevoHijo)

#Guarda los modelos de la última elite generada
def GuardarElite(nombre = 'Generacion1'):
    for xi in range(TamElite):
        Elite[xi].save_weights(directorio + nombre + ' Individuo' + str(xi) + '.h5')
    print('Elite guardada!')

#Carga los modelos de la élite guardada
def CargarElite(nombre = 'Generacion1'):
    Elite.clear()
    for x in range(TamElite):
        path_model = directorio + nombre + ' Individuo' + str(x) + '.h5'
        new_model = modelo()
        new_model.load_weights(path_model)
        Elite.append(new_model)
    print('Elite cargada!')

def EntrenarPoblacion(env, behavior_name, spec):

    if(Epoca < 18):
        MaxSteps = 25 * Epoca + 50
    else:
        MaxSteps = 500
        
    FinalStep = MaxSteps
        
    steps = 0
    NumChocados = 0
    NumMeta = 0

    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    action = spec.action_spec.random_action(len(decision_steps))
    
    done = False 
    while not done:
        
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        Movimientos = [[]]
        for id in decision_steps.agent_id:
            
            if(Chocados[id] == 1 and Sectores[id] == 7):
                Chocados[id] = 0
                Puntuaciones[id] += (MaxSteps - steps)/50
                NumMeta = NumMeta + 1
                pred = np.array([[0, 0.5]], dtype = np.float32)
                
            if(Chocados[id] == 1):
                laseres = decision_steps.obs[0][id]
                laser1 = laseres[9]
                laser2 = laseres[5]
                laser3 = laseres[1]
                laser4 = laseres[3]
                laser5 = laseres[7]
                
                Laser = np.atleast_2d([laser1, laser2, laser3, laser4, laser5])
                Tensor = tf.constant(Laser)
                
                pred = Modelos[id].call(Tensor, training=None, mask=None)
                
                estado = decision_steps.obs[1][id]
                posX = estado[0]
                posZ = estado[2]
                
                Puntuaciones[id], Sectores[id] = CalcularPunt(posX, posZ, Sectores[id])
                
                if estado[3] == 0:
                    Chocados[id] = 0
                    NumChocados = NumChocados + 1
                    
            else:
                pred = np.array([[0, 0]], dtype = np.float32)
        
            if len(Movimientos[0]) == 0:
                Movimientos = pred
            else:
                nuevoMovimiento = pred
                Movimientos = np.concatenate((Movimientos, nuevoMovimiento), axis=0)
            
        action.add_continuous(Movimientos)
        env.set_actions(behavior_name, action)
        env.step()
        
        if(Sectores[0] == 7 and FinalStep == MaxSteps):
            FinalStep = steps + 50
            print("Paso final: " + str(FinalStep))
            
        if(steps % 25 == 0):
            punt = 0
            for i in range(TamPoblacion):
                if(punt < Puntuaciones[i]):
                    punt = Puntuaciones[i]
            print("Paso: " + str(steps) + " \t| Chocados: " + str(NumChocados) + "\t| Mejor Punt: " + "%.2f" % punt)
            
        steps = steps + 1
        
        if(steps >= FinalStep):
            done = True
        if(NumChocados >= TamPoblacion):
            done = True
        if(NumMeta >= TamElite):
            done = True

def AjustarMuatciones():
    global PuntActual
    global PuntPasada
    global ProbMuta
    global IndiceMuta
    
    auxX = max(PuntActual - PuntPasada, 0)
    ProbMuta = IncrPM/(auxX + IncrPM/MaxPM)
    IndiceMuta = IncrIM/(auxX + IncrIM/MaxIM)
        
    with open(directorio + '/Datos.txt', 'a') as f:
        if(Epoca == 1):
            f.write(str(IncrPM) + " | " + str(MaxPM) + " | " + str(IncrIM) + " | " + str(MaxIM) + '\n')
        f.write(str(PuntActual) + '\n')
        
    print("PuntActual:", PuntActual, ", PuntPasada:", PuntPasada) 
    print("ProbMuta:", ProbMuta, ", IndiceMuta:", IndiceMuta)
        
    PuntPasada = PuntActual

def Entrenar():
    
    global Epoca
    global PuntActual
    global PuntPasada
    
    Epoca = EpocaPartida
    PuntActual = 0
    PuntPasada = 0

    NuevaPoblacion()
    
    if(EpocaPartida != 0):
        aux = "Generacion" + str(EpocaPartida)
        CargarElite(aux)
        NuevaGeneracion()
        
    env = UnityEnvironment(file_name="Carrera", seed=1, side_channels=[])
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]
    print("Number of observations : ", len(spec.observation_specs))

    if spec.action_spec.continuous_size > 0:
        print(f"There are {spec.action_spec.continuous_size} continuous actions")
    if spec.action_spec.is_discrete():
        print(f"There are {spec.action_spec.discrete_size} discrete actions")

    for i in range(MaxEpocas - EpocaPartida):
        
        if(i != 0):
            SelecElite()
            if(AjustMutacion and Epoca > 20):
                AjustarMuatciones()
            GuardarElite('Generacion' + str(Epoca))
            NuevaGeneracion()
            ReiniciarCoches()
        
        print("Entrenando epoca: " + str(Epoca + 1))
        EntrenarPoblacion(env, behavior_name, spec)
        Epoca += 1
        
    env.close()
    print('UwU')
    
def MostrarPoblacion():
    
    global Epoca
    
    Epoca = EpocaPartida
    
    NuevaPoblacion()
    
    if(EpocaPartida != 0):
        aux = "Generacion" + str(EpocaPartida)
        CargarElite(aux)
        NuevaGeneracion()
    else:
        print("Generación inexistente")
        return
        
    env = UnityEnvironment(file_name="Carrera", seed=1, side_channels=[])
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]
    print("Number of observations : ", len(spec.observation_specs))

    if spec.action_spec.continuous_size > 0:
        print(f"There are {spec.action_spec.continuous_size} continuous actions")
    if spec.action_spec.is_discrete():
        print(f"There are {spec.action_spec.discrete_size} discrete actions")
        
    EntrenarPoblacion(env, behavior_name, spec)

def Experimentos():
    
    global IncrPM
    global IncrIM 
    global MaxPM 
    global MaxIM 
    global directorio
    global AjustMutacion
    
    AjustMutacion = True
    IncrPM = 0.1 
    IncrIM = 0.2 
    MaxPM = 0.15
    MaxIM = 0.1
    
    Entrenar()

def CrearDirectorio():

    path = ""
    for folder_path in directorio.split("/"):
        path += folder_path+"/"
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")

def Comienzo():
    
    global directorio
    global EpocaPartida

    directorio = "Elite/Experimento1/"
    EpocaPartida = 0
    
    CrearDirectorio()

    EstablecerMarcas()
    EstablecerDistancias()
    
    Experimentos()

if __name__ == '__main__':
    Comienzo()

