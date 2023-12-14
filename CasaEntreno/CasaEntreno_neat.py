import random
import numpy as np
import mlagents
import neat

import tensorflow as tf
import math
import os
import time

from neat.checkpoint import Checkpointer
from mlagents_envs.environment import UnityEnvironment
from tensorflow import keras
from keras import models, layers
from Dron import *


so = os.name

if so == 'posix':
    FILE_NAME = 'CasaEntreno.x86_64'
    CLEAR_COMMAND = 'clear'
elif so == 'nt':
    FILE_NAME = 'CasaEntreno.exe'
    CLEAR_COMMAND = 'cls'

#Atrivutos de puntuacion
NumZonas = 0
ZonasPunt = 50
Penalizacion = 40
SectorVis = []
DronesZona = [] 
Zonas = [] 

#Variiables Poblacion
TamPoblacion = 100 
TamElite = 10
Epoca = 0
EpocaPartida = 0
MaxEpocas = 50
MaxSteps = 2000

#Normalizar Láseres
normalizar = True

#Variables de Mutacion
AjustMutacion = False
ProbMuta = 0.1
IndiceMuta = 0.1

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
Chocados = []

#Directorio de guardado
directorio = ""

#Modelo
def modelo(n_actions=4):
    
    bias_init = tf.keras.initializers.he_uniform()
    funct_atc = tf.nn.relu

    n_inputs = 78 + n_actions*4
  
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape = (n_inputs,), bias_initializer=bias_init, activation = funct_atc)) # type: ignore
    model.add(layers.Dense(8, bias_initializer=bias_init, activation = funct_atc)) # type: ignore
    model.add(layers.Dense(4, bias_initializer=bias_init, activation = tf.nn.tanh)) # type: ignore

    return model

def distEuclidea(p1, p2):
  return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def EstablecerZonas():
    
    global Zonas
    global NumZonas
    
    Zonas.clear()
    Zonas.append((-13, -6, -2, 0))
    Zonas.append((-13, -14, -2, -6))
    Zonas.append((-2, -14, 15, 0))
    Zonas.append((-13, 0, 15, 13))

    NumZonas = len(Zonas)
        
def CargarMapa():
    EstablecerZonas()
    
def CalcularZona(x, z):
    
    for i in range(NumZonas):
        
        x0 = Zonas[i][0]
        z0 = Zonas[i][1]
        x1 = Zonas[i][2]
        z1 = Zonas[i][3]
            
        if(x > x0 and x < x1 and z > z0 and z < z1):
            return i
        
    return 0

def CalcularPunt(id, x, z):
    
    auxX = int(x)
    auxZ = int(z)
    
    ZonaPas = DronesZona[id]
    DronesZona[id] = CalcularZona(x, z)
    
    if(DronesZona[id] != ZonaPas):
        Puntuaciones[id] += ZonasPunt
    
    if(DronesZona[id] != 0):
        list = SectorVis[id]
        
        if((auxX, auxZ) not in list):
            list.append((auxX, auxZ))
            Puntuaciones[id] += 1

def ReiniciarGeneracion():
    Puntuaciones.clear()
    for i in range(TamPoblacion):
        Puntuaciones.append(0.0)
        Chocados.append(1)
        DronesZona.append(0)
        SectorVis.append([])

def ReiniciarDrones():
    for i in range(TamPoblacion):
        Chocados[i] = 1
        DronesZona[i] = 0
        SectorVis[i].clear()

#Normaliza los valores de los láseres entre 0 y 1
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

#Entrena una población
def EntrenarPoblacion(env, behavior_name, spec, n_actions=4):

    if(Epoca < 39):
        auxMS = (MaxSteps - 200)//40
        FinalStep = auxMS * (Epoca + 1) + 200
    else:
        FinalStep = MaxSteps
        
    steps = 0
    NumChocados = 0
    mejor = 0
    camara = 0
    mejorPunt = 0
    
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    action = spec.action_spec.random_action(len(decision_steps))

    # Inicializar el historial de acciones para cada agente
    historial_acciones = {id: [np.zeros(4, dtype=np.float32) for _ in range(n_actions)] for id in range(TamPoblacion)}

    pred = np.array([0, 0, 0, 0], dtype = np.float32)
    
    done = False 
    while not done:
        
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        Movimientos = [[]]
        
        for id in decision_steps.agent_id:
                
            if(Chocados[id] == 1):
                
                Laseres = np.atleast_2d([])
                
                for i in range(7):
                    laser = decision_steps[id][0][i]
                    lectura = np.atleast_2d([laser[21], laser[17], laser[13], laser[9], laser[5], laser[1], laser[3], laser[7], laser[11], laser[15], laser[19]])
                    Laseres  = np.concatenate((Laseres, lectura), axis=1)
                
                altura = np.atleast_2d([decision_steps[id][0][7][1]])
                if(normalizar):
                    Laseres = Normalizar(Laseres)
                Laseres = np.concatenate((Laseres, altura), axis=1)

                historial_aplanado = np.concatenate([accion.flatten() for accion in historial_acciones[id]])
                entrada_red = np.concatenate([Laseres.flatten(), historial_aplanado])
                entrada_red = entrada_red.reshape(1, -1)
                Tensor = tf.constant(entrada_red)
                
                pred = Modelos[id].prediction(Tensor)
                pred_array = pred.numpy().flatten()

                if len(historial_acciones[id]) >= n_actions:
                    historial_acciones[id].pop(0)  # Eliminar la acción más antigua si ya tenemos n acciones
                
                historial_acciones[id].append(pred_array)  # Añadir la nueva acción

                estado = decision_steps[id][0][8]
                posX = estado[0]
                posZ = estado[2]
                
                CalcularPunt(id, posX, posZ)
                
                if(Chocados[id] == 0):
                    NumChocados += 1
                
                if estado[3] == 0:
                    Chocados[id] = 0
                    if(Puntuaciones[id] > 0):
                        Puntuaciones[id] -= Penalizacion
                    NumChocados = NumChocados + 1
                    
            else:
                pred = np.array([[0, 0, 0, 0]], dtype = np.float32)
            

            """
            if(Chocados[id] == 1 and Sectores[id] == NumZonas - 1):
                Chocados[id] = 0
                Puntuaciones[id] += (MaxSteps - steps)/100
                NumMeta = NumMeta + 1
                pred = np.concatenate((pred, np.array([[0.4]])), axis=1) 
            """    
            
            if(Chocados[id] == 0):
                pred = np.concatenate((pred, np.array([[0.1]])), axis=1)
            elif(id == 0 and steps < 11):
                pred = np.concatenate((pred, np.array([[0.2]])), axis=1)
            elif(Chocados[camara] == 0 and id == mejor):
                pred = np.concatenate((pred, np.array([[0.2]])), axis=1)
                camara = id
            elif(id < TamElite):
                pred = np.concatenate((pred, np.array([[0.3]])), axis=1)    
            else:
                pred = np.concatenate((pred, np.array([[0]])), axis=1)

            if len(Movimientos[0]) == 0:
                Movimientos = pred
            else:
                nuevoMovimiento = pred
                Movimientos = np.concatenate((Movimientos, nuevoMovimiento), axis=0)
        
        action.add_continuous(Movimientos)
        env.set_actions(behavior_name, action)
        env.step()
            
        if(steps % 25 == 0):
            mejorPunt = 0
            mejor = 0
            for i in range(TamPoblacion):
                if(mejorPunt < Puntuaciones[i] and Chocados[i] == 1):
                    mejorPunt = Puntuaciones[i]
                    mejor = i
            print("Paso: " + str(steps) + " \t| Chocados: " + str(NumChocados) + "\t| Mejor Punt: " + "%.2f" % mejorPunt + "\t| Mejor Zona: " + str(DronesZona[mejor]))
            
        steps = steps + 1
        
        if(steps >= FinalStep):
            done = True
        if(NumChocados >= TamPoblacion):
            done = True
        
        """ 
        if(steps % 100 == 0):
            if(auxMP == mejorPunt):
                done = True
            else:
                auxMP = mejorPunt
        """

def assign_rewards(genomes):
    for i, (_, genome) in enumerate(genomes):
        genome.fitness = Puntuaciones[i]

def Entrenar():
    
    global Epoca
    global EpocaPartida
    global PuntActual
    global PuntPasada
    global Modelos
    
    Epoca = EpocaPartida
    PuntActual = 0
    PuntPasada = 0
        
    env = UnityEnvironment(file_name=FILE_NAME, seed=1, no_graphics=True, side_channels=[])
    env.reset()
    time.sleep(5)

    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]
    print("Number of observations : ", len(spec.observation_specs))

    if spec.action_spec.continuous_size > 0:
        print(f"There are {spec.action_spec.continuous_size} continuous actions")
    if spec.action_spec.is_discrete():
        print(f"There are {spec.action_spec.discrete_size} discrete actions")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        os.path.dirname(os.path.abspath(__file__)) + '/neat_config.txt')  # Asegúrate de cambiar 'config_file_path' al camino a tu archivo de configuración
        
    # Crea un objeto Checkpointer
    checkpointer = Checkpointer(generation_interval=None, time_interval_seconds=None, filename_prefix='src/prueba1/src/neat-checkpoints/neat-checkpoint-')
    # Decide si cargar desde un checkpoint o crear una población nueva
    load_checkpoint = False  # Cambiar a True si desea cargar desde un checkpoint
    checkpoint_file = 'src/prueba1/src/neat-checkpoints/neat-checkpoint-1'  # Cambia 'x' por el número de generación del checkpoint que quieres cargar
    if load_checkpoint:
        population = checkpointer.restore_checkpoint(checkpoint_file)
        EpocaPartida = population.generation
    else:
        population = neat.Population(config)
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        EpocaPartida = 0

    Modelos = [Dron(config) for i in range(100)]

    for i in range(MaxEpocas - EpocaPartida):
        ReiniciarGeneracion()
        genomes = list(population.population.items())
        
        ReiniciarDrones()

        for i, node in enumerate(Modelos):
            node.setGenome(genomes[i][1])

        EntrenarPoblacion(env, behavior_name, spec)
        
        # Guarda un checkpoint después de cada generación
        checkpointer.save_checkpoint(config, population.population, population.species, i)

        # Actualiza la población basándose en las recompensas calculadas
        population.run(assign_rewards,1)
    env.close()

    
def MostrarPoblacion():
    
    global Epoca
    
    Epoca = EpocaPartida
    
    env = UnityEnvironment(file_name=FILE_NAME, seed=1, side_channels=[])
    env.reset()
    
    time.sleep(5)

    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]
    print("Number of observations : ", len(spec.observation_specs))

    if spec.action_spec.continuous_size > 0:
        print(f"There are {spec.action_spec.continuous_size} continuous actions")
    if spec.action_spec.is_discrete():
        print(f"There are {spec.action_spec.discrete_size} discrete actions")
        
    for i in range(6):
        print("Mostrando epoca: " + str(Epoca + 1))
        EntrenarPoblacion(env, behavior_name, spec)

def CrearDirectorio():

    path = ""
    for folder_path in directorio.split("/"):
        path += folder_path+"/"
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")


def Comienzo():
    
    global EpocaPartida
    EpocaPartida = 0
    
    global directorio
    directorio = "Elite/Experimento1/"

    CrearDirectorio()
    CargarMapa()
    #MostrarPoblacion()
    Entrenar()

if __name__ == '__main__':
    Comienzo()