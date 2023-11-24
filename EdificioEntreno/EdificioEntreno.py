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
NumZonas = 0
ZonasPunt = 10
NuevaZonasPunt = 50
Penalizacion = 40
SectorVis = []
ZonasVis = []
DronesZona = [] 
Zonas = [] 

#Variiables Poblacion
TamPoblacion = 100 
TamElite = 10
Epoca = 0
EpocaPartida = 0
MaxEpocas = 100
MaxSteps = 4000

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
def modelo():
    
    bias_init = tf.keras.initializers.he_uniform()
    funct_atc = tf.nn.relu
  
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape = (78,), bias_initializer=bias_init, activation = funct_atc)) # type: ignore
    model.add(layers.Dense(8, bias_initializer=bias_init, activation = funct_atc)) # type: ignore
    model.add(layers.Dense(4, bias_initializer=bias_init, activation = tf.nn.tanh)) # type: ignore

    return model

def distEuclidea(p1, p2):
  return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def EstablecerZonas():
    
    global Zonas
    global NumZonas
    
    Zonas.clear()
    Zonas.append((-6, -3, 0, 3))
    Zonas.append((-12, -15, -6, -3))
    Zonas.append((-12, -27, 0, -3))
    Zonas.append((0, -27, 18, -9))
    Zonas.append((18, -27, 30, -15))
    Zonas.append((24, -15, 36, -9))
    Zonas.append((24, -9, 36, 3))
    Zonas.append((18, 3, 36, 15))
    Zonas.append((12, 3, 18, 15))
    Zonas.append((0, 3, 12, 15))
    Zonas.append((9, -3, 18, 3))

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
        
    return NumZonas

def CalcularPunt(id, x, z):
    
    auxX = int(x)
    auxZ = int(z)
    
    ZonaPas = DronesZona[id]
    DronesZona[id] = CalcularZona(x, z)
    zona = DronesZona[id]
    
    """
    if(zona != ZonaPas):
        Puntuaciones[id] += ZonasPunt
            
    """
    if(zona != ZonaPas):
        list = ZonasVis[id]
        
        if(zona not in list):
            list.append(zona)
            Puntuaciones[id] += NuevaZonasPunt
        else:
            Puntuaciones[id] += ZonasPunt     
    
    if(DronesZona[id] != 0):
        list = SectorVis[id]
        
        if((auxX, auxZ) not in list):
            list.append((auxX, auxZ))
            Puntuaciones[id] += 1

#Genera una nueva poblacion desde 0
def NuevaPoblacion():
    
    Modelos.clear()
    Puntuaciones.clear()
    Chocados.clear()
    DronesZona.clear()
    SectorVis.clear()
    ZonasVis.clear()
    
    for i in range(TamPoblacion):
        model = modelo()
        Modelos.append(model)
        Puntuaciones.append(0.0)
        Chocados.append(1)
        DronesZona.append(0)
        SectorVis.append([])
        ZonasVis.append([])
        
def ReiniciarDrones():
    for i in range(TamPoblacion):
        Puntuaciones[i] = 0.0
        Chocados[i] = 1
        DronesZona[i] = 0
        SectorVis[i].clear()
        ZonasVis[i].clear()
        
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

#Dado unos pesos aplica a cada neurona una probabilidad de mutar(ProbMuta) un cierto indice(IndiceMuta)
def mutarPesos2(pesos):
    for i in range(len(pesos)):
        if(i%2 == 0):
            for k in range(len(pesos[i][0])):
                if(random.uniform(0,1) < ProbMuta):
                    for j in range(len(pesos[i])):
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
    pesos1 = mutarPesos2(pesos1)
    nuevoModelo.set_weights(pesos1)

    return nuevoModelo

#Genera una poblacion nueva a partir de la Elite
def NuevaGeneracion():

    Modelos.clear()

    for x in Elite:
        Modelos.append(x)

    for x in range(TamPoblacion - TamElite):
        p1 = random.randint(0,TamElite-1)
        p2 = random.randint(0,TamElite-1)
        while(p1 == p2):
            p2 = random.randint(0,TamElite-1)
    
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

#Normaliza los valores de los láseres entre 0 y 1
def Normalizar(Laseres):
    
    """
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
    """     
    
    return Laseres

#Entrena una población
def EntrenarPoblacion(env, behavior_name, spec):

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
                #Laseres = Normalizar(Laseres)
                Laseres = np.concatenate((Laseres, altura), axis=1)
                Tensor = tf.constant(Laseres)
                
                pred = Modelos[id].call(Tensor, training=None, mask=None)
                
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

def AjustarMuatciones():
    
    global PuntActual
    global PuntPasada
    global ProbMuta
    global IndiceMuta
    
    ProbMuta = 10 / (PuntActual/5 + 10)
    
    with open(directorio + '/Datos.txt', 'a') as f:
        if(Epoca == 0):
            f.write(str(IncrPM) + " | " + str(MaxPM) + " | " + str(IncrIM) + " | " + str(MaxIM) + '\n')
        
    PuntPasada = PuntActual

def GuardarDatos():
    
    with open(directorio + '/Datos.txt', 'a') as f:
        f.write(str(PuntActual) + '\n')

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
        
    env = UnityEnvironment(file_name="EdificioEntreno", seed=1, side_channels=[])
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
            print("PuntActual:", PuntActual, ", PuntPasada:", PuntPasada) 
            #print("ProbMuta:", ProbMuta, ", IndiceMuta:", IndiceMuta)
            if(AjustMutacion):
                AjustarMuatciones()
            GuardarDatos()
            GuardarElite('Generacion' + str(Epoca))
            if(PuntActual > 0):
                NuevaGeneracion()
            else:
                NuevaPoblacion()
            ReiniciarDrones()
        
        print("Entrenando epoca: " + str(Epoca + 1))
        EntrenarPoblacion(env, behavior_name, spec)
        Epoca += 1
        
    env.close()
    print('UwU')
    
def MostrarPoblacion():
    
    global Epoca
    
    Epoca = EpocaPartida
    
    NuevaPoblacion()
    
    aux = "Generacion" + str(1)
    CargarElite(aux)
    NuevaGeneracion()
        
    env = UnityEnvironment(file_name="EdificioEntreno", seed=1, side_channels=[])
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
        
    for i in range(9):

        Epoca = i*10 + 1
        
        if(i != 0):
            
            aux = "Generacion" + str(Epoca)
            env.reset()
            ReiniciarDrones()
            CargarElite(aux)
            NuevaGeneracion()

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