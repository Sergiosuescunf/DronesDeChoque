import random
import numpy as np
import mlagents

import tensorflow as tf
import math
import os
import time

import argparse

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from tensorflow import keras
from keras import models, layers
from drone import *

os_name = os.name

if os_name == 'posix':
    FILE_NAME = 'CasaEntreno.x86_64'
    CLEAR_COMMAND = 'clear'
elif os_name == 'nt':
    FILE_NAME = 'CasaEntreno.exe'
    CLEAR_COMMAND = 'cls'

parser = argparse.ArgumentParser(description="Especifica los parámetros de la arquitectura y del entrenamiento.")

# Define los argumentos que aceptará el programa
parser.add_argument("-i", "--inputs", type=int, help="Number of inputs", default=2)
parser.add_argument("-a", "--actions", type=int, help="Number of previous actions", required=True)
parser.add_argument("-g", "--generation", type=int, help="Number of generation to start", default=0)
parser.add_argument("-mg", "--max_generations", type=int, help="Max number of generations", default=100)
parser.add_argument("-ps", "--population_size", type=int, help="Number of individuals per generation", default=200)
parser.add_argument("-es", "--elite_size", type=int, help="Number of elite individuals per generation", default=10)
parser.add_argument("-t", "--train", type=bool, help="Train the population", default=False)
# Analiza los argumentos
args = parser.parse_args()

# Normalize score by grid and penalty and add weights (coefficients)

N_INPUTS = args.inputs 
N_ACTIONS = args.actions 

# Score attributes
NumZones = 0
ZoneScore = 50
Penalty = 40
ProximityPenalty = 20
NewZoneScore = 100
MaxDistance = 0.3
Zones = [] 

# Population Variables
PopulationSize = args.population_size
EliteSize = args.elite_size
Epoch = 0
GameEpoch = args.generation
print("GameEpoch:", GameEpoch)  
MaxEpochs = args.max_generations 
MaxSteps = 3000

# Normalize Lasers
normalize = True

# Mutation Variables
AdjustMutation = False
MutationProb = 0.1
MutationRate = 0.1

IncrMP = 0.1 # Increment of mutation probability
IncrMR = 0.2 # Increment of mutation rate
MaxMP = .25
MaxMR = 0.2

# Population Attributes
Models = []
Elite = []
Scores = [] 
Penalties = []
CurrentScore = 0
PreviousScore = 0
Crashed = []

# Save directory
directory = f"Elite_simple_{N_ACTIONS}_actions_dinamic_arquitecture/Experiment1/"

# Model
def create_model():
    
    bias_init = tf.keras.initializers.he_uniform()
    activation_func = tf.nn.relu

    n_inputs = 78 + N_ACTIONS*N_INPUTS
    n_intermediate_inputs =  n_inputs/2
    n_intermediate_inputs_2 = n_intermediate_inputs/2 
  
    model = models.Sequential()
    model.add(layers.Dense(n_intermediate_inputs, input_shape = (n_inputs,), bias_initializer=bias_init, activation = activation_func))
    model.add(layers.Dense(n_intermediate_inputs_2, bias_initializer=bias_init, activation = activation_func))
    model.add(layers.Dense(2, bias_initializer=bias_init, activation = tf.nn.tanh))

    return model

def EuclideanDistance(p1, p2):
  return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def SetZones():
    
    global Zones
    global NumZones
    
    Zones.clear()
    Zones.append((-13, -6, -2, 0))
    Zones.append((-13, -14, -2, -6))
    Zones.append((-2, -14, 15, 0))
    Zones.append((-13, 0, 15, 13))

    NumZones = len(Zones)
        
def LoadMap():
    SetZones()
    
def CalculateZone(x, z):
    
    for i in range(NumZones):
        
        x0 = Zones[i][0]
        z0 = Zones[i][1]
        x1 = Zones[i][2]
        z1 = Zones[i][3]
            
        if(x > x0 and x < x1 and z > z0 and z < z1):
            return i
        
    return 0

def CalculateScore(id, x, z):
    
    zone = CalculateZone(x, z)
    
    if(zone not in Models[id].explored_zones):
        Models[id].explored_zones.append(zone)
    
    Models[id].update_grid(x,z) 

def DistancePenalty(distance):
    return -ProximityPenalty * (MaxDistance - distance)/MaxDistance 

def CalculateDistancePenalty(id, dist_center, dist_left, dist_right):

    if dist_center <= MaxDistance:
        pen_center = DistancePenalty(dist_center)
        Penalties[id] += pen_center

    if dist_left <= MaxDistance:
        pen_left = DistancePenalty(dist_left)
        Penalties[id] += pen_left

    if dist_right <= MaxDistance:
        pen_right = DistancePenalty(dist_right)
        Penalties[id] += pen_right

# Generates a new population from scratch
def NewPopulation():
    
    Models.clear()
    Scores.clear()
    Penalties.clear()
    Crashed.clear()
    
    for i in range(PopulationSize):
        model = create_model()
        Models.append(Drone(model))
        Scores.append(0.0)
        Penalties.append(0.0)
        Crashed.append(1)
        
def ResetDrones():
    Scores.clear()
    Crashed.clear()
    Penalties.clear()
    for i in range(PopulationSize):
        Models[i].explored_zones.clear()
        Models[i].clean_grid()
        Scores.append(0.0)
        Penalties.append(0.0)
        Crashed.append(1)
        
# Selects the best individuals of the generation as Elite
def SelectElite():
    
    global CurrentScore
    
    Elite.clear()
    CurrentScore = 0

    pos = 0
    aux = []
    for x in range(EliteSize):
        best = -9999
        for i in range(len(Models)):
            if(Scores[i] > best and not (i in aux)):
                best = Scores[i]
                pos = i
        aux.append(pos)
        
    os.system(CLEAR_COMMAND)
            
    for x in aux:
        print("Drone:", x, end=" ")
        print("Score:", "%.2f" % Scores[x])
        
        CurrentScore += Scores[x]
        Elite.append(Models[x])

# Given some weights, applies a chance to mutate each one based on MutationProb and MutationRate
def mutateWeights(weights):
    for i in range(len(weights)):
        if(i%2 == 0):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    if(random.uniform(0,1) < MutationProb):
                        change = random.uniform(- MutationRate, MutationRate)
                        weights[i][j][k] += change
        else: # Bias
            for j in range(len(weights[i])):
                if(random.uniform(0,1) < MutationProb):
                    change = random.uniform(- MutationRate, MutationRate)
                    weights[i][j] += change
    return weights

# Given some weights, applies a chance to mutate each neuron based on MutationProb and MutationRate
def mutateWeights2(weights):
    for i in range(len(weights)):
        if(i%2 == 0):
            for k in range(len(weights[i][0])):
                if(random.uniform(0,1) < MutationProb):
                    for j in range(len(weights[i])):
                        change = random.uniform(- MutationRate, MutationRate)
                        weights[i][j][k] += change
        else: # Bias
            for j in range(len(weights[i])):
                if(random.uniform(0,1) < MutationProb):
                    change = random.uniform(- MutationRate, MutationRate)
                    weights[i][j] += change
    return weights

# Given two models, creates a new one by mixing their weights and applying mutateWeights()
def CrossModels(parent1, parent2):

    weights1 = parent1.model.get_weights()
    weights2 = parent2.model.get_weights()

    for i in range(len(weights1)):
        if(i%2 == 0):
            for k in range(len(weights1[i][0])):
                if(random.randint(0,1) == 0):
                    for j in range(len(weights1[i])):
                        weights1[i][j][k] = weights2[i][j][k]
                    weights1[i+1][k] = weights2[i+1][k]      

    newModel = create_model()
    weights1 = mutateWeights2(weights1)
    newModel.set_weights(weights1)

    return newModel

# Generates a new population from the Elite
def NewGeneration():

    Models.clear()

    for x in Elite:
        Models.append(x)
    if(args.train):
        for x in range(PopulationSize - EliteSize):
            p1 = random.randint(0,EliteSize-1)
            p2 = random.randint(0,EliteSize-1)
            while(p1 == p2):
                p2 = random.randint(0,EliteSize-1)
        
            newChild = CrossModels(Elite[p1], Elite[p2])
            Models.append(Drone(newChild))

# Saves the models of the last generated elite
def SaveElite(name = 'Generation1'):
    for xi in range(EliteSize):
        Elite[xi].save_model(directory + name + ' Individual' + str(xi) + '.h5')
    print('Elite saved!')

# Loads the models of the saved elite
def LoadElite(name = 'Generation1'):
    Elite.clear()
    for x in range(EliteSize):
        path_model = directory + name + ' Individual' + str(x) + '.h5'
        new_model = create_model()
        new_model.load_weights(path_model)
        Elite.append(Drone(new_model))
    print('Elite loaded!')

# Normalizes the values of the lasers between 0 and 1
def Normalize(Lasers):
    
    min_val = 1.0
    max_val = 0.0
    
    for i in range(len(Lasers[0])):
        if Lasers[0][i] < min_val:
            min_val = Lasers[0][i]
        if Lasers[0][i] > max_val:
            max_val = Lasers[0][i]
    
    diff = max_val - min_val
    
    for i in range(len(Lasers[0])):
        Lasers[0][i] = (Lasers[0][i] - min_val) / diff
        
    return Lasers

# Trains a population
def TrainPopulation(env, behavior_name, spec):

    if(Epoch < 39):
        auxMS = (MaxSteps - 200)//40
        FinalStep = auxMS * (Epoch + 1) + 200
    else:
        FinalStep = MaxSteps
        
    steps = 0
    NumCrashed = 0
    best = 0
    camera = 0
    bestScore = 0
    
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    action = spec.action_spec.random_action(len(decision_steps))

    # Initialize the action history for each agent
    action_history = {id: [np.zeros(2, dtype=np.float32) for _ in range(N_ACTIONS)] for id in range(PopulationSize)}

    pred = np.array([0, 0, 0, 0], dtype = np.float32)
    
    done = False 
    while not done:
        
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        Movements = [[]]
        
        for id in decision_steps.agent_id:
                
            if(Crashed[id] == 1):
                
                Lasers = np.atleast_2d([])
                
                for i in range(7):
                    laser = decision_steps[id][0][i]
                    reading = np.atleast_2d([laser[21], laser[17], laser[13], laser[9], laser[5], laser[1], laser[3], laser[7], laser[11], laser[15], laser[19]])
                    Lasers  = np.concatenate((Lasers, reading), axis=1)
                
                height = np.atleast_2d([decision_steps[id][0][7][1]])
                if(normalize):
                    Lasers = Normalize(Lasers)
                Lasers = np.concatenate((Lasers, height), axis=1)

                flattened_history = np.concatenate([action.flatten() for action in action_history[id]])
                network_input = np.concatenate([Lasers.flatten(), flattened_history])
                network_input = network_input.reshape(1, -1)
                Tensor = tf.constant(network_input)
                
                pred = Models[id].prediction(Tensor)
                pred_array = pred.numpy().flatten()

                if len(action_history[id]) >= N_ACTIONS:
                    action_history[id].pop(0)  # Remove the oldest action if we already have n actions
                
                action_history[id].append(pred_array)  # Add the new action

                state = decision_steps[id][0][9]
                posX = state[0]
                posZ = state[2]
                
                CalculateScore(id, posX, posZ)

                dist_center = decision_steps[id][0][8][1]
                dist_left = decision_steps[id][0][8][3]
                dist_right = decision_steps[id][0][8][5]

                CalculateDistancePenalty(id, dist_center, dist_left, dist_right)
                
                if(Crashed[id] == 0):
                    NumCrashed += 1
                
                if state[3] == 0:
                    Crashed[id] = 0
                    NumCrashed = NumCrashed + 1
                    
            else:
                pred = np.array([[0, 0]], dtype = np.float32)
            

            if(Crashed[id] == 0):
                pred = np.concatenate((pred, np.array([[0.1]])), axis=1)
            elif(id == 0 and steps < 11):
                pred = np.concatenate((pred, np.array([[0.2]])), axis=1)
            elif(Crashed[camera] == 0 and id == best):
                pred = np.concatenate((pred, np.array([[0.2]])), axis=1)
                camera = id
            elif(id < EliteSize):
                pred = np.concatenate((pred, np.array([[0.3]])), axis=1)    
            else:
                pred = np.concatenate((pred, np.array([[0]])), axis=1)

            if len(Movements[0]) == 0:
                Movements = pred
            else:
                newMovement = pred
                Movements = np.concatenate((Movements, newMovement), axis=0)
        
        for i in range(len(Scores)):
            Scores[i] = Models[i].grid_score() + Penalties[i] + len(Models[i].explored_zones) * NewZoneScore

        action.add_continuous(Movements)
        env.set_actions(behavior_name, action)
        env.step()
            
        if(steps % 25 == 0):
            bestScore = 0
            best = 0
            for i in range(PopulationSize):
                if(bestScore < Scores[i] and Crashed[i] == 1):
                    bestScore = Scores[i]
                    best = i
            print("Step: " + str(steps) + " \t| Crashed: " + str(NumCrashed) + "\t|Best Drone: " + str(best) + "\t|Score of the Best : " + "%.2f" % bestScore + "\t| Zone of the Best: " + str(max(Models[best].explored_zones)))
            
        steps = steps + 1
        
        if(steps >= FinalStep):
            done = True
        if(NumCrashed >= PopulationSize):
            done = True
        
        """ 
        if(steps % 100 == 0):
            if(auxBestScore == bestScore):
                done = True
            else:
                auxBestScore = bestScore
        """


def ShowPopulationElite(env, behavior_name, spec):

    if(Epoch < 39):
        auxMS = (MaxSteps - 200)//40
        FinalStep = auxMS * (Epoch + 1) + 200
    else:
        FinalStep = MaxSteps
        
    steps = 0
    NumCrashed = 0
    best = 0
    camera = 0
    bestScore = 0
    
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    action = spec.action_spec.random_action(len(decision_steps))

    # Initialize the action history for each agent
    action_history = {id: [np.zeros(2, dtype=np.float32) for _ in range(N_ACTIONS)] for id in range(PopulationSize)}

    pred = np.array([0, 0, 0, 0], dtype = np.float32)
    
    done = False 
    while not done:
        
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        Movements = [[]]
        
        for id in decision_steps.agent_id:
                
            if(Crashed[id] == 1):
                
                Lasers = np.atleast_2d([])
                
                for i in range(7):
                    laser = decision_steps[id][0][i]
                    reading = np.atleast_2d([laser[21], laser[17], laser[13], laser[9], laser[5], laser[1], laser[3], laser[7], laser[11], laser[15], laser[19]])
                    Lasers  = np.concatenate((Lasers, reading), axis=1)
                
                height = np.atleast_2d([decision_steps[id][0][7][1]])
                if(normalize):
                    Lasers = Normalize(Lasers)
                Lasers = np.concatenate((Lasers, height), axis=1)

                flattened_history = np.concatenate([action.flatten() for action in action_history[id]])
                network_input = np.concatenate([Lasers.flatten(), flattened_history])
                network_input = network_input.reshape(1, -1)
                Tensor = tf.constant(network_input)
                
                pred = Models[id].prediction(Tensor)
                pred_array = pred.numpy().flatten()

                if len(action_history[id]) >= N_ACTIONS:
                    action_history[id].pop(0)  # Remove the oldest action if we already have n actions
                
                action_history[id].append(pred_array)  # Add the new action

                state = decision_steps[id][0][9]
                posX = state[0]
                posZ = state[2]
                
                CalculateScore(id, posX, posZ)

                dist_center = decision_steps[id][0][8][1]
                dist_left = decision_steps[id][0][8][3]
                dist_right = decision_steps[id][0][8][5]

                CalculateDistancePenalty(id, dist_center, dist_left, dist_right)
                
                if(Crashed[id] == 0):
                    NumCrashed += 1
                
                if state[3] == 0:
                    Crashed[id] = 0
                    NumCrashed = NumCrashed + 1
                    
            else:
                pred = np.array([[0, 0]], dtype = np.float32)
            

            if(Crashed[id] == 0):
                pred = np.concatenate((pred, np.array([[0.1]])), axis=1)
            elif(id == 0 and steps < 11):
                pred = np.concatenate((pred, np.array([[0.2]])), axis=1)
            elif(Crashed[camera] == 0 and id == best):
                pred = np.concatenate((pred, np.array([[0.2]])), axis=1)
                camera = id
            elif(id < EliteSize):
                pred = np.concatenate((pred, np.array([[0.3]])), axis=1)    
            else:
                pred = np.concatenate((pred, np.array([[0]])), axis=1)

            if len(Movements[0]) == 0:
                Movements = pred
            else:
                newMovement = pred
                Movements = np.concatenate((Movements, newMovement), axis=0)
        
        # for i in range(len(Scores)):
        #     Scores[i] = Models[i].grid_score() + Penalties[i] + len(Models[i].explored_zones) * NewZoneScore

        action.add_continuous(Movements)
        env.set_actions(behavior_name, action)
        env.step()
            
        # if(steps % 25 == 0):
        #     bestScore = 0
        #     best = 0
        #     for i in range(PopulationSize):
        #         if(bestScore < Scores[i] and Crashed[i] == 1):
        #             bestScore = Scores[i]
        #             best = i
        #     print("Step: " + str(steps) + " \t| Crashed: " + str(NumCrashed) + "\t|Best Drone: " + str(best) + "\t|Score of the Best : " + "%.2f" % bestScore + "\t| Zone of the Best: " + str(max(Models[best].explored_zones)))
            
        steps = steps + 1
        
        if(steps >= FinalStep):
            done = True
        if(NumCrashed >= PopulationSize):
            done = True
        
def save_stats():
    max_score = max(Scores)
    max_index = Scores.index(max_score)
    max_grid_score = Models[max_index].grid_score()
    total_cells = Models[max_index].total_cells()
    with open(f"{directory}/stats.txt", "a") as f:
        f.write(f"Generation: {Epoch} | Drone: {max_index} | Score: {max_score} | Explored Cells: {max_grid_score} of {total_cells} total cells\n")

def AdjustMutations():
    
    global CurrentScore
    global PastScore
    global MutaProb
    global MutaIndex
    
    MutaProb = 10 / (CurrentScore/5 + 10)
    
    with open(directory + '/Data.txt', 'a') as f:
        if(Epoch == 0):
            f.write(str(IncrMP) + " | " + str(MaxMP) + " | " + str(IncrMR) + " | " + str(MaxMR) + '\n')
        
    PastScore = CurrentScore

def SaveData():
    
    with open(directory + '/Data.txt', 'a') as f:
        f.write(str(CurrentScore) + '\n')

def Train():
    
    global Epoch
    global CurrentScore
    global PastScore
    
    Epoch = GameEpoch
    CurrentScore = 0
    PastScore = 0

    NewPopulation()
    
    if(GameEpoch != 0):
        aux = "Generation" + str(GameEpoch)
        LoadElite(aux)
        NewGeneration()

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(height=1024, width=1024)
        
    env = UnityEnvironment(file_name=FILE_NAME, seed=1, no_graphics=True, side_channels=[channel])
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

    for i in range(MaxEpochs - GameEpoch):
        
        if(i != 0):
            SelectElite()
            print("CurrentScore:", CurrentScore, ", PastScore:", PastScore) 
            #print("MutaProb:", MutaProb, ", MutaIndex:", MutaIndex)
            if(AdjustMutation):
                AdjustMutations()
            SaveData()
            SaveElite('Generation' + str(Epoch))
            NewGeneration()
            ResetDrones()
        
        print("Training epoch: " + str(Epoch + 1))
        TrainPopulation(env, behavior_name, spec)
        Epoch += 1

        save_stats()
        
    env.close()
    print('UwU')
    
def ShowPopulation():
    
    global Epoch

    Epoch = GameEpoch

    NewPopulation()
    
    aux = "Generation" + str(Epoch)
    LoadElite(aux)
    NewGeneration()

    
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

        if(i != 0):
            Epoch = i*5 + 1
            aux = "Generation" + str(Epoch)
            env.reset()
            ResetDrones()
            LoadElite(aux)
            NewGeneration()

        print("Showing epoch: " + str(Epoch + 1))
        ShowPopulationElite(env, behavior_name, spec)

def CreateDirectory():

    path = ""
    for folder_path in directory.split("/"):
        path += folder_path+"/"
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")


def Start():
    CreateDirectory()
    LoadMap()
    if(args.train):
        Train()
    else:
        ShowPopulation()
    # Train()

if __name__ == '__main__':
    Start()