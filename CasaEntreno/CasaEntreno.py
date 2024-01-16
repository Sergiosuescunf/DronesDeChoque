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

parser = argparse.ArgumentParser(description="Specify the architecture and train parameters.")

# Define los argumentos que aceptará el programa
parser.add_argument("-i", "--inputs", type=int, help="Number of inputs", default=2)
parser.add_argument("-a", "--actions", type=int, help="Number of previous actions", default=0)
parser.add_argument("-o", "--obs", type=int, help="Number of previous observations", default=0)
parser.add_argument("-g", "--generation", type=int, help="Number of generation to start", default=0)
parser.add_argument("-mg", "--max_generations", type=int, help="Max number of generations", default=200)
parser.add_argument("-ps", "--population_size", type=int, help="Number of individuals per generation", default=100)
parser.add_argument("-es", "--elite_size", type=int, help="Number of elite individuals per generation", default=10)
parser.add_argument("-t", "--train", type=bool, help="Train the population", default=False)
parser.add_argument("-ng", "--nographics", type=bool, help="Don't show graphics", default=False)

args = parser.parse_args()

# Normalize score by grid and penalty and add weights (coefficients)

N_INPUTS = args.inputs 
N_ACTIONS = args.actions
N_OBS = args.obs 
NO_GRAPH = args.nographics

USES_ACT = N_ACTIONS > 0
USES_OBS = N_OBS > 0

assert N_INPUTS == 2, "N_INPUTS must be 2"  

print("")
print("N_INPUTS:", N_INPUTS)
print("N_ACTIONS:", N_ACTIONS)
print("N_OBS:", N_OBS)
print("USES_ACT:", USES_ACT)
print("USES_OBS:", USES_OBS)
print("")
print("Generation:", args.generation)
print("Max Generations:", args.max_generations)
print("Population Size:", args.population_size)
print("Elite Size:", args.elite_size)
print("")
print("Train:", args.train)
print("no_graph:", NO_GRAPH)

# TODO: Poner todas las puntuaciones entre 0 y 1 con sus pesos añadir puntuacion constante entre 0 y 10 que tenga en cuenta cuanto anda hacia delante menos los laterales ([alante - abs(laterales)] * 10(ejemplo))

# Score attributes
NumZones = 0
NewZoneScore = 100
MaxDistance = 0.55

## Score weights
w_grid_score = 70
w_zones_score = 100
w_movement_score = 0.01

## w_crash_penalty weights
w_crash_penalty = 50
w_proximity_penalty = 20

Zones = []

# Population Variables
PopulationSize = args.population_size
EliteSize = args.elite_size
Epoch = 0
GameEpoch = args.generation
print("GameEpoch:", GameEpoch)  
MaxEpochs = args.max_generations 
MaxSteps = 2000

# Normalize Lasers
normalize = False

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
MovementScores = []
ProximityPenalties = []
CurrentScore = 0
PreviousScore = 0
Crashed = []
Grid_coordinates = []

if args.train:
    Grid_coordinates = [-14.0, -16.0, 16.0, 14.0]
else:
    Grid_coordinates = [-12.0, -27.0, 36.0, 15.0]

# Save directory
directory = f"Elite_simple_{N_OBS}_obs_{N_ACTIONS}_act_dinamic_arquitecture/Experiment2/"

# Model
def create_model():
    
    bias_init = tf.keras.initializers.he_uniform()
    activation_func = tf.nn.relu

    n_inputs = 78 * (N_OBS+1) + N_ACTIONS * N_INPUTS
    n_intermediate_inputs =  32
    n_intermediate_inputs_2 = 16
  
    model = models.Sequential()
    model.add(layers.Dense(n_intermediate_inputs, input_shape = (n_inputs,), bias_initializer=bias_init, activation = activation_func))
    model.add(layers.Dense(n_intermediate_inputs_2, bias_initializer=bias_init, activation = activation_func))
    model.add(layers.Dense(N_INPUTS, bias_initializer=bias_init, activation = tf.nn.tanh))

    return model

def EuclideanDistance(p1, p2):
  return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def SetZones():
    
    global Zones
    global NumZones
    
    Zones.clear()

    if args.train:
        Zones.append((-13, -6, -2, 0))
        Zones.append((-13, -14, -2, -6))
        Zones.append((-2, -14, 15, 0))
        Zones.append((-13, 0, 15, 13))
    else:
        # Zones.append((-13, -6, -2, 0))
        # Zones.append((-13, -14, -2, -6))
        # Zones.append((-2, -14, 15, 0))
        # Zones.append((-13, 0, 15, 13))
        Zones.append((-6, -3, 0, 3))
        Zones.append((-12, -15, -6, -3))
        Zones.append((-12, -27, 0, -3))
        Zones.append((0, -27, 18, -9))
        Zones.append((18, -27, 30, -15))
        Zones.append((24, -15, 36, -9))
        Zones.append((24, -9, 36, 3))
        Zones.append((18, 3, 36, 15))
        Zones.append((12, 3, 18, 15))
        Zones.append((0, 3, 12, 15))
        Zones.append((9, -3, 18, 3))

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

def UpdateZonesAndGrid(id, x, z):
    
    zone = CalculateZone(x, z)
    
    if(zone not in Models[id].explored_zones):
        Models[id].explored_zones.append(zone)
    
    Models[id].update_grid(x,z) 

def CalculateGridScore(id):
    return w_grid_score * (Models[id].grid_score()/Models[id].total_cells())

def CalculateExploredZones(id):
    if len(Models[id].explored_zones) == 1:
        return 0
    
    return w_zones_score * (NumZones - len(Models[id].explored_zones)-1)/NumZones

def CalculateScore(grid_score, explored_zones_score, movement_score, proximy_penalty, crashed_penalty):
    return grid_score + explored_zones_score + movement_score - crashed_penalty - proximy_penalty

def DistancePenalty(distance):
    return w_proximity_penalty * (MaxDistance - distance)/MaxDistance 

def CalculateDistancePenalty(id, dist_center, dist_left, dist_right):

    if dist_center <= MaxDistance:
        pen_center = DistancePenalty(dist_center)
        ProximityPenalties[id] += pen_center

    if dist_left <= MaxDistance:
        pen_left = DistancePenalty(dist_left)
        ProximityPenalties[id] += pen_left

    if dist_right <= MaxDistance:
        pen_right = DistancePenalty(dist_right)
        ProximityPenalties[id] += pen_right

def CalculateMovementScore(id, movement):
    movement_score = w_movement_score * (movement[1] - abs(movement[0]))
    if movement_score < 0:
        movement_score = 0
    
    MovementScores[id] += movement_score
    

# Generates a new population from scratch
def NewPopulation():
    
    Models.clear()
    Scores.clear()
    ProximityPenalties.clear()
    MovementScores.clear()
    Crashed.clear()
    
    for i in range(PopulationSize):
        model = create_model()
        Models.append(Drone(model, Grid_coordinates))
        Scores.append(0.0)
        ProximityPenalties.append(0.0)
        MovementScores.append(0.0)
        Crashed.append(1)
        
def ResetDrones():
    Scores.clear()
    Crashed.clear()
    ProximityPenalties.clear()
    MovementScores.clear()
    for i in range(PopulationSize):
        Models[i].explored_zones.clear()
        Models[i].clean_grid()
        Scores.append(0.0)
        ProximityPenalties.append(0.0)
        MovementScores.append(0.0)
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
            Models.append(Drone(newChild, Grid_coordinates))

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
        Elite.append(Drone(new_model, Grid_coordinates))
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

    # TODO: remove if not needed
    # if(Epoch < 39):
    #     auxMS = (MaxSteps - 200)//40
    #     FinalStep = auxMS * (Epoch + 1) + 200
    # else:
    FinalStep = MaxSteps
        
    steps = 0
    NumCrashed = 0
    best = 0
    camera = 0
    bestScore = 0
    
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    action = spec.action_spec.random_action(len(decision_steps))

    obs_history = {}
    act_history = {}

    # Initialize the observation history for each agent
    if USES_OBS:
        obs_history = {id: [np.zeros(78, dtype=np.float32) for _ in range(N_OBS)] for id in range(PopulationSize)}

    # Initialize the observation history for each agent
    if USES_ACT:
        act_history = {id: [np.zeros(N_INPUTS, dtype=np.float32) for _ in range(N_ACTIONS)] for id in range(PopulationSize)}

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

                network_input = Lasers.flatten()
                
                if USES_OBS:
                    flattened_obs_history = np.concatenate([action.flatten() for action in obs_history[id]])
                    network_input = np.concatenate([network_input, flattened_obs_history])

                if USES_ACT:
                    flattened_act_history = np.concatenate([action.flatten() for action in act_history[id]])
                    network_input = np.concatenate([network_input, flattened_act_history])

                network_input = network_input.reshape(1, -1)
                Tensor = tf.constant(network_input)
                
                pred = Models[id].prediction(Tensor)

                pred_array = pred.numpy().flatten()

                CalculateMovementScore(id, pred_array)

                if USES_OBS:
                    obs_history[id].append(Lasers)  # Add the new action

                    if len(obs_history[id]) >= N_OBS:
                        obs_history[id].pop(0)  # Remove the oldest action if we already have n actions

                if USES_ACT:
                    act_history[id].append(pred_array)  # Add the new action

                    if len(act_history[id]) >= N_ACTIONS:
                        act_history[id].pop(0)  # Remove the oldest action if we already have n actions

                state = decision_steps[id][0][9]
                posX = state[0]
                posZ = state[2]
                
                UpdateZonesAndGrid(id, posX, posZ)

                dist_center = decision_steps[id][0][8][1]
                dist_left = decision_steps[id][0][8][3]
                dist_right = decision_steps[id][0][8][5]

                # CalculateDistancePenalty(id, dist_center, dist_left, dist_right)
                
                if state[3] == 0:
                    Crashed[id] = 0
                    Models[id].crashed = True
                    NumCrashed += 1
                    
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
            grid_score = CalculateGridScore(i)
            explored_zones_score = CalculateExploredZones(i)
            crash_penalty = int(Models[i].crashed) * w_crash_penalty
            proximy_penalty = ProximityPenalties[i]
            movement_score = MovementScores[i]
            Scores[i] = CalculateScore(grid_score, explored_zones_score, movement_score, proximy_penalty, crash_penalty)

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
            print("Step: " + str(steps) + " \t| Crashed: " + str(NumCrashed) + "\t|Best Drone: " + str(best) + "\t|Score of the Best : " + "%.2f" % bestScore + "\t| Zone of the Best: " + str(max(Models[best].explored_zones)) + "\t| Cells explored: " + str(Models[best].grid_score()))
            
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

    obs_history = {}
    act_history = {}

    # Initialize the observation history for each agent
    if USES_OBS:
        obs_history = {id: [np.zeros(78, dtype=np.float32) for _ in range(N_OBS)] for id in range(PopulationSize)}

    # Initialize the observation history for each agent
    if USES_ACT:
        act_history = {id: [np.zeros(N_INPUTS, dtype=np.float32) for _ in range(N_ACTIONS)] for id in range(PopulationSize)}

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

                network_input = Lasers.flatten()
                
                if USES_OBS:
                    flattened_obs_history = np.concatenate([action.flatten() for action in obs_history[id]])
                    network_input = np.concatenate([network_input, flattened_obs_history])

                if USES_ACT:
                    flattened_act_history = np.concatenate([action.flatten() for action in act_history[id]])
                    network_input = np.concatenate([network_input, flattened_act_history])

                network_input = network_input.reshape(1, -1)
                Tensor = tf.constant(network_input)
                
                pred = Models[id].prediction(Tensor)

                pred_array = pred.numpy().flatten()

                CalculateMovementScore(id, pred_array)

                if USES_OBS:
                    obs_history[id].append(Lasers)  # Add the new action

                    if len(obs_history[id]) >= N_OBS:
                        obs_history[id].pop(0)  # Remove the oldest action if we already have n actions

                if USES_ACT:
                    act_history[id].append(pred_array)  # Add the new action

                    if len(act_history[id]) >= N_ACTIONS:
                        act_history[id].pop(0)  # Remove the oldest action if we already have n actions

                state = decision_steps[id][0][9]
                posX = state[0]
                posZ = state[2]
                
                UpdateZonesAndGrid(id, posX, posZ)

                dist_center = decision_steps[id][0][8][1]
                dist_left = decision_steps[id][0][8][3]
                dist_right = decision_steps[id][0][8][5]

                CalculateDistancePenalty(id, dist_center, dist_left, dist_right)
                
                if state[3] == 0:
                    Crashed[id] = 0
                    Models[id].crashed = True
                    NumCrashed += 1
                    
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
            grid_score = CalculateGridScore(i)
            explored_zones_score = CalculateExploredZones(i)
            crash_penalty = int(Models[i].crashed) * w_crash_penalty
            proximy_penalty = ProximityPenalties[i]
            movement_score = MovementScores[i]
            Scores[i] = CalculateScore(grid_score, explored_zones_score, movement_score, proximy_penalty, crash_penalty)

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
            print("Step: " + str(steps) + " \t| Crashed: " + str(NumCrashed) + "\t|Best Drone: " + str(best) + "\t|Score of the Best : " + "%.2f" % bestScore + "\t| Zone of the Best: " + str(max(Models[best].explored_zones)) + "\t| Cells explored: " + str(Models[best].grid_score()))
            
        steps = steps + 1
        
        if(steps >= FinalStep):
            done = True
        if(NumCrashed >= PopulationSize):
            done = True
    print('Steps:', steps)
        
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
        
    env = UnityEnvironment(file_name=FILE_NAME, seed=1, no_graphics=NO_GRAPH, side_channels=[channel])
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

    
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(height=1024, width=1024)
        
    env = UnityEnvironment(file_name="CasaEntrenoTest.x86_64", seed=1, side_channels=[channel])
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
        
    aux = "Generation" + str(Epoch)

    test_scores = []

    for i in range(6):
        if(i != 0):
            env.reset()
            ResetDrones()
            LoadElite(aux)
            NewGeneration()

        print("Showing epoch: " + str(Epoch))
        ShowPopulationElite(env, behavior_name, spec)
        print("Score of the best drone: ", max(Scores))
        test_scores.append(max(Scores))
    
    print("Test scores:", np.mean(test_scores))

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