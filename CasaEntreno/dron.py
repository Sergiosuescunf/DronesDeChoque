import neat
from grid import Grid
from neat.activations import tanh_activation

GRID_SIZE = 1

class Dron:
    def __init__(self, config):
        self.config = config
        self.grid = Grid(-14, -16, 16, 14, GRID_SIZE)
        self.zonas_exploradas = []

    def setGenome(self, genome):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
        for i in range(len(self.net.node_evals)):
            if self.net.node_evals[i][0] in self.net.output_nodes:
                self.net.node_evals[i] = self.net.node_evals[i][:1] + (tanh_activation,) + self.net.node_evals[i][2:]

    def updateGrid(self, x, z):
        self.grid.update(x, z)

    def clean_grid(self):
        self.grid.clean_grid()
    
    def puntuacionGrid(self):
        return self.grid.puntuacion()

    def setFitness(self, fitness):
        self.genome.fitness = fitness    

    def prediction(self, inputs):
        return self.net.activate(inputs)