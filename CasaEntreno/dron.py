import neat
from grid import Grid
from neat.activations import tanh_activation

class Dron:
    def __init__(self, config):
        self.config = config
        self.grid = Grid(-14, -16, 16, 14, 2)
        auto = False

    def setGenome(self, genome):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
        for i in range(len(self.net.node_evals)):
            if self.net.node_evals[i][0] in self.net.output_nodes:
                self.net.node_evals[i] = self.net.node_evals[i][:1] + (tanh_activation,) + self.net.node_evals[i][2:]

    def updateGrid(self, x, z):
        self.grid.update(x, z)
    
    def puntuacionGrid(self):
        return self.grid.puntuacion()

    def setFitness(self, fitness):
        self.genome.fitness = fitness    

    def prediction(self, inputs):
        return self.net.activate(inputs)