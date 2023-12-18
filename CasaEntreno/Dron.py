import neat
from neat.activations import tanh_activation

class Dron:
    def __init__(self, config):
        self.config = config

    def setGenome(self, genome):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
        for i in range(len(self.net.node_evals)):
            if self.net.node_evals[i][0] in self.net.output_nodes:
                self.net.node_evals[i] = self.net.node_evals[i][:1] + (tanh_activation,) + self.net.node_evals[i][2:]


    def setFitness(self, fitness):
        self.genome.fitness = fitness    

    def prediction(self, inputs):
        return self.net.activate(inputs)