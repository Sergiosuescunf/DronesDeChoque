import neat

class Dron:
    def __init__(self, config):
        self.config = config

    def setGenome(self, genome):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)


    def setFitness(self, fitness):
        self.genome.fitness = fitness    

    def prediction(self, inputs):
        return self.net.activate(inputs)