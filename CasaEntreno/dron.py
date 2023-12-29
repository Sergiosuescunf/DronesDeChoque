import neat
from grid import Grid
from neat.activations import tanh_activation

GRID_SIZE = 1

class Dron:
    def __init__(self, modelo):
        self.modelo = modelo
        self.grid = Grid(-14, -16, 16, 14, GRID_SIZE)
        self.zonas_exploradas = []

    def guardar_modelo(self, filename):
        self.modelo.save_weights(filename)

    def updateGrid(self, x, z):
        self.grid.update(x, z)

    def clean_grid(self):
        self.grid.clean_grid()
    
    def puntuacionGrid(self):
        return self.grid.puntuacion()
    
    def total_cells(self):
        return self.grid.cell_count()

    def prediction(self, inputs):
        return self.modelo.call(inputs, training=None, mask=None)