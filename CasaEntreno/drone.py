from grid import Grid

GRID_SIZE = 1

class Drone:
    def __init__(self, modelo):
        self.model = modelo
        self.grid = Grid(-14, -16, 16, 14, GRID_SIZE)
        self.explored_zones = []

    def save_model(self, filename):
        self.model.save_weights(filename)

    def update_grid(self, x, z):
        self.grid.update(x, z)

    def clean_grid(self):
        self.grid.clean_grid()
    
    def grid_score(self):
        return self.grid.puntuation()
    
    def total_cells(self):
        return self.grid.cell_count()

    def prediction(self, inputs):
        return self.model.call(inputs, training=None, mask=None)