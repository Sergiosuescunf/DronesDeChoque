from grid import Grid
import tensorflow as tf

GRID_SIZE = 0.5

class Drone:
    def __init__(self, modelo, grid_coordinates):
        self.model = modelo
        self.grid = Grid(grid_coordinates[0], grid_coordinates[1], grid_coordinates[2], grid_coordinates[3], GRID_SIZE)
        self.explored_zones = []
        self.crashed = False

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
        predictions = self.model.call(inputs, training=None, mask=None)

        second_output = predictions[:, 1]
        transformed_second_output = (second_output + 1) / 2
        predictions = tf.concat([predictions[:, :1], tf.expand_dims(transformed_second_output, -1)], axis=1)

        return predictions