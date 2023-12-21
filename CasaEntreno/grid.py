class Cell:
    def __init__(self, x, z, grid_size):
        self.x_init = x
        self.z_init = z
        self.x_final = x + grid_size
        self.z_final = z + grid_size
        self.visited = False

# X: -14 z: 14; X: 16 z: -16
class Grid:
    def __init__(self, x_init, z_init, x_final, z_final, grid_size):
        self.x_init = x_init
        self.z_init = z_init
        self.grid = [[Cell(x, z, grid_size) for z in range(z_init, z_final, grid_size)] for x in range(x_init, x_final, grid_size)]
        self.grid_size = grid_size
        
    def get_cell(self, x, z):
        for row in self.grid:
            for cell in row:
                if cell.x_init <= x < cell.x_final and cell.z_init <= z < cell.z_final:
                    return cell
    
    def update(self, x, z):
        cell = self.get_cell(x, z)
        cell.visited = True

    def puntuacion(self):
        puntuacion = 0
        for row in self.grid:
            for cell in row:
                if cell.visited:
                    puntuacion += 1
        return puntuacion