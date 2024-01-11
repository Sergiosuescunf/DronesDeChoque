class Cell:
    def __init__(self, x, z, grid_size):
        self.x_init = x
        self.z_init = z
        self.x_final = x + grid_size
        self.z_final = z + grid_size
        self.visited = False

class Grid:
    def __init__(self, x_init, z_init, x_final, z_final, grid_size):
        self.x_init = x_init
        self.z_init = z_init
        self.grid_size = grid_size
        self.grid = []

        x = x_init
        while x < x_final:
            row = []
            z = z_init
            while z < z_final:
                row.append(Cell(x, z, grid_size))
                z += grid_size
            self.grid.append(row)
            x += grid_size
        
    def get_cell(self, x, z):
        for row in self.grid:
            for cell in row:
                if cell.x_init <= x < cell.x_final and cell.z_init <= z < cell.z_final:
                    return cell
    
    def clean_grid(self):
        for row in self.grid:
            for cell in row:
                cell.visited = False
    
    def update(self, x, z):
        cell = self.get_cell(x, z)
        cell.visited = True

    def puntuation(self):
        puntuation = 0
        for row in self.grid:
            for cell in row:
                if cell.visited:
                    puntuation += 1
        return puntuation
    
    def cell_count(self):
        count = 0
        for row in self.grid:
            for _ in row:
                count += 1
        return count
