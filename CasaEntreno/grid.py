class Cell:
    def __init__(self, x, y, grid_size) -> None:
        self.x = x*grid_size
        self.y = y*grid_size

class Grid:
    def __init__(self, width, height, grid_size):
        self.width = width
        self.height = height
        self.grid = [[Cell(x, y, grid_size) for y in range(width)] for x in range(height)]
