import torch, torch.nn as nn

class MemoryBufXY:
    def __init__(self, capacity, x_dim, y_dim):
        self.capacity = capacity  # memory capacity (number of examples)
        self.count    = 0         # number of examples added   

        self.X = torch.empty( (capacity, x_dim), dtype=torch.float32)
        self.Y = torch.empty( (capacity, y_dim), dtype=torch.float32)

    def add(self, x,y):
        """ Add to memory x, y """
        idx = self.count % self.capacity
        self.X[idx] = torch.tensor(x, dtype=torch.float32)
        self.Y[idx] = torch.tensor(y, dtype=torch.float32)        
        self.count += 1

    def get(self, count):
        """ Return count of examples for (x,y) """        
        high = min(self.count, self.capacity)
        num  = min(count, high)
        ids = torch.randint(high = high, size = (num,) )
        return self.X[ids], self.Y[ids]
