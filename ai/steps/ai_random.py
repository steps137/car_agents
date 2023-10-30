import numpy as np

class AI_Random:    
    def __init__(self, ticks=10) -> None:
        self.ticks = ticks or 10
        self.cur_tick  = 0        
        self.actions = None 

    def reset(self, init_state, state):
        """ Receives initial state """
        self.n_cars = init_state['n_cars']
        

    def step(self, state, reward):
        """ 
        Receives state and reward, returns random actions.
        To stop it from twitching so much, we change the action every 10 calls.
        """        
        if self.cur_tick % self.ticks == 0 or self.actions is None:
            self.actions = 2. * np.random.randint(0, 2, size = (self.n_cars, 2)) - 1.
        self.cur_tick += 1
        return self.actions
