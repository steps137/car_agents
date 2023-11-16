import numpy as np

class AI_Random:    
    def __init__(self, ticks_df=50, ticks_dw=10) -> None:
        self.ticks_df = ticks_df
        self.ticks_dw = ticks_dw
        self.cur_tick_df  = 0        
        self.cur_tick_dw  = 0        
        self.actions = None 

    def reset(self, init_state, state):
        """ Receives initial state """
        self.n_cars = init_state['n_cars']
        

    def step(self, state, reward=None, done=None):
        """ 
        Receives state and reward, returns random actions.
        To stop it from twitching so much, we change the action every 10 calls.
        """        
        if self.actions is None:
            self.actions = np.zeros((self.n_cars, 2))

        if self.cur_tick_df % self.ticks_df == 0:
            self.actions[:, 0] = 2. * np.random.randint(0, 2, size = (self.n_cars, )) - 1.

        if self.cur_tick_dw % self.ticks_dw == 0:
            self.actions[:, 1] = 2. * np.random.randint(0, 2, size = (self.n_cars, )) - 1.

        self.cur_tick_df += 1
        self.cur_tick_dw += 1

        return self.actions
