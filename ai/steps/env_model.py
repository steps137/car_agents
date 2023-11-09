import numpy as np

class AI_EnvModel:    
    """
    """
    def __init__(self) -> None:                
        pass 

    def reset(self, init_state, state):
        """ Receives initial state """
        N = len(init_state['kind'] )
        self.t_tot   = 0
        self.times   = np.zeros((N,))  
        self.v_avr   = np.zeros((N,))  # average agent speed
        self.beta    = 0.99                          # EMA speed averaging
    #---------------------------------------------------------------------------

    def step(self, state, reward=None, done=None):
        """ 
        Receives state and reward, returns simple actions.        
        """ 
        pos, vel, dir, t_pos = state['pos'], state['vel'], state['dir'], state['target_pos']
        self.times += state['dt']
        self.t_tot += state['dt']
        self.v_avr = self.v_avr*self.beta + (1-self.beta)*np.linalg.norm(vel, axis=-1)
                        
        action = self.policy(t_pos, pos, vel, dir)    
        return action
    #---------------------------------------------------------------------------    

    def policy(self, X, x, v, dir, eps=1e-8):
        """ From point x to point X with velocity v """
        action = np.ones( (len(v), 2) )

        r = X - x                                             # radius vector to target
        dist = np.linalg.norm(r, axis=-1)                     # distance to target
        n = r / (dist.reshape(-1,1)+eps)                      # unit vectors to distance

        vel = np.linalg.norm(v, axis=-1)                      # velocity
        v /= (vel.reshape(-1,1)+eps)                          # unit vector of velocity                

        si = n[:,0] * dir[:,1] - n[:,1] * dir[:,0]  # rotate
        co = (dir * n).sum(axis=-1)                 # cos with target

        action[si < 0, 1] = -1
        action[(dist < 5) & (co < 0.3), 1] *= -1

        if self.t_tot > 3:
            self.times[ self.v_avr < 0.5 ] = -3.
            action[self.times < 0, 0] = -1
            action[self.times < 0, 1] =  0
        
        return action
