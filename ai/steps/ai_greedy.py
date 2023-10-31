import numpy as np

class AI_Greedy:    
    def __init__(self) -> None:        
        pass        

    def reset(self, init_state, state):
        """ Receives initial state """
        self.ai_kind = init_state['ai_kind'] 
        self.kinds   = init_state['kind'] 
    #---------------------------------------------------------------------------

    def step(self, state, reward):
        """ 
        Receives state and reward, returns simple actions.        
        """ 
        pos, vel, t_pos = state['pos'], state['vel'], state['target_pos']
        my = (self.ai_kind == self.kinds)
        
        action = np.zeros( (len(pos),5) )
        
        my_action = self.policy(t_pos[my], pos[my], vel[my])

        action[my] = my_action[:]
        return action 
    #---------------------------------------------------------------------------    

    def policy(self, X, x, v, eps=1e-8):
        """ From point x to point X with velocity v """
        r = X - x
        d = np.linalg.norm(r, axis=-1, keepdims=True)
        r /= (d+eps)  # unit vectors to dest
        v_len = np.linalg.norm(v, axis=-1, keepdims=True)
        v /= (v_len+eps)                                      # unit vector of velocity

        action = np.ones( (len(v),5) )

        si = v[:,0] * r[:,1] - v[:,1] * r[:,0]  # rotate
        action[si > 0, 1] = -1

        action[:,0] = (r * v).sum(axis=-1)      # force        
        action[v_len.reshape(-1) < 5, 0] = 1.   # if the car is parked, you have to go
        action[:, 2:] = r.copy()                # target direction
        return action
