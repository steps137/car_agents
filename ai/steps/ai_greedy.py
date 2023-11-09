import numpy as np

class AI_Greedy:    
    """
    Ignoring possible collisions, we strive for the goal.
    Various heuristic control algorithms have been implemented (algo=0,1,2,3)
    For a description of the algorithms, see https://qudata.com/ml/ru/agents/car_01.html
    """

    def __init__(self, algo = 0) -> None:        
        self.algo = algo                       # kind of the system control

    def reset(self, init_state, state):
        """ Receives initial state """
        self.ai_kind = init_state['ai_kind'] 
        self.kinds   = init_state['kind'] 
        self.t_tot   = 0
        self.times   = np.zeros((len(self.kinds),))  
        self.v_avr   = np.zeros((len(self.kinds),))  # average agent speed
        self.beta    = 0.99                           # EMA speed averaging
    #---------------------------------------------------------------------------

    def step(self, state, reward=None, done=None):
        """ 
        Receives state and reward, returns simple actions.        
        """ 
        pos, vel, dir, t_pos = state['pos'], state['vel'], state['dir'], state['target_pos']
        self.times += state['dt']
        self.t_tot += state['dt']
        self.v_avr = self.v_avr*self.beta + (1-self.beta)*np.linalg.norm(vel, axis=-1)

        my = (self.ai_kind == self.kinds)
        
        action = np.zeros( (len(pos),5) )
        
        my_action = self.policy(t_pos[my], pos[my], vel[my], dir[my])

        action[my] = my_action[:]
        return action 
    #---------------------------------------------------------------------------    

    def policy(self, X, x, v, dir, eps=1e-8):
        """ From point x to point X with velocity v """
        r = X - x                                             # radius vector to target
        dist = np.linalg.norm(r, axis=-1)                     # distance to target
        n = r / (dist.reshape(-1,1)+eps)                      # unit vectors to distance

        vel = np.linalg.norm(v, axis=-1)                      # velocity
        v /= (vel.reshape(-1,1)+eps)                          # unit vector of velocity        

        action = np.ones( (len(v), 5) )

        si = n[:,0] * dir[:,1] - n[:,1] * dir[:,0]  # rotate
        co = (dir * n).sum(axis=-1)                 # cos with target

        if self.algo == 0:
            action[:,      0] =  1
            action[si < 0, 1] = -1

        elif self.algo == 1:
            action[ (vel > 2) & (dist < 10), 0] =  -1.            
            action[si < 0, 1] = -1

        elif self.algo == 2:                                    
            action[si < 0, 1] = -1
            action[(dist < 5) & (co < 0.3), 1] *= -1

        elif self.algo == 3:        
            action[si < 0, 1] = -1
            action[(dist < 5) & (co < 0.3), 1] *= -1

            if self.t_tot > 3:
                self.times[ self.v_avr < 0.5 ] = -3.
                action[self.times < 0, 0] = -1
                action[self.times < 0, 1] =  0

        action[:, 2:] = n.copy()                    # target direction
        return action


