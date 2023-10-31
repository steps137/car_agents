import numpy as np
import matplotlib.pyplot as plt

class AI_Log:    
    def __init__(self, kind=0) -> None:        
        self.kind = kind
        self.iter = 0
        self.tm   = []
        self.v    = None; self.s    = None; self.w    = None
        self.r    = None;

    def reset(self, init_state, state):
        self.n_cars = init_state['n_cars']

    def step(self, state, reward=0):
        pos, vel, dir = state['pos'][0], state['vel'][0], state['dir'][0]
        wheels  = state['wheels'][0]
        tar_pos  = state['target_pos'][0]
        
        self.tm.append(state['dt'])

        self.r = pos if self.r is None else np.vstack((self.r, pos))

        self.w = wheels if self.w is None else np.vstack((self.w, wheels))
        v = np.linalg.norm(vel, axis=-1)        
        self.v = v if self.v is None else np.vstack((self.v, v))
        if self.iter == 0:
            self.prev_pos = pos
        s = np.linalg.norm(self.prev_pos - pos, axis=-1)
        self.s = s if self.s is None else np.vstack((self.s, s))
        self.prev_pos = pos

        action = np.zeros( (len(state['pos']), 2))
        #-----------------------------------------------------------------------
        if self.kind == 0:
            n1, n2 = 200,300
            if self.iter < n1:
                action[:,0] = 1.
            elif self.iter == n1+n2:
                tm = np.array(self.tm).cumsum()
                fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(12,3), facecolor ='w')
                ax1.set_title(f"Velocity n1:{n1}, n2:{n2}"); ax1.plot(tm, self.v); ax1.grid(ls=":")
                ax2.set_title("Distance"); ax2.plot(tm, self.s.cumsum(0)); ax2.grid(ls=":")
                ax3.set_title("Track");    ax3.plot(self.r[:,0], self.r[:,1]); ax3.grid(ls=":")
                plt.show()
                exit()
        #-----------------------------------------------------------------------
        elif self.kind == 1:            
            action[:,0] = 1.
            n1, n2 = 100,200
            if self.iter < n1:
                action[:,1] = 1.
            elif self.iter == n1+n2:
                tm = np.array(self.tm).cumsum()
                fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(12,3), facecolor ='w')
                ax1.set_title(f"wheels n1:{n1}, n2:{n2}")
                ax1.plot(tm, self.w[:, 0]); 
                ax1.plot(tm, self.w[:, 1]); 
                ax1.grid(ls=":")
                ax2.set_title(f"Velocity n1:{n1}, n2:{n2}"); ax2.plot(tm, self.v); ax2.grid(ls=":")
                ax3.set_title("Track");    ax3.plot(self.r[:,0], self.r[:,1]); ax3.grid(ls=":")
                plt.show()
                exit()

        self.iter += 1
        return action
