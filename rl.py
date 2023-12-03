import numpy as np, time
import torch

from quenv import Environment, Car
from qunet import  DQN, MLP

class MultiEnvironment:
    def __init__(self, N, reset_kind) -> None:                
        Environment.REWARD_TIME    = -1
        Environment.REWARD_TARGET  =  0    
        Environment.REWARD_CAR_CAR =  0
        Environment.REWARD_CAR_SEG =  0

        ai={'car': {'ai':None, 'num': N} }
        self.env = Environment(ai=ai, w=60, h=40,  mt2px=20, level=0)   # small space
        self.env.set_params(car_collision = False, seg_collision = False, 
                            moving_targets=True, moving_percent=100, stop_on_target=False)

        self.tot_resets = 0
        self.reset_kind = reset_kind

        #self.actions = np.array([[-1.,-1.],[-1.,0.],[-1.,1.], [ 0.,-1],[ 0.,0],[ 0.,1.], [1.,-1.],[1.,0],[1.,1.]])
        self.actions = np.array([[-1.,-1.], [-1.,1.], [1.,-1.], [1.,0], [1.,1.]])
        self.dt = 1/40.

    #------------------------------------------------------------------------------------

    def reset(self, cur=-1):
        if cur < 0:
            init_s, s0 = self.env.reset()        
            return self.features(s0)
        else:
            if self.reset_kind == 1:
                car = self.env.cars[cur]
                car.vel = np.zeros((3,))
                state = self.env.state(self.dt)
                return self.features(state)
            else:
                self.tot_resets += 1
                if self.tot_resets > 100:
                    self.tot_resets = 0
                    init_s, s0 = self.env.reset()        
                    return self.features(s0)
                else:
                    return None            

    #------------------------------------------------------------------------------------

    def step(self, a):
        """
        a: (N, 9) in (-1,-1); (-1,0); ...; (1,1)
        """
        action = np.zeros((len(a), 2))
        for i in range(len(a)):
            action[i] = self.actions[a[i]]

        state, reward, done = self.env.step(action, dt=self.dt)

        #reward = self.reward(state, reward, done)

        state = self.features(state)
        return state, reward, done, ""
    #------------------------------------------------------------------------------------
    def reward(self, state, reward, done):
        v = state['vel']
        v_len = np.linalg.norm(v, axis=-1)
        
        idx = (v_len < 5) & (done == 0)
        reward[idx] = reward[idx] - 0.5
        return reward
    #------------------------------------------------------------------------------------

    def close(self):
        pass
    #------------------------------------------------------------------------------------

    def features(self, state, vel_scale=20, dist_scale=10, eps=1e-6):
        """
        Начнём с преследования цели. Фичи - простые скаляры
        """
        pos, vel, dir1 = state['pos'], state['vel'], state['dir'] 
        whe  = state['wheels'] 
        t_pos, t_vel = state['target_pos'],  state['target_vel']

        vel   = vel   / vel_scale
        t_vel = t_vel / vel_scale

        v_len = np.linalg.norm(vel, axis=-1, keepdims=True)
        dir2= np.zeros_like(dir1)
        dir2[:, 0] = dir1[:, 1];   dir2[:, 1] = -dir1[:, 0];

        R = t_pos - pos
        V = t_vel - vel
        R_len = np.linalg.norm(R, axis=-1, keepdims=True)
        V_len = np.linalg.norm(V, axis=-1, keepdims=True)        

        tar1    = R / (R_len + eps)
        tar2    = np.zeros_like(tar1)
        tar2[:, 0] = tar1[:, 1];   tar2[:, 1] = -tar1[:, 0]

        K     = t_vel - tar1*(tar1*t_vel).sum(axis=-1).reshape(-1,1)

        state = np.hstack([
            v_len,
            (vel*dir1).sum(axis=-1).reshape(-1,1),
            (vel*dir2).sum(axis=-1).reshape(-1,1),
            whe,

            (tar1*dir1).sum(axis=-1).reshape(-1,1),
            (tar1*dir2).sum(axis=-1).reshape(-1,1),            

            np.tanh( R_len / dist_scale ),
            np.linalg.norm(t_vel, axis=-1, keepdims=True),
            V_len,            
            np.linalg.norm(t_vel, axis=-1, keepdims=True),
            (V * tar1).sum(axis=-1).reshape(-1, 1),
            (V * tar2).sum(axis=-1).reshape(-1, 1),
            (t_vel * dir1).sum(axis=-1).reshape(-1, 1),
            (t_vel * dir2).sum(axis=-1).reshape(-1, 1),
            (K * dir1).sum(axis=-1).reshape(-1, 1),
            (K * dir2).sum(axis=-1).reshape(-1, 1),
        ])
        return state
    #------------------------------------------------------------------------------------

dqn = DQN( )    
dqn.params(                  
    ticks      = 200,            
    decays     = 2000,      # number of episodes to decay eps1 - > eps2
    update     = 100,       # target model update rate (in frames = time steps)                         
    capacity   = 100_000,   # memory size
    reset      = 2,         # reset i-th agent in muli-agent mode when done
    loss       = 'mse',     # loss function (mse, huber)
    optim      = 'adam',    # optimizer (sgd, adam)
    lm         = 0.001,     # learning rate           
)
nS, nA, N = 17, 5, 10

env = MultiEnvironment(N, reset_kind=dqn.params.reset)
model = MLP(input=nS, output=nA,  hidden=[256, 64])      
dqn.view(ymin=-200, ymax=0)

print(dqn.params)
beg = time.time()
dqn.init(env, model, nS=nS, nA=nA)
dqn.multi_agent_training(30000, plots=10000)
print(f"time: {(time.time()-beg)/60: .1f}m")
#dqn.plot(f"best: {dqn.best.reward:7.1f}")

state = {
    'model':  dqn.best.model.state_dict(),
    'reward': dqn.best.reward,
    'cfg':    dqn.best.model.cfg.get_str(),
    'hist':   dqn.history,
    'params': dqn.params,
}