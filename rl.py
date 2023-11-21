import numpy as np, time
import torch

from environment.pygame.environment import Environment
from qunet import  DQN, MLP

class MultiEnvironment:
    def __init__(self, N) -> None:                
        Environment.REWARD_TIME    = -1
        Environment.REWARD_TARGET  =  0    
        Environment.REWARD_CAR_CAR =  0
        Environment.REWARD_CAR_SEG =  0

        ai={'car': {'ai':None, 'num': N} }
        self.env = Environment(ai=ai, w=60, h=40,  mt2px=20, level=0)   # small space
        self.env.set_params(car_collision = False, seg_collision = False, moving_targets=True)

        self.actions = np.array([[-1.,-1.],[-1.,0.],[-1.,1.], [ 0.,-1],[ 0.,0],[ 0.,1.], [1.,-1.],[1.,0],[1.,1.]])

    #------------------------------------------------------------------------------------

    def reset(self):
        init_s, s0 = self.env.reset()        
        return self.features(s0)
    #------------------------------------------------------------------------------------

    def step(self, a):
        """
        a: (N, 9) in (-1,-1); (-1,0); ...; (1,1)
        """
        action = np.zeros((len(a), 2))
        for i in range(len(a)):
            action[i] = self.actions[a[i]]

        state, reward, done = self.env.step(action, dt=1/40.)
        state = self.features(state)
        return state, reward, done, ""
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

        n1    = R / (R_len + eps)
        n2    = np.zeros_like(n1)
        n2[:, 0] = n1[:, 1];   n2[:, 1] = -n1[:, 0]

        K     = t_vel - n1*(n1*t_vel).sum(axis=-1).reshape(-1,1)

        state = np.hstack([
            v_len,
            (vel*dir1).sum(axis=-1).reshape(-1,1),
            (vel*dir2).sum(axis=-1).reshape(-1,1),
            whe,

            np.tanh( R_len / dist_scale ),
            np.linalg.norm(t_vel, axis=-1, keepdims=True),
            V_len,            
            np.linalg.norm(t_vel, axis=-1, keepdims=True),
            (V * n1).sum(axis=-1).reshape(-1, 1),
            (V * n2).sum(axis=-1).reshape(-1, 1),
            (t_vel * dir1).sum(axis=-1).reshape(-1, 1),
            (t_vel * dir2).sum(axis=-1).reshape(-1, 1),
            (K * dir1).sum(axis=-1).reshape(-1, 1),
            (K * dir2).sum(axis=-1).reshape(-1, 1),
        ])
        return state
    #------------------------------------------------------------------------------------

def main():
    """ Easy launch of the environment. """   
    nS, nA, N = 15, 9, 2

    env = MultiEnvironment(N)

    #env.reset()
    #action = np.random.randint(0,9, size=(2,))
    #state, reward, done, _ = env.step(action)
    #print(state)

    dqn = DQN( )
    
    dqn.params(                  
        ticks      = 200,            
        decays     = 500,       # number of episodes to decay eps1 - > eps2
        update     = 100,       # target model update rate (in frames = time steps)                         
        capacity   = 100_000,   # memory size
        reset    = False,            # reset i-th agent in muli-agent mode when done
        loss     = 'mse',       # loss function (mse, huber)
        optim    = 'sgd',       # optimizer (sgd, adam)
        lm       = 0.001,       # learning rate           
    )
    
    model = MLP(input=nS, output=nA,  hidden=[256, 64])  
    torch.save(model.state_dict(), "models/model_01.pt") 
    dqn.view(ymin=-200, ymax=0)

    print(dqn.params)
    dqn.init(env, model, nS=nS, nA=nA)
    dqn.multi_agent_training(3000)
    dqn.plot(f"best: {dqn.best.reward:7.1f}")

    state = {
        'model':  dqn.best.model.state_dict(),
        'reward': dqn.best.reward,
    }
    torch.save(state, "models/model_01.pt") 

#-------------------------------------------------------------------------------

main()
