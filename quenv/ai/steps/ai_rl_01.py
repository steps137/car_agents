import numpy as np
import torch
from qunet import MLP


class AI_RL01:
    def __init__(self) -> None:                                 # tar_seg_collision:             True                   False
        # reset_steps=1000, steps=100_000                         moving_targets:        False            True          True 
        # Phys(f_car=False, f_seg=False) на цель                                     # [2.62|3.62]    [3.34|4.50]    [7.99|10.1]
        # Phys(f_car=False, f_seg=False) упреждение (v_max=20)                       # [2.62|3.65]    [3.10|4.16]    [5.13|6.53]

        #nS, nA, hidden = 8, 5, [256, 64]                                           #  tar_seg_collision:             True                   False
        #                                                         moving_targets:        False            True          True
        #state = torch.load("models/01/model_v0_s8a5_0163_r-82.88_e8080.pt")         # [1.95|3.88]    [3.76|]         [7.75|10.11]

        nS, nA, hidden = 16, 5, [256, 64]             # учил на tar_seg_collision = True (!)                            
        #state = torch.load("models/01/model_v20p50_s16a5_0149_r-83.02_e5115.pt")
        #state = torch.load("models/01/model_v20p50_s16a5_0149_r-83.02_e5115.pt")     # [2.07|]         [2.68|]       [4.37|]        
        #state = torch.load("models/01/model_v20p50_s16a5_0149_r-84.79_e8819.pt")     # [2.07|3.85]     [2.58|4.51]   [4.07|6.00]        
        #state = torch.load("models/01/model_v20p30_s16a5_0168_r-83.03_e9948.pt")      #                 [3.42|]

        #state = torch.load("models/01/model_v20p50_s16a5_0561_r-98.53_e2297.pt")  #                     [2.66|]  len: 106   ticks: 87 ±   2
        #state = torch.load("models/01/model_v20p50_s16a5_0567_r-95.43_e2339.pt")  #                     [5.82|]  len: 233  ticks:133 ±   6
        #state = torch.load("models/01/model_v20p50_s16a5_0569_r-94.32_e2348.pt")  #                     [3.85|]  len: 154  ticks:102 ±   3
        #state = torch.load("models/01/model_v20p50_s16a5_0560_r-99.32_e2274.pt")  #                   [|]         len: 177  ticks:224  ±   3

        state = torch.load("models/01/a/model_v20p50_s16a5_0089_r-78.65_e7348.pt")  #                   [|]         len: 177  ticks:224  ±                                                                   
    
        print('reward:', state['reward'])
        self.model = MLP(input=nS, output=nA,  hidden=hidden) 
        self.model.load_state_dict(state['model'])
        print(self.model)

        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()
        with torch.no_grad():
            print(self.model(torch.zeros(1,nS)))            

        #self.actions = np.array([[-1.,-1.],[-1.,0.],[-1.,1.], [ 0.,-1],[ 0.,0],[ 0.,1.], [1.,-1.],[1.,0],[1.,1.]])
        self.actions = np.array([[-1.,-1.], [-1.,1.], [1.,-1.], [1.,0], [1.,1.]])

    def reset(self, init_state, state):
        """ Receives initial state """
        self.ai_kind = init_state['ai_kind']
        self.kinds   = init_state['kind']
        self.N       = init_state['n_cars']
        self.seg_p1  = torch.tensor(init_state['string'][:,0,:])  # begin and end
        self.seg_p2  = torch.tensor(init_state['string'][:,1,:])  # of segment
    #---------------------------------------------------------------------------

    def step(self, state, reward, done):
        """
        Receives state and reward, returns simple actions.
        """    
        state = self.features(state)
        with torch.no_grad():
            state = torch.tensor(state, device=self.device)
            Q = self.model(state)
            a = torch.argmax(Q, dim=-1).cpu().numpy() 

        action = np.zeros((len(a), 2))
        for i in range(len(a)):
            action[i] = self.actions[a[i]]        
        
        return action
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

        state1 = np.hstack([
            v_len,
            (vel*dir1).sum(axis=-1).reshape(-1,1),
            (vel*dir2).sum(axis=-1).reshape(-1,1),
            whe,

            (tar1*dir1).sum(axis=-1).reshape(-1,1),
            (tar1*dir2).sum(axis=-1).reshape(-1,1),

            np.tanh( R_len / dist_scale ),
        ])

        state2 = np.hstack([
            np.linalg.norm(t_vel, axis=-1, keepdims=True),
            V_len,

            (V * tar1).sum(axis=-1).reshape(-1, 1),
            (V * tar2).sum(axis=-1).reshape(-1, 1),

            (t_vel * dir1).sum(axis=-1).reshape(-1, 1),
            (t_vel * dir2).sum(axis=-1).reshape(-1, 1),

            (K * dir1).sum(axis=-1).reshape(-1, 1),
            (K * dir2).sum(axis=-1).reshape(-1, 1),
        ])
        return np.hstack([state1, state2])
    #------------------------------------------------------------------------------------
