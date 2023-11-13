import numpy as np
import torch, torch.nn as nn
from   .memory_buf import MemoryBufXY
from qunet import Config, Change, MLP, CNN, Data, Trainer, ModelState, Callback

CFG = Config(
    capacity   = 128,
    batch_size = 32,
    nX         = 5,  # 5
    nY         = 2,
)

class Model(nn.Module):
    def __init__(self, *arg, **kvargs):
        super().__init__()
        self.mlp = MLP(input=CFG.nX, output=CFG.nY)

    def forward(self, x):                     # (B,1,28,28) or (B,28,28)
        """ model logit """
        y = self.mlp(x)
        return y

    def training_step(self, batch, batch_id):
        """ Called by the trainer during the training step """
        x, y_true = batch
        v, d1, d2 = y_true[:,:3],  y_true[:,3:6], y_true[:,6:]

        y_pred = self(x)
        vp = y_pred[:,:1]*d1 + y_pred[:,1:]*d2

        loss   = ((vp-v)**2).sum(-1).mean()
        return {'loss': loss  }

#---------------------------------------------------------------------------------

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
        self.beta    = 0.99            # EMA speed averaging

        self.prev_vel = None           # velocity value at the previous step
        self.prev_dir = None
        self.prev_act = None

        self.ticks     = 10            # for random policy
        self.cur_tick  = 0        
        self.actions   = None 

        self.memo  = MemoryBufXY(CFG.capacity, CFG.nX, 9)
        self.model = Model()
        self.trainer = Trainer(self.model)
        self.trainer.set_optimizer( torch.optim.Adam(self.model.parameters(), lr=1e-3) )
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

        if self.prev_vel is not None:
            self.add_data(vel)
            if self.memo.count > self.memo.capacity:
                self.train()

        self.prev_vel = vel.copy()
        self.prev_dir = dir.copy()
        self.prev_act = action.copy()

        return action
    #---------------------------------------------------------------------------

    def add_data(self, vel_new, v_max=30, eps=1e-8):
        vel_new /= v_max
        v  = self.prev_vel / v_max
        d1 = self.prev_dir
        d2 = np.zeros((len(d1), 3))               # (-d_y, d_x, 0)
        d2[:, 0] = -d1[:, 1]
        d2[:, 1] =  d1[:, 0]

        vel_new_unit = vel_new  / (np.linalg.norm(vel_new, axis=-1, keepdims=True) + eps)
        idx = (d1 * vel_new_unit).sum(axis=-1)  < 1

        if np.sum(idx) > 0:
            Y = np.hstack([vel_new, d1, d2])  # (N, 3+3+3)
            Y = Y[idx]

            vn = np.linalg.norm(v, axis=-1).reshape(-1,1)
            vd = (self.prev_vel * d1).sum(axis=-1).reshape(-1,1)
            vxd = (d1[:, 0] * v[:, 1] - d1[:, 1] * v[:, 0]).reshape(-1,1)

            #X = np.hstack([self.prev_act, vn, vd, vxd])   # (N, 2+1+1+1)
            #X = np.hstack([vxd])   # (N, 2+1+1+1)
            X  = np.random.randn(len(v), 5)
            X = X[idx]
            for i in range(len(X)):
                self.memo.add(X[i], Y[i])
    #---------------------------------------------------------------------------

    def train(self):
        X, Y = self.memo.get(CFG.batch_size)
        self.trainer.data(  trn=Data((X,Y),  batch_size = CFG.batch_size)  )
        self.trainer.fit(1, show_stat=False)

    #---------------------------------------------------------------------------

    def policy(self, X, x, v, dir, eps=1e-8):
        return self.policy0(X, x, v, dir, eps)
        #return self.policy1(X, x, v, dir, eps)
    #---------------------------------------------------------------------------

    def policy0(self, X, x, v, dir, eps=1e-8):      
        if self.cur_tick % self.ticks == 0 or self.actions is None:
            self.actions = 2. * np.random.randint(0, 2, size = (len(v), 2)) - 1.
        self.cur_tick += 1
        return self.actions
    #---------------------------------------------------------------------------

    def policy1(self, X, x, v, dir, eps=1e-8):
        """ From point x to point X with velocity v """
        action = np.ones( (len(v), 2) )

        r = X - x                                             # radius vector to target
        dist = np.linalg.norm(r, axis=-1)                     # distance to target
        n = r / (dist.reshape(-1,1)+eps)                      # unit vectors to distance

        vel = np.linalg.norm(v, axis=-1)                      # velocity
        v /= (vel.reshape(-1,1)+eps)                          # unit vector of velocity

        si = n[:,0] * dir[:,1] - n[:,1] * dir[:,0]            # rotate
        co = (dir * n).sum(axis=-1)                           # cos with target

        action[vel > 10, 0] = 0                               # speed limit
        action[si < 0, 1] = -1
        action[(dist < 5) & (co < 0.3), 1] *= -1

        if self.t_tot > 3:
            self.times[ self.v_avr < 0.5 ] = -3.
            action[self.times < 0, 0] = -1
            action[self.times < 0, 1] =  0

        return action
