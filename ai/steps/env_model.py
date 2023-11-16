import numpy as np
import torch, torch.nn as nn
from .memory_buf import MemoryBufXY
from .ai_random  import AI_Random
from .ai_greedy  import AI_Greedy
from .ai_phys    import AI_Phys

from qunet import Config, Change, MLP, CNN, Data, Trainer, ModelState, Callback

CFG = Config(
    capacity   = 256,
    batch_size = 32,
    nX         = 6,
    nY         = 2,
)

class Model(nn.Module):
    def __init__(self, *arg, **kvargs):
        super().__init__()
        self.mlp = MLP(input=CFG.nX, hidden=10, output=CFG.nY)

    def forward(self, x):                     # (B,1,28,28) or (B,28,28)
        """ model logit """
        y = self.mlp(x)
        return y

    def training_step(self, batch, batch_id):
        """ Called by the trainer during the training step """
        x, y_true = batch
        v_new, d1_old, d2_old, v_old  = y_true[:,:3],  y_true[:,3:6], y_true[:,6:9], y_true[:,9:]

        y_pred = self(x)
        vp = y_pred[:,:1]*d1_old + y_pred[:,1:]*d2_old

        loss   = ((vp-v_new)**2).sum(-1).mean()
        return {'loss': loss  }

    def predict_step(self, batch, batch_id):
        """ Required when using the predict method """
        x, y = batch
        v, d1, d2 = y[:,:3],  y[:,3:6], y[:,6:]
        with torch.no_grad():
            y_pred, v_true = self(x), v.norm(dim=-1)
            return {'y': y_pred, 'v_true': v_true, 'x': x }

#---------------------------------------------------------------------------------

class AI_EnvModel:
    """
    """
    def __init__(self) -> None:
        pass

    def reset(self, init_state, state):
        """ Receives initial state """
        #elf.agent     = AI_Greedy(3)
        self.agent     = AI_Random()
        self.agent.reset(init_state, state)        

        self.memo  = MemoryBufXY(CFG.capacity, CFG.nX, 12)
        self.model = Model()
        self.trainer = Trainer(self.model)
        self.trainer.set_optimizer( torch.optim.Adam(self.model.parameters(), lr=1e-3) )

        self.prev_vel = None           # velocity value at the previous step
        self.prev_dir = None
        self.prev_whe = None
        self.prev_act = None        
    #---------------------------------------------------------------------------

    def step(self, state, reward=None, done=None):
        """
        Receives state and reward, returns simple actions.
        """
        action = self.agent.step(state, reward, done)

        vel, dir, whe = state['vel'], state['dir'], state['wheels']
        if self.prev_vel is not None:
            self.add_data(vel)
            if self.memo.count > self.memo.capacity:
                self.train()

        self.prev_vel = vel.copy()
        self.prev_dir = dir.copy()
        self.prev_whe = whe.mean(axis=-1)
        self.prev_act = action.copy()
        print(f"{np.linalg.norm(self.prev_vel, axis=-1).mean():.3f}")

        return action
    #---------------------------------------------------------------------------

    def add_data(self, vel_new, v_max=20, eps=1e-8):
        vel_new = vel_new / v_max
        v  = self.prev_vel / v_max
        d1 = self.prev_dir
        d2 = np.zeros((len(d1), 3))                        # (-d_y, d_x, 0)
        d2[:, 0] = -d1[:, 1]
        d2[:, 1] =  d1[:, 0]

        vel_new_unit = vel_new  / (np.linalg.norm(vel_new, axis=-1, keepdims=True) + eps)
        idx = (d1 * vel_new_unit).sum(axis=-1)  < 1

        if np.sum(idx) > 0:
            Y = np.hstack([vel_new, d1, d2, v])            # (N, 3+3+3)
            Y = Y[idx]

            vn = np.linalg.norm(v, axis=-1).reshape(-1,1)
            vd = (self.prev_vel * d1).sum(axis=-1).reshape(-1,1)
            vxd = (d1[:, 0] * v[:, 1] - d1[:, 1] * v[:, 0]).reshape(-1,1)
            whe = self.prev_whe.reshape(-1,1)

            X = np.hstack([self.prev_act, vn, vd, vxd, whe])   # (N, 2+1+1+1+1)
            #X = np.hstack([vxd])   # (N, 2+1+1+1)
            #X[:,0:2]  = np.random.randn(len(v), 2)         # random actions
            #X[:,2:4]  = np.random.randn(len(v), 2)         
            #X  = np.random.randn(X.shape[0], X.shape[1])   # all random
            X = X[idx]
            for i in range(len(X)):
                self.memo.add(X[i], Y[i])
    #---------------------------------------------------------------------------

    def train(self):
        X, Y = self.memo.get(CFG.batch_size)
        self.trainer.data(  trn=Data((X,Y),  batch_size = CFG.batch_size)  )
        self.trainer.best(copy = True)
        self.trainer.fit(1, show_stat=False, monitor=['loss'])

    #---------------------------------------------------------------------------