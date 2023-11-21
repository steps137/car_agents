import numpy as np
import torch, torch.nn as nn
from .memory_buf import MemoryBufXY
from .ai_random  import AI_Random
from .ai_greedy  import AI_Greedy
from .ai_phys    import AI_Phys

from qunet import Config, Change, MLP, CNN, Data, Trainer, ModelState, Callback

CFG = Config(    
    nX         = 6,
    nY         = 2,
    capacity   = 1024,
    batch_size = 32,
    waiting    = 2,    # waiting time before data collection starts
)

class Model(nn.Module):
    def __init__(self, *arg, **kvargs):
        super().__init__()
        self.mlp = MLP(input=CFG.nX, hidden=10, output=CFG.nY)

    def forward(self, x):                     # (B,1,28,28) or (B,28,28)
        """ model logit """
        y = self.mlp(x)
        return y
    #----------------------------------------------------------------------------

    def training_step(self, batch, batch_id):
        """ 
        Called by the trainer during the training step 
        X = np.hstack([self.act_prv, vn_prv, vd_prv, vxd_prv, whe_prv])# (N,2+4)
        Y = np.hstack([vel_new, d1_prv, d2_prv, vel_prv, vel_prv_prv]) # (N,3*5)        
        """
        x, y = batch
        v_new, d1_prv, d2_prv, v_prv  = y[:,:3], y[:,3:6], y[:,6:9], y[:,9:]

        y_pred = self(x)
        v_pred = y_pred[:,:1]*d1_prv + y_pred[:,1:]*d2_prv

        loss_v = ((v_pred.detach().norm(dim=1) - v_new.norm(dim=1))**2).mean()

        loss   = ((v_pred-v_new)**2).sum(-1).mean()
        return {'loss': loss, 'loss_v': loss_v  }
    #----------------------------------------------------------------------------

    def predict_step(self, batch, batch_id):
        """ 
        Required when using the predict method 
        X = np.hstack([self.act_prv, vn_prv, vd_prv, vxd_prv, whe_prv])# (N,2+4)
        Y = np.hstack([vel_new, d1_prv, d2_prv, vel_prv, vel_prv_prv]) # (N,3*5)        
        """
        x, y = batch
        v_new, d1_prv, d2_prv, v_prv, v_prv_prv  = y[:,:3],y[:,3:6],y[:,6:9],y[:,9:12],y[:,12:15]
        
        with torch.no_grad():
            y_pred = self(x)             
            return {'y': y_pred, 'x': x, 
                    'v_new':        v_new.norm(dim=-1),
                    'dv_prv':       (v_prv-v_prv_prv).norm(dim=-1), 
                    'dv_new':       (v_new-v_prv)    .norm(dim=-1), 
                    'dv_new_d_prv': ((v_new-v_prv)*d1_prv).sum(dim=-1),
                    'd1_prv_norm' :  d1_prv.norm(dim=-1) }

#================================================================================

class AI_EnvModel:
    def __init__(self, epochs=500) -> None:
        self.epochs   = epochs
        self.was_test = False

    def reset(self, init_state, state):
        """ Receives initial state """
        #elf.agent     = AI_Greedy(3)
        self.agent     = AI_Random()
        self.agent.reset(init_state, state)        

        self.memo  = MemoryBufXY(CFG.capacity, CFG.nX, 15)
        self.model = Model()
        self.trainer = Trainer(self.model, score_max=False)
        self.trainer.set_optimizer( torch.optim.Adam(self.model.parameters(), lr=1e-3) )

        self.vel_prv = None           # velocity value at the previous step
        self.dir_prv = None
        self.whe_prv = None
        self.act_prv = None        
        self.vel_prv_prv = None
                
        self.tot_time = 0
    #---------------------------------------------------------------------------

    def step(self, state, reward=None, done=None):
        """
        Receives state and reward, returns simple actions.
        """
        action = self.agent.step(state, reward, done)
        self.tot_time += state['dt']

        vel, dir, whe = state['vel'], state['dir'], state['wheels']

        if self.tot_time > CFG.waiting and not self.was_test:        
            if self.vel_prv is not None:
                self.add_data(vel)
                if self.memo.count > 2*CFG.batch_size:
                    if self.epochs > 0:
                        self.train()
                    else:
                        self.test()

        self.vel_prv_prv = self.vel_prv.copy() if self.vel_prv is not None else vel.copy()
        self.vel_prv = vel.copy()
        self.dir_prv = dir.copy()
        self.whe_prv = whe.mean(axis=-1).copy()
        self.act_prv = action.copy()
        if not self.was_test:
            print(f"{np.linalg.norm(self.vel_prv, axis=-1).mean():.3f}")

        return action
    #---------------------------------------------------------------------------

    def add_data(self, vel_new, v_max=20, eps=1e-8):
        vel_new       = vel_new            / v_max
        vel_prv      = self.vel_prv      / v_max
        vel_prv_prv = self.vel_prv_prv / v_max

        d1_prv = self.dir_prv
        d2_prv = np.zeros((len(d1_prv), 3))           # (-d_y, d_x, 0)
        d2_prv[:, 0] = -d1_prv[:, 1]
        d2_prv[:, 1] =  d1_prv[:, 0]

        vel_new_unit = vel_new  / (np.linalg.norm(vel_new, axis=-1, keepdims=True) + eps)
        idx = (d1_prv * vel_new_unit).sum(axis=-1)  <= 1

        if np.sum(idx) > 0:
            Y = np.hstack([vel_new, d1_prv, d2_prv, vel_prv, vel_prv_prv]) # (N, 3*5)
            Y = Y[idx]

            vn_prv  = np.linalg.norm(vel_prv, axis=-1).reshape(-1,1)
            vd_prv  = (self.vel_prv * d1_prv).sum(axis=-1).reshape(-1,1)
            vxd_prv = (d1_prv[:,0] * vel_prv[:,1] - d1_prv[:,1] * vel_prv[:,0]).reshape(-1,1)
            whe_prv = self.whe_prv.reshape(-1,1)

            X = np.hstack([self.act_prv, vn_prv, vd_prv, vxd_prv, whe_prv]) # (N,2+1+1+1+1)
            #X = np.hstack([vxd])   # (N, 2+1+1+1)
            #X[:,0:2]  = np.random.randn(X.shape[0], 2)         # random actions
            #X[:,2:4]  = np.random.randn(X.shape[0], 2)         
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
        self.epochs -= 1
    #---------------------------------------------------------------------------

    def test(self):
        self.was_test = True

        self.trainer.plot()    
        self.plot()
    #---------------------------------------------------------------------------

    def plot(self):        
        import matplotlib.pyplot as plt

        data = Data( (self.memo.X,  self.memo.Y),  batch_size = 32)
        res = self.trainer.predict(self.trainer.best.score_model, data)
        x, y = res['x'].numpy(),  res['y'].numpy()
        v_new  = res['v_new'].numpy()
        dv_prv = res['dv_prv'].numpy()
        dv_new = res['dv_new'].numpy()
        dv_new_d_prv = res['dv_new_d_prv'].numpy()

        fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(12,3), facecolor ='w')
        ax1.hist(y[:,0], bins=100, label="y1")
        ax1.hist(y[:,1], bins=100, label="y2")
        ax1.grid(ls=":"); ax1.legend() 
    
        ax2.hist(v_new, bins=100, label="v_new")
        ax2.grid(ls=":"); ax2.legend()
    
        ax3.hist(x[:,0], bins=100, label="df")
        ax3.hist(x[:,1], bins=100, label="dw")
        ax3.grid(ls=":"); ax3.legend()
        plt.show()
            
        fig, axs = plt.subplots(2,2, figsize=(12,12), facecolor ='w')
        (ax1,ax2,ax3,ax4) = axs.flatten()
        ax1.scatter(dv_new, dv_prv, s=3);  
        ax1.set_xlabel('dv(t)');  ax1.set_ylabel('dv(t+1)'); ax1.grid(ls=":")

        ax2.hist(dv_new, bins=100)
        ax2.set_xlabel('|v(t+1)-v(t)|');  ax2.grid(ls=":")

        ax3.scatter(x[:,0], dv_new_d_prv, s=3)
        ax3.set_xlabel('df(t) = action[0]');     ax3.set_ylabel(r'$(v_{t+1}-v_t)*d_t$');  ax3.grid(ls=":")

        ax4.scatter(x[:,1], dv_new_d_prv, s=3)
        ax4.set_xlabel('dw(t) = action[1]');  ax4.set_ylabel(r'$(v_{t+1}-v_t)*d_t$');  ax4.grid(ls=":")
        plt.show()
        
        self.stat(x, ['df', 'dw', 'vn', 'vd', 'vxd', 'whe'])
        print(f"d1_prv_norm: {res['d1_prv_norm'].numpy().min()} ... {res['d1_prv_norm'].numpy().max()}")
    #---------------------------------------------------------------------------

    def stat(self, x, names):
        n = x.shape[-1]
        print("====== x:", {x.shape}, "=======")
        for i in range(n):
            print(f"{i:2d}. {names[i]:5s}  {x[:,i].mean():7.4f} ± {x[:,i].std():7.4f}  [{x[:,i].min():7.4f} ... {x[:,i].max():7.4f}]")
        print("corr:")
        cor = np.corrcoef( [x[:, i]  for i in range(n)] )
        for j in range(n):        
            print( f"{names[j]:5s}", ", ".join([ f"{cor[j,i]:6.3f}" for i in range(n) ]))        
