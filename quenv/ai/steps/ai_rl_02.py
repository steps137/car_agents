import numpy as np
import torch
from qunet import MLP


class AI_RL02:
    def __init__(self) -> None:                         # tar_seg_collision:             True                   False

        nS, nA = 16+7*2, 5             # учил на tar_seg_collision = True (!)
        state = torch.load("models/02/model_v20p50_s30a5_0604_r-130.22_e7140.pt")  #

        print('reward:', state['reward'])
        self.model = MLP(input=nS, output=nA,  hidden=[256, 64])
        self.model.load_state_dict(state['model'])
        print(self.model)

        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        #self.actions = np.array([[-1.,-1.],[-1.,0.],[-1.,1.], [ 0.,-1],[ 0.,0],[ 0.,1.], [1.,-1.],[1.,0],[1.,1.]])
        self.actions = np.array([[-1.,-1.], [-1.,1.], [1.,-1.], [1.,0], [1.,1.]])

    def reset(self, init_state, state):
        """ Receives initial state """
        self.ai_kind = init_state['ai_kind']
        self.kinds   = init_state['kind']
        self.N       = init_state['n_cars']
        self.seg_p1  = init_state['string'][:,0,:]  # begin and end
        self.seg_p2  = init_state['string'][:,1,:]  # of segment
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
        state_dir = self.features_dir(state, vel_scale, dist_scale, eps)
        state_seg = self.features_seg(state, vel_scale, dist_scale, eps)
        return np.hstack([state_dir, state_seg])
    #------------------------------------------------------------------------------------

    def features_dir(self, state, vel_scale=20, dist_scale=10, eps=1e-6):
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

    #-----------------------------------------------------------------------

    def features_seg(self, state, vel_scale=20, dist_scale=10, eps=1e-6):
        """ force from segments """
        pos, vel, dir1 = state['pos'], state['vel'], state['dir']
        dir2= np.zeros_like(dir1)
        dir2[:, 0] = dir1[:, 1];   dir2[:, 1] = -dir1[:, 0];

        r, t = self.seg_r(pos, self.seg_p1, self.seg_p2, eps)# (N,M,3) from cars to segments
        N, M = r.shape[0],  r.shape[1]

        D = np.linalg.norm(r, axis=-1)                       # (N,M)
        idx = np.argmin(D, axis=1)

        n  = r / (D.reshape(N,M,1)+eps)                      # (N,M,3)
        nv = (n*vel.reshape(N,1,3)).sum(-1)                  # (N,M)
        Q  = vel.reshape(N,1,3) - n*nv.reshape(N,M,1)        # (N,M,3)

        f1 = (n*dir1.reshape(N,1,3)).sum(-1)
        f2 = (n*dir2.reshape(N,1,3)).sum(-1)
        f3 = (Q*dir1.reshape(N,1,3)).sum(-1)
        f4 = (Q*dir2.reshape(N,1,3)).sum(-1)

        idx = np.argsort(D, axis=1)[:, :2]

        state = np.hstack([
            np.take_along_axis(t,                           axis=1, indices=idx),
            np.take_along_axis( np.tanh( D / dist_scale ),  axis=1, indices=idx),
            np.take_along_axis(nv,                          axis=1, indices=idx),
            np.take_along_axis(f1,                          axis=1, indices=idx),
            np.take_along_axis(f2,                          axis=1, indices=idx),
            np.take_along_axis(f3,                          axis=1, indices=idx),
            np.take_along_axis(f4,                          axis=1, indices=idx),
        ])
        return state
    #------------------------------------------------------------------------------------

    def seg_r(self, x, p1, p2, eps):
        """
        Radius vectors and distances from point x to segments (p1,p2); there are N points and M segments:
        x: (N,3);  p1, p2: (M,3);
        return r: (N,) distance to nearest segment
        """
        N, M   = len(x), len(p1)
        x      = x.reshape(-1, 1, 3)                          # (N,1,3)
        p1, p2 = p1.reshape(1, -1, 3), p2.reshape(1, -1, 3)   # (1,M,3)
        p21     = p2 - p1                                     # (1,M,3)
        tt = ((x-p1)*p21).sum(-1) / ((p21*p21).sum(-1)+ eps)   # (N,M)
        t = np.broadcast_to(tt.reshape(N,M,1), (N, M, 3))      # (N,M,3)
        r = np.where(t < 0,  p1-x,                            # (N,M,3)
                             np.where( t > 1,  p2-x,
                                               (1-t)*p1+t*p2-x))
        t = np.tanh(3 * (1-np.abs(2*t-1)))                    # (N,M)
        return r, tt                                          # (N,M,3)  (N,M)
