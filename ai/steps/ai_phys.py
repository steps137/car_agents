import numpy as np
import torch

class AI_Phys:
    def __init__(self) -> None:
        self.car_R          = 2.

        self.car_tar_lambda = 10.

        self.car_seg_mu     = 1.
        self.car_seg_a      = 2*self.car_R
        self.car_seg_rate   = 0.5
        self.car_seg_lambda = 10.

        self.car_car_mu     = 10.
        self.car_car_a      = 2*self.car_R
        self.car_car_lambda = 10.

    def reset(self, init_state, state):
        """ Receives initial state """
        self.ai_kind = init_state['ai_kind']
        self.kinds   = init_state['kind']
        self.N       = init_state['n_cars']
        self.seg_p1  = torch.tensor(init_state['string'][:,0,:])  # begin and end
        self.seg_p2  = torch.tensor(init_state['string'][:,1,:])  # of segment

    #---------------------------------------------------------------------------

    def step(self, state, reward):
        """
        Receives state and reward, returns simple actions.
        """
        pos, vel, dir = torch.tensor(state['pos']), torch.tensor(state['vel']), torch.tensor(state['dir'])
        tar_pos  = torch.tensor(state['target_pos'])

        f1, f2, f3 = self.features(pos, vel, tar_pos, eps=1e-8)
        f =     self.car_tar_lambda * f1
        f = f + self.car_seg_lambda * f2
        #f = f + self.car_car_lambda * f3

        vd =  f * state['dt']          # desired velocity

        my = (self.ai_kind == self.kinds)
        action = np.zeros( (len(pos), 5) )
        my_action = self.policy(vd, vel, dir).numpy()
        action[my] = my_action[:]
        return action
    #---------------------------------------------------------------------------

    def features(self, pos, vel, tar_pos, eps=1e-8):
        """
        Create features for Decision Model
        return forces f1, f2: (N,3)
        """
        f1 = self.force_target (pos, vel, tar_pos, eps)
        f2 = self.force_segment(pos, vel, eps)
        f3 = self.force_car    (pos, vel, eps)

        return f1, f2, f3                                   # (N,3)
    #---------------------------------------------------------------------------

    def force_target(self, pos, vel, tar_pos, eps):
        r = tar_pos - pos                                   # (N,3) from car to target pos
        return r/( r.norm(dim=-1, keepdim=True) + eps)

    #---------------------------------------------------------------------------

    def force_car(self, pos, vel, eps):
        """
        pos, vel: (N,3);  return force from all cars: (N,3)
        """
        r = pos.view(len(pos),1,3) - pos.view(1,len(pos),3)   # vec (N,N,3): r_{ij}=r_i-r_j

        z = torch.zeros_like(r)
        z[:,:,-1] = 1.
        f = torch.linalg.cross(z,r)                           # [z x r] - passing cars on the right

        d = r.norm(dim=-1)                                    # dist d_{ij}
        d.fill_diagonal_(1/eps)                               # no interactions with yourself
        d = d.view(len(r),len(r),1)

        mu, a = self.car_car_mu, self.car_car_a
        w = torch.exp(-mu * (d/a-1)) / ((d/a-1)**2 + eps)
        f = w*f

        return 0.5*f.sum(dim=1)                               # (N,3)
    #---------------------------------------------------------------------------

    def force_segment(self, pos, vel, eps):
        # force from segments:
        r = self.seg_r(pos, self.seg_p1, self.seg_p2, eps)# (N,M,3) from cars to segments
        d = r.norm(dim=-1, keepdim=True)
        n = r / (d+eps)

        z = torch.zeros_like(n)
        z[:,:,-1] = 1.
        f = torch.linalg.cross(z,n)                           # [z x r] - passing cars on the right

        mu, a = self.car_seg_mu, self.car_seg_a
        w = torch.exp(-mu * (d/a-1)) / ((d/a-1)**2 + eps)

        f = f - n  # self.car_seg_rate *

        return (f*w).sum(1)
    #---------------------------------------------------------------------------

    def seg_r(self, x, p1, p2, eps):
        """
        Radius vector from point x to segment (p1,p2); there are N points and M segments:
        x: (N,3);  p1, p2: (M,3); return r: (N,M,3) from each point to each segment
        """
        N, M   = len(x), len(p1)
        x      = x.view(-1, 1, 3)                               # (N,1,3)
        p1, p2 = p1.view(1, -1, 3), p2.view(1, -1, 3)           # (1,M,3)
        p21     = p2 - p1                                       # (1,M,3)
        t = ((x-p1)*p21).sum(-1) / ((p21*p21).sum(-1)+ eps)     # (N,M)
        t = t.view(N,M,1).expand(N, M, 3)                       # (N,M,3)

        return torch.where(t < 0,  p1-x,                        # (N,M,3)
                                   torch.where( t > 1,  p2-x,
                                                        (1-t)*p1+t*p2-x))
    #---------------------------------------------------------------------------

    def policy(self, vd, v, dir, eps=1e-8):
        """ From point x to point X with velocity v """
        action = torch.ones( (len(v),5) )

        vdn = vd.norm(dim=-1, keepdim=True)
        ud = vd / (vdn + eps)
        action[:, 2:] = ud.clone()

        vn = v.norm(dim=-1, keepdim=True)
        u  = v / (vn + eps)

        si = (u[:,0] * ud[:,1] - u[:,1] * ud[:,0])  # rotate
        action[si > 0, 1] = -1

        co = (ud * dir).sum(axis=-1)
        action[:,0]  = co
        #action[vn.view(-1) < 5,  0]  = torch.sign() # if the car is parked, you have to go
        action[vn.view(-1) > 25, 0]  = 0. # don't let you accelerate quickly
        return action
