import numpy as np
import torch
from   torch import tensor

class AI_Phys:
    def __init__(self, f_dir=True, f_car=True, f_seg=True) -> None:
        self.car_R          = 2.

        self.f_dir=f_dir; self.f_car=f_car; self.f_seg=f_seg

        self.car_tar_lm_r   = 1.                
        self.car_tar_v_max  = 20.

        self.car_car_lm_r   = 1 #1      
        self.car_car_lm_v   = 1 #1
        self.car_car_mu     = 1.
        self.car_car_a      = 1.

        self.car_seg_lm     = 1.
        self.car_seg_lm_v   = 2.
        self.car_seg_mu     = 1.
        self.car_seg_a      = 1

    #---------------------------------------------------------------------------

    def reset(self, init_state, state):
        """ Receives initial state """
        self.ai_kind = init_state['ai_kind']
        self.kinds   = init_state['kind']
        self.N       = init_state['n_cars']
        self.seg_p1  = tensor(init_state['string'][:,0,:])  # begin and end
        self.seg_p2  = tensor(init_state['string'][:,1,:])  # of segment

        self.t_tot   = 0        
        self.times   = np.zeros((self.N,))  
        self.v_avr   = np.zeros((self.N,))                # average agent speed
        self.beta    = 0.99                               # EMA speed averaging     

        self.info    = {}   
    #---------------------------------------------------------------------------

    def step(self, state, reward=None, done=None):
        """
        Receives state and reward, returns simple actions.
        my для простоты не используем (все со всеми)
        """
        pos, vel, dir = tensor(state['pos']), tensor(state['vel']), tensor(state['dir'])
        tar_pos, tar_vel  = tensor(state['target_pos']), tensor(state['target_vel'])

        f1, f2, f3 = self.features(pos, vel, tar_pos, tar_vel, eps=1e-8)
        f = torch.zeros_like(f1)
        if self.f_dir:  
            f = f + f1
        if self.f_car:  
            f = f + f2
        if self.f_seg:  
            f = f + f3

        desired_dir =  f #* state['dt']          # desired direction

        self.times += state['dt']
        self.t_tot += state['dt']
        
        action = np.zeros( (len(pos), 2) )

        self.v_avr = self.v_avr*self.beta + (1-self.beta)*np.linalg.norm(vel, axis=-1)

        action = self.policy(desired_dir, vel, dir, pos, tar_pos).numpy()        
                
        #self.info = { 'vec': torch.hstack([f.clone(), f1.clone(), f2.clone(), f3.clone()]).numpy()}
        self.info = { 'vec': f.clone() }
        return action
    #---------------------------------------------------------------------------

    def features(self, pos, vel, tar_pos, tar_vel, eps=1e-6):
        """
        Create features for Decision Model
        return forces f1, f2, f3: (N,3)
        """
        f1 = self.force_target (pos, vel, tar_pos, tar_vel, eps)
        f2 = self.force_car    (pos, vel, eps)
        f3 = self.force_segment(pos, vel, eps)        

        return f1, f2, f3                                   # (N,3)
    #---------------------------------------------------------------------------

    def force_target(self, pos, vel, tar_pos, tar_vel, eps):
        """ 
        Args:   pos, vel, tar_pos: (N,3)
        Return: force to target: (N,3)        
        """
        N = len(pos)
        r = tar_pos - pos                                   # (N,3) from car to target pos        
        dist = r.norm(dim=-1, keepdim=True)                 # (N,3) dist to target        
        n = r / ( dist + eps)                               # unit vector to target
        
        k = tar_vel - n * (n * tar_vel).sum(-1).view(N,1)
        k_len = k.norm(dim=-1, keepdim=True)

        v = torch.empty(N,1).fill_(self.car_tar_v_max)      # max speed
        v = torch.where( v > k_len, v, k_len)
        k = k / (v + eps)
    
        f = n * ( (1-(k*k).sum(-1).view(N,1)).abs() )**0.5 + k          
        #f = n      
        return self.car_tar_lm_r * f
    #---------------------------------------------------------------------------

    def force_car(self, pos, vel, eps):
        """
        Args:   pos, vel: (N,3)
        Return: force from all cars: (N,3)        
        """
        N = len(pos)
        # положение i-го агента относительно j-того
        rij = pos.view(N,1,3) - pos.view(1,N,3)             # (N,N,3): vec  r_{ij}=r_i-r_j
        vij = vel.view(N,1,3) - vel.view(1,N,3)             # (N,N,3): relative vel i from  j-th

        dij = rij.norm(dim=-1)                              # (N,N): dist d_{ij}
        dij.fill_diagonal_(1/eps)                           # no interactions with yourself
        dij = dij.view(N,N,1)

        nij = rij / ( dij + eps)                             # nij
        nvij = (nij*vij).sum(dim=-1).view(N,N,1)

        # r x [v x r]
        f2 = nij * nvij  - vij
        f2 = f2 / (f2.norm(dim=-1).view(N,N,1) + eps)     

        mu, a = self.car_car_mu, self.car_car_a

        dij = dij / (2*self.car_R)
        #w = torch.exp(-mu * (dij-1)) / ((dij-1)**2 + eps)
        w = (1+np.exp(mu*(1-a)))/(1+torch.exp(mu*(dij-a)))

        f  = -self.car_car_lm_r * nij + self.car_car_lm_v * f2
        f = f * (nvij < 0)

        fj = (w * f).sum(dim=0)

        return fj                                           # (N,3)
    #---------------------------------------------------------------------------

    def force_segment(self, pos, vel, eps):
        # force from segments:        
        r = self.seg_r(pos, self.seg_p1, self.seg_p2, eps)# (N,M,3) from cars to segments
        N, M = r.shape[0],  r.shape[1]
        d = r.norm(dim=-1, keepdim=True)
        n = r / (d+eps)

        f1 = -n
        f2 = vel.view(N,1,3) - n*(n*vel.view(N,1,3)).sum(-1).view(N,M,1)
        f2 = f2 / (f2.norm(dim=-1, keepdim=True)+eps)

        f = self.car_seg_lm * (f1 + self.car_seg_lm_v*f2)

        mu, a = self.car_seg_mu, self.car_seg_a
        d = d / self.car_R
        w = (1+np.exp(mu*(1-a)))/(1+torch.exp(mu*(d-a)))                        

        return (f*w).sum(1)
        #return torch.zeros_like(vel)
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

    def policy(self,  desired_dir, vel, dir, pos, tar_pos, eps=1e-8):
        """ 
        """
        action = torch.ones( (len(vel), 2) )

        r = tar_pos - pos                          # radius vector to target
        dist =  r.norm(dim=-1)                     # distance to target
        r /= (dist.view(-1,1) + eps)               # unit vector to target
        speed = vel.norm(dim=-1)

        n = desired_dir / (desired_dir.norm(dim=-1).view(-1,1) + eps)  # unit vectors to desired

        si = dir[:,0] * n[:,1] - dir[:,1] * n[:,0]  # rotate
        co = (dir * r).sum(axis=-1)                 # cos with target

        action[si < 0, 1] = -1
        action[(speed > 5) & (dist < 10) & (co < 0.3), 1] *= -1        

        #action[(speed > 10), 0] = 0

        if self.t_tot > 3:
            self.times[ self.v_avr < 0.5 ] = -3.
            action[self.times < 0, 0] = -1
            action[self.times < 0, 1] =  0
        
        return action