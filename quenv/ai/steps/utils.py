    #---------------------------------------------------------------------------

    def dist_to_seg(self, x, p1, p2, eps):
        """
        Radius vectors and distances from point x to segments (p1,p2); there are N points and M segments:
        x: (N,3);  p1, p2: (M,3); 
        return r: (N,) distance to nearest segment
        """
        N, M   = len(x), len(p1)
        x      = x.reshape(-1, 1, 3)                          # (N,1,3)
        p1, p2 = p1.reshape(1, -1, 3), p2.reshape(1, -1, 3)   # (1,M,3)
        p21     = p2 - p1                                     # (1,M,3)
        t = ((x-p1)*p21).sum(-1) / ((p21*p21).sum(-1)+ eps)   # (N,M)
        t = np.broadcast_to(t.reshape(N,M,1), (N, M, 3))      # (N,M,3)

        r = np.where(t < 0,  p1-x,                            # (N,M,3)
                             np.where( t > 1,  p2-x,
                                               (1-t)*p1+t*p2-x))
        d = np.linalg.norm(r, axis=-1)                        # (N,M)
        return np.min(d, axis=-1)                             # (N,)