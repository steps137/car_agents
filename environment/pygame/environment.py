import numpy as np
import time
import pygame as pg
from .car import Car, CarColors

class Environment:

    REWARD_TARGET  = 100  # reward for a given car for reaching a target point
    REWARD_CAR_CAR = -10  # reward (punishment) for car collision
    REWARD_CAR_SEG = -1   # reward (punishment) for a collision between a car and a segment

    IMG_GROUND = "environment/pygame/img/ground.png"
    IMG_TARGET = "environment/pygame/img/target.png"
    IMG_ICON   = "environment/pygame/img/icon.png"

    def __init__(self, ai=None, n_cars=20, w=100, h=50, d=100, mt2px=10, level=0):
        self.n_cars = n_cars     # number of cars
        self.level  = level      # environment complexity level (0,1,...)

        self.space_w = w         # width  of space in meters
        self.space_h = h         # height of space in meters
        self.space_d = d         # depth  of space in meters
        self.mt2px   = mt2px     # converting meters to pixels

        self.params = {
            'car_collision':        True,  # handle car collisions
            'segment_collision':    True,  # handle collisions with segments
            'all_targets_are_same': False, # the coordinates of all targets are the same
            'show_target_line':     False, # show line from car to target
            'show_actions':         False  # show current car actions
        }

        self.ai     = self.set_ai_dict(ai, n_cars)

        pg.init()
        self.screen = None
        self.clock  = pg.time.Clock()
        self.camera = 0                    # game camera number
        self.camera_dir  = None
        self.camera_beta = 0.9

        self.fps           = None          # real fps in phys: self.calc_fps(...)
        self.prev_time     = None
        self.prev_time_out = None

        self.tot_phys_time = 0
        self.tot_phys_run  = 0

        self.img_ground  = None            # image for ground texture
        self.img_target  = None            # image of target
        self.font        = None
        self.pause       = False

        self.reset()
    #---------------------------------------------------------------------------

    def set_params(self,**kvargs):
        for k, v in kvargs.items():
            if    k == 'car_collision':        self.params['car_collision']        = v
            elif  k == 'segment_collision':    self.params['segment_collision']    = v
            elif  k == 'all_targets_are_same': self.params['all_targets_are_same'] = v
            elif  k == 'show_target_line':     self.params['show_target_line']     = v
            elif  k == 'show_actions':         self.params['show_actions']         = v
            else: print(f"!!! Warning: unknown parameter {k} = {v}")
    #---------------------------------------------------------------------------

    def set_ai_dict(self, ai, n_cars):
        if ai is None:
            self.kinds = np.zeros(shape=(n_cars,), dtype=np.int32)        
        else:
            if not (type(ai) is dict):
                ai = {"car": {'ai': ai, 'num': n_cars} }
            self.n_cars = sum([v['num'] for k,v in ai.items()])
            self.kinds = np.zeros(shape=(self.n_cars,), dtype=np.int32)        
            i = 0
            for kind, (k,v) in enumerate(ai.items()):
                v['kind'] = kind               
                self.kinds[i:i+v['num']] = kind
                i += v['num']
            self.n_cars = i

        return ai
    #---------------------------------------------------------------------------

    def init_state(self):
        seg    = np.empty(shape=(len(self.segs), 2, 3), dtype=np.float32)
        for i, s in enumerate(self.segs):
            seg[i] = s.copy()

        return {
            'space':    np.array([self.space_w, self.space_h, self.space_d], dtype=float),
            'n_cars':   self.n_cars,
            'ai_kind':  0,
            'kind':     self.kinds,
	        'string':   seg
        }
    #---------------------------------------------------------------------------

    def state(self, dt):
        """ Get the current state of the environment """
        pos    = np.empty(shape=(self.n_cars, 3), dtype=np.float32)
        vel    = np.empty(shape=(self.n_cars, 3), dtype=np.float32)
        dir    = np.empty(shape=(self.n_cars, 3), dtype=np.float32)
        wheels = np.empty(shape=(self.n_cars, 2), dtype=np.float32)

        for i, c in enumerate(self.cars):
            pos[i] = c.pos.copy();  vel[i] = c.vel.copy();  dir[i] = c.dir.copy()
            wheels[i][0] = c.w11;   wheels[i][1] = c.w12

        t_pos = np.empty(shape=(self.n_cars, 3), dtype=np.float32)
        t_vel = np.zeros(shape=(self.n_cars, 3), dtype=np.float32)
        for i, target in enumerate(self.targets):
            t_pos[i] = target.copy()

        state = {
            'dt':         dt,
            'pos':        pos,
            'vel':        vel,
            'dir':        dir,
            'wheels':     wheels,
            'target_pos': t_pos,
            'target_vel': t_vel,
        }
        return state
    #---------------------------------------------------------------------------

    def reset(self):
        """ Restart environment """
        self.create_segments        (self.space_w, self.space_h)
        self.create_cars_and_tragets(self.space_w, self.space_h)

        self.tot_targets = [0]*len(self.ai) if self.ai is not None else [0]
        self.tot_phys_time = 0
        self.tot_phys_run  = 0

        init_s, s = self.init_state(), self.state(0)
        if self.ai is not None:                               # reset ai agents:
            for i, (name, ai) in enumerate(self.ai.items()):
                init_s['ai_kind'] = i
                if ai['ai'] is not None:
                    ai['ai'].reset(init_s, s)
        return  init_s, s
    #---------------------------------------------------------------------------

    def create_segments(self, w, h):
        segs   = []         # list of the segments (boundaries of space)
        segs.append( [[0.,0.,0.],  [w, 0.,0.] ] )
        segs.append( [[w, 0.,0.],  [w, h, 0.] ] )
        segs.append( [[w, h, 0.],  [0.,h, 0.] ] )
        segs.append( [[0.,h, 0.],  [0.,0.,0.] ] )

        if  self.level > 0:      # segments inside space
            segs.append( [[w/2, 3*h/14, 0.],  [w/2,  6*h/14, 0.] ] )
            segs.append( [[w/2, 8*h/14, 0.],  [w/2, 11*h/14, 0.] ] )

        self.segs = np.array(segs)    # (M,2,3)
    #---------------------------------------------------------------------------

    def create_cars_and_tragets(self, w, h, pad = 2):
        self.cars   = []         # list of the cars
        self.targets= []         # list of the targets for agents

        for i in range(self.n_cars):            
            vel = np.array([0., 0., 0.])
            if i == 0:
                pos = np.array([w/4., h/2., 0.])
                dir = np.array([1., 0., 0.])
            else:
                pos = np.zeros(3,)
                pos[0] = pad + np.random.rand()*(w-2*pad);
                pos[1] = pad + np.random.rand()*(h-2*pad)
                alpha = np.random.rand()*2*np.pi
                dir = np.array([np.cos(alpha), np.sin(alpha), 0.])

            self.cars.append( Car(pos=pos, dir=dir, vel=vel, index=i, kind=self.kinds[i]) )

            if self.params['all_targets_are_same'] and i > 0:
                target = self.targets[0]
            else:
                target = self.get_random_target(w, h, pad)
            self.targets.append(target)
    #---------------------------------------------------------------------------

    def get_random_target(self, w, h, pad):
        for _ in range(100):
            tar = np.array([pad+np.random.rand()*(w-2*pad), pad+np.random.rand()*(h-2*pad), 0.])
            ok = True
            for i in range(4, len(self.segs)):  # only internal segments
                r = self.r_seg(tar, self.segs[i])
                if np.linalg.norm(r) < pad:
                    ok = False
                    break
            if ok:
                return tar

        return tar
    #---------------------------------------------------------------------------

    def step(self, action, dt):
        """ Execute actions of all agents  """
        for a,car in zip(action, self.cars):
            car.action(a, dt)
        self.reward = np.zeros((self.n_cars,))

        self.phys(dt)                                      # physics processing

        state =  self.state(dt)
        return state, self.reward , False

    #---------------------------------------------------------------------------

    def r_seg(self, x, seg):
        """ radius vector from segment seg to point x """
        p1, p2 = seg
        t = ((x-p1)@(p2-p1)) / ((p2-p1)@(p2-p1))
        if t < 0: return p1 - x
        if t > 1: return p2 - x
        return (1-t)*p1 + t*p2 - x
    #---------------------------------------------------------------------------

    def phys(self, dt):
        if dt < 1e-8: return
        for car in self.cars:
            car.phys(dt)
            self.collisions_segs(car)

        self.collect_targets()

        if self.params['car_collision']:
            for i in range(self.n_cars):
                c1 = self.cars[i]
                for j in range(i+1, self.n_cars):
                    self.collisions_cars(c1, self.cars[j], dt)

        self.tot_phys_time += dt
        self.tot_phys_run  += 1
        self.calc_fps()
    #---------------------------------------------------------------------------

    def collect_targets(self, pad=10):
        """ Collecting car targets"""
        new_target = None
        for i, (car,target) in enumerate(zip(self.cars, self.targets)):
            pos = car.pos
            if (pos-target)@(pos-target) < car.radius**2:
                self.targets[i] = self.get_random_target(self.space_w, self.space_h, pad)
                new_target = target
                self.tot_targets[car.kind] += 1
                self.reward[i] += Environment.REWARD_TARGET

        if self.params['all_targets_are_same'] and new_target is not None:
            for i in range(len(self.targets)):
                self.targets[i] = new_target.copy()
    #---------------------------------------------------------------------------

    def collisions_segs(self, car):
        """ Car collision with segments """
        if self.params['segment_collision']:
            for seg in self.segs:
                r = self.r_seg(car.pos, seg)
                d = np.linalg.norm(r)         # dist to segment
                if d <= car.radius:
                    car.pos -= r* ((car.radius/d) - 1)
                    car.vel = -0.5*car.vel
                    self.reward[car.index] += Environment.REWARD_CAR_SEG

        # на всякий случай:
        D = 2*car.radius
        if car.pos[0] < -D:              car.pos[0] = self.space_w - D
        if car.pos[0] > self.space_w+D:  car.pos[0] = D
        if car.pos[1] < -D:              car.pos[1] = self.space_h - D
        if car.pos[1] > self.space_h+D:  car.pos[1] = D
    #---------------------------------------------------------------------------

    def collisions_cars(self, c1, c2, dt, elast=0.5, push=1.2):
        """
        Cars c1 and c2 collision. TODO: speed up N^2 (split space into cells)
        """
        r = c2.pos - c1.pos
        d = np.linalg.norm(r)
        if d > c1.radius + c2.radius:
            return 0.
        n = r / (d+1e-8)

        # move the "balls" apart:
        c = 0.5 * (c1.pos +  c2.pos + (c1.radius-c2.radius) * n )
        c1.pos = c - (c1.radius*push)*n  # We greatly increase the distance;
        c2.pos = c + (c2.radius*push)*n  # these are still not balls!
        r = c2.pos - c1.pos
        d = np.linalg.norm(r)

        # change "ball" velocities:
        v  = c1.vel - c2.vel
        dv = r * ((r @ v)/(d**2+1e-8))
        c1.vel -= dv;  c2.vel += dv

        c1.vel *= elast;  c2.vel *= elast

        c1.set_dir(dt, np.linalg.norm(c1.vel), turn=False)
        c2.set_dir(dt, np.linalg.norm(c2.vel), turn=False)

        self.reward[c1.index] += Environment.REWARD_CAR_CAR
        self.reward[c2.index] += Environment.REWARD_CAR_CAR
    #---------------------------------------------------------------------------

    def calc_fps(self, beta = 0.9):
        if self.prev_time is None:
            self.prev_time = time.time()
            return

        t = time.time()
        fps  = 1 / (t-self.prev_time+1e-8)
        self.fps = fps if  self.fps is None else self.fps * beta + fps * (1-beta)
        self.prev_time = t
    #---------------------------------------------------------------------------

    def run(self, draw_fps=40, phys_fps=40, speed_up=True, steps=1000, verbose=1):
        """
        Args
        ------------
        draw_fps (float = 40):
            frames per second for drawing the environment
        phys_fps (float = 40):
            frames per second for physics (if speed_up then virtual)
        speed_up (bool = True):
            to accelerate physics or not; if True, then the equations will be solved for time dt = 1/phys_fps, but in computer time this will be done as quickly as possible
        steps    (int = 1000):
            number of calls to the step function of AI (same as the number of calls to phys)
        """
        if self.ai is None:
            print("Set the object AI in the constructor ")
            return

        init_s, s = self.reset()
        for i, (name,ai) in enumerate(self.ai.items()):
            init_s['ai_kind'] = i
            if ai['ai'] is not None:
                ai['ai'].reset(init_s, s)

        iter, beg_draw, beg_phys = 0, time.time(), time.time()
        while True:
            cur = time.time()
            if speed_up or cur - beg_phys >= 1/phys_fps:
                dt = 1/phys_fps if speed_up else cur - beg_phys
                if not self.pause:
                    action = np.zeros(shape=(self.n_cars, 5))
                    for i, (name,ai) in enumerate(self.ai.items()):
                        if ai['ai'] is not None:
                            a = ai['ai'].step(s, reward=0)
                            action[i==self.kinds, : a.shape[-1]] = a[i==self.kinds,:]
                    s, rew, done = self.step(action, dt)
                beg_phys = time.time()
                iter += 1

            cur = time.time()
            if  draw_fps > 0 and cur - beg_draw >= 1/draw_fps:
                self.draw()                                # if you need to draw the environment
                if  self.event() == 'quit':                # all sorts of messages from the keyboard (if necessary)
                    break
                beg_draw = time.time()

            if iter >= steps:
                break

        if verbose:
            print(f"finish: fps={0 if self.fps is None else self.fps:.2f}  steps={iter} of {steps}")
        self.close()
    #---------------------------------------------------------------------------

    def draw(self):
        """ Drawing the current state of the environment """

        W, H, mt2px = self.space_w*self.mt2px, self.space_h*self.mt2px, self.mt2px
        if self.screen is None:
            self.screen = pg.display.set_mode( ( W, H) )
            pg.display.set_icon(pg.image.load(Environment.IMG_ICON))

        if self.camera == 0:
            surf = self.screen
        else:
            self.screen.fill( (255,255,255) )   # clear the screen with white color
            surf = pg.Surface((W, H))
            surf = surf.convert_alpha()

        if self.params['show_actions'] and self.font is None:
            pg.font.init()
            self.font = pg.font.SysFont('Comic Sans MS', 16)

        self.draw_ground (surf, W,H)
        #if self.camera == 1:
        #    p0, p1 = self.cars[0].pos * mt2px, self.targets[0] * mt2px
        #    pg.draw.line(surf, (0,0,0), (p0[0],p0[1]), (p1[0],p1[1]), 1)
        if self.params['show_target_line']:
            for car, target in zip(self.cars, self.targets):
                p, tp = car.pos * mt2px, target * mt2px
                pg.draw.line(surf, (0,0,0), (p[0],p[1]), (tp[0],tp[1]), 1)
        self.draw_targets(surf, mt2px)

        for seg in self.segs:
            p1, p2 = seg[0][:-1]*mt2px,  seg[1][:-1]*mt2px
            pg.draw.line(surf, (0,0,0), (p1[0], p1[1]), (p2[0], p2[1]), 5)

        for car in self.cars:
            car.draw(surf, mt2px)
            if self.params['show_actions'] and car.actions is not None:
                text = f"{car.actions[0]:4.1f},{car.actions[1]:4.1f}"
                txt_surf = self.font.render(text, False, (0, 0, 0))
                surf.blit(txt_surf, (car.pos[0]*mt2px,car.pos[1]*mt2px))
                if len(car.actions) == 5:
                    p0 = car.pos*mt2px
                    p1 = p0 + car.actions[2:]*(2*car.radius*mt2px)
                    pg.draw.line(surf, (255,0,0), (p0[0],p0[1]), (p1[0],p1[1]), 2)

        if self.camera == 1:
            car = self.cars[0]
            if self.camera_dir is None: self.camera_dir = car.dir
            else: self.camera_dir = self.camera_dir*self.camera_beta + car.dir*(1-self.camera_beta)
            angle = (np.arctan2(self.camera_dir[1], self.camera_dir[0]) + np.pi/2) * 180 / np.pi
            s = pg.transform.rotate(surf, angle)
            pivot, center  = pg.math.Vector2(W/2, H/2),  pg.math.Vector2(W/2, H/2)
            pos  = pg.math.Vector2(car.pos[0]*self.mt2px, car.pos[1]*self.mt2px)
            rect = s.get_rect(center=center - (pos-pivot).rotate(-angle))
            self.screen.blit(s, rect)

        pg.display.update()
        self.set_caption()
    #---------------------------------------------------------------------------

    def draw_ground(self, surf, W,H):
        if self.img_ground is None:
            self.img_ground =  pg.image.load(Environment.IMG_GROUND)

        rect = self.img_ground.get_rect()
        for y in range(H // rect.size[1] + 1):
            for x in range(W // rect.size[0] + 1):
                surf.blit(self.img_ground, (x*rect.size[0], y*rect.size[1]))
    #---------------------------------------------------------------------------

    def draw_targets(self, surf,  mt2px):
        if self.img_target is None:
            self.img_target =  pg.image.load(Environment.IMG_TARGET)

        for i, target in enumerate(self.targets):
            s = pg.transform.scale( self.img_target, ( 2*mt2px, 2*mt2px))
            rect = s.get_rect(center=(target[0]*mt2px, target[1]*mt2px))
            surf.blit(s, rect)
            color =  CarColors.colors[self.cars[i].kind % len(CarColors.colors)]
            if not self.params['all_targets_are_same']:
                pg.draw.circle(surf, color, (target[0]*mt2px, target[1]*mt2px), 0.4*mt2px)
    #---------------------------------------------------------------------------

    def get_info(self):
        times = ", ".join([f"{self.tot_phys_time/(n/len(self.kinds==k)):.3f}" if n else "???"  for k,n in enumerate(self.tot_targets) ])
        return f'fps: {0. if self.fps is None else self.fps:.1f} ({self.tot_phys_time/self.tot_phys_run:.3f} sec);  time: {self.tot_phys_time:.0f} sec;  {self.tot_phys_run} steps;  targets: {self.tot_targets}; time: {times}s per kind'
    #---------------------------------------------------------------------------

    def set_caption(self):
        cur = time.time()
        if self.prev_time_out  is None: self.prev_time_out = cur
        elif cur - self.prev_time_out > 0.1:
            car = self.cars[0]
            v = np.linalg.norm(car.vel)
            if self.tot_phys_run:
                pg.display.set_caption(f' vel ={v:6.1f} m/sec ={v*3.600:6.1f} km/h | {self.get_info()}')
            self.prev_time_out = cur
    #---------------------------------------------------------------------------

    def event(self):
        """ Event handling """
        for ent in pg.event.get():
            if ent.type == pg.QUIT:
                return 'quit'
            elif ent.type == pg.KEYDOWN:
                if ent.key == pg.K_ESCAPE:
                    print(self.get_info())
                    return 'quit'
                elif ent.key == pg.K_TAB:
                    return 'tab'
                elif ent.key == pg.K_r:
                    return 'r'
                elif ent.key == pg.K_i:
                    self.set_params(show_actions = not self.params['show_actions'])
                    return 'i'
                elif ent.key == pg.K_t:
                    self.set_params(show_target_line = not self.params['show_target_line'])
                    return 't'
                elif ent.key == pg.K_p:
                    self.pause = not self.pause
                    return 'p'

        res = []
        keys=pg.key.get_pressed()
        if keys[pg.K_LEFT]:  res.append('left')
        if keys[pg.K_RIGHT]: res.append('right')
        if keys[pg.K_UP]:    res.append('up')
        if keys[pg.K_DOWN]:  res.append('down')

        return res
    #---------------------------------------------------------------------------

    def tick(self, fps):
        self.clock.tick(fps)                        # fall asleep for this fps
    #---------------------------------------------------------------------------

    def close(self):
        pg.quit()                             # close the window pygame

"""
#-------------------------------------------------------------------------------
def main_2(dt = 0.02):
    ```
    The interaction between the agent and the environment is implemented in the OpenAI Gym style
    ```
    env = Environment(ai=None, n_cars=5)
    ai  = AI_Greedy()
    (init_state, state), reward = env.reset(), 0                 # re-creation of all objects; cars are stationary at random points
    ai.reset(init_state, state)
    for _ in range(1000000):
        action = ai.step(state, reward)            # choice of actions for all agents
        state, reward, done = env.step(action, dt) # get a new state, a reward and a info about of the end of the session

        env.draw()                                 # if you need to draw the environment
        if  env.event() == 'quit' or done:         # ESC message from the keyboard or stopped by environment
            break
        env.tick(fps=1/dt)
"""