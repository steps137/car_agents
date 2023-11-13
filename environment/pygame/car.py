import numpy as np
import pygame as pg

class CarModelNormal:
    W       =  75   # cm  #     ---             /    in centimeters!!!
    L1      = 100   # cm  #    W |             /|    (pixels in a drawing of a car)
    L2      = 100   # cm  #      =======*=======
    wheel_w = 75    # cm  #    W |   L2    L1   |/   <- right front wheel
    wheel_h = 35    # cm  #     ---             /

    mass    = 3000  # kg  # mass of the car

    mu1     = 0.005        # coefficient of friction on air
    mu2     = 2.           # coefficient of friction on asphalt    

    omega = (np.pi/2)/0.5 # angular velocity rad/sec

    engine_forward  =  7   # acceleration by pressing the gas pedal once
    engine_back     =  5   # same, moving backwards
    engine_brake    = 25   # braking
    steering_sensitivity = 100

    v_max = 20            # 20 m/sec max velocity * 3.6 = 72 km/h
    w_max = 30*np.pi/180  # maximum rotation angle     in rad    
    w_back_T = 0.5        # the wheels time align in a second

#-------------------------------------------------------------------------------

class CarColors:  # (R,G,B) object colors:
    colors = [(255,200, 200), (100,200,100),  (100,100,200), (100,200,200), (200,100,200), (200,200,100)]
    c_base       = (192, 192, 192) # (154,   154, 154)
    c_front      = (192, 192, 192)
    c_wheel      = (0, 0, 0)
    c_headlights = (100, 100, 255)

#-------------------------------------------------------------------------------

class Car:
    def __init__(self, pos, dir, vel, index, kind=0):
        self.index = index         # number of car in total car list
        self.kind  = kind          # the ai type that controls it

        self.pos = pos       # m   # position of the car's center of mass
        self.vel = vel       # m/s # velocity of the car's center of mass
        self.dir = dir             # car orientation (from the center of mass forward)

        self.w11 = 0               # left wheel steering angle  in rad
        self.w12 = 0               # right wheel steering angle in rad

        self.force   = 0           # current gas-brake force
        self.actions = None        # current actions        

        self.model   = CarModelNormal()

        self.radius  = (self.model.L1+self.model.L2+self.model.wheel_w)/200 # in meters
        self.L       = (self.model.L1 + self.model.L2)/100                  # in meters
        self.mass    =  self.model.mass                                     # in kg (collisions)

    #---------------------------------------------------------------------------

    def action(self, act, dt):
        df, dw = act[0], act[1]
        # turn the wheels (w11 - angle left-front, a12 - right-front wheel):
        self.w11 += self.model.steering_sensitivity * dt * dw * np.pi/180   # not an instant turn 
        self.w11 = max( min(self.w11, self.model.w_max), -self.model.w_max) # angle limitation
        self.w12 = self.w11         
                        
        # engine power: 3 modes (forward, braking, reverse)
        if df > 0:                                         # gas forward
            self.force   = self.model.engine_forward*df    # speed up
        else:                                              # brake or reverse
            if (self.dir*self.vel).sum(axis=-1) > 0:       # speed along the car 
                self.force = self.model.engine_brake*df    #  slow down
            else:
                self.force = self.model.engine_back*df     # driving backwards              

        self.actions = act.copy() 
    #---------------------------------------------------------------------------

    def set_dir(self, dt, v, turn=True):
        """
        Sets the car's orientation.
        At high speeds, immediately in the direction of the speed (or against - to the nearest)
        At low speeds, turn orientation with angular velocity omega
        """
        if v > 10:
            co = self.dir @ self.vel
            self.dir = self.vel / v
            if co < 0 and turn:
                self.dir = -self.dir
        elif v > 0:                     # not stable at high speed
            n = self.vel / v
            co = self.dir @ n
            phi =-np.arcsin(self.dir[0]*n[1]-self.dir[1]*n[0])
            if co < 0:
                phi *= -1

            a = min(self.model.omega * dt, np.abs(phi))
            a *= np.sign(phi)

            self.dir = np.array([np.cos(a)*self.dir[0]+np.sin(a)*self.dir[1],
                                -np.sin(a)*self.dir[0]+np.cos(a)*self.dir[1], 0.])

    #---------------------------------------------------------------------------

    def phys(self, dt, eps=1e-8):
        """
        Physics of the car. Look: https://qudata.com/ml/ru/agents/car_phys.html
        """
        dir = self.dir
        vel1 = self.vel
        v = np.linalg.norm(self.vel)
        
        #co, si, dir = np.cos(self.w11), np.sin(self.w11), self.dir
        #force = np.array([dir[0]*co + dir[1]*si, -dir[0]*si + dir[1]*co, 0.]) * self.force       
        force = dir * (self.force)              # rear wheel drive car
    
        fc = np.array([-self.dir[1], self.dir[0], 0.])*v*v*np.tan(self.w11)/self.L
        a = fc                                   # centripetal force from wheels
        self.vel += a*dt                         # new velocity value        
        
        dv2 = (np.linalg.norm(fc)*dt)**2         # corrections of second order
        v2  = np.linalg.norm(self.vel)**2               
        self.vel *= (max(0, v2 - dv2) / (v2 + eps))**0.5

        a = force                                #  engine power
        a -= self.model.mu1*v*vel1           # air friction
        a -= self.model.mu2*vel1/(v+eps)     # road friction
        self.vel += a*dt                         # new velocity value                 

        v = np.linalg.norm(self.vel)             # limit the speed value
        if v > self.model.v_max:
            self.vel *= self.model.v_max / v
            v = self.model.v_max

        self.pos += (vel1+self.vel) * (dt/2)     # new car position

        self.set_dir(dt, v)                      # changing car orientation

        # the wheels themselves return to the zero angle:
        self.w11 += -np.sign(self.w11) * self.model.w_max *dt / self.model.w_back_T
        self.w12 = self.w11

        self.force = 0                           # reset the gas pressure once

    #---------------------------------------------------------------------------

    def draw(self, screen, mt2px):
        def wheel(surf, p, w):
            s = pg.Surface((self.model.wheel_w, self.model.wheel_h))
            s = s.convert_alpha()
            pg.draw.rect(s,  CarColors.c_wheel, [0,0, self.model.wheel_w, self.model.wheel_h], border_radius=10)
            s = pg.transform.rotate(s, -np.rad2deg(w))
            rect = s.get_rect(center=(p[0], p[1]))
            surf.blit(s, rect)

        p0  = [self.model.wheel_w+self.model.L2, self.model.W+self.model.wheel_h]
        p1  = [p0[0]+self.model.L1, p0[1]]
        p2  = [p0[0]-self.model.L2, p0[1]]
        p11 = [p1[0], p1[1]+self.model.W];  p21 = [p2[0], p2[1]+self.model.W]
        p12 = [p1[0], p1[1]-self.model.W];  p22 = [p2[0], p2[1]-self.model.W]

        w, h = self.model.L1+self.model.L2+2*self.model.wheel_w, 2*self.model.W+2*self.model.wheel_h
        surf = pg.Surface((w,h))

        surf.fill((255,255,255))

        color = CarColors.colors[self.kind % len(CarColors.colors)]
        pg.draw.rect(surf, color, (0,0,w,h), border_radius=50)
        pg.draw.line(surf, CarColors.c_front, p0, p1, 20)
        pg.draw.line(surf, CarColors.c_base, p0, p2,20)
        color = (255,0,0) if self.index == 0 else CarColors.c_base
        pg.draw.circle(surf, color, p0, 20)

        pg.draw.line(surf, CarColors.c_base, p1, p11, 10)
        pg.draw.line(surf, CarColors.c_base, p1, p12, 10)
        pg.draw.line(surf, CarColors.c_base, p2, p21, 10)
        pg.draw.line(surf, CarColors.c_base, p2, p22, 10)

        pg.draw.circle(surf, CarColors.c_headlights, (p11[0]+self.model.wheel_w,p11[1]-self.model.wheel_h), 30)
        pg.draw.circle(surf, CarColors.c_headlights, (p12[0]+self.model.wheel_w,p12[1]+self.model.wheel_h), 30)

        wheel(surf, p11, self.w11)
        wheel(surf, p12, self.w12)
        wheel(surf, p21, 0.)
        wheel(surf, p22, 0.)

        surf = surf.convert_alpha()   # all sizes were in sm (!):
        surf = pg.transform.scale( surf, (surf.get_width()*mt2px/100, surf.get_height()*mt2px/100))
        surf = pg.transform.rotate(surf, np.rad2deg(np.arctan2(-self.dir[1], self.dir[0])))
        rect = surf.get_rect(center=(self.pos[0]*mt2px, self.pos[1]*mt2px))
        screen.blit(surf, rect)
