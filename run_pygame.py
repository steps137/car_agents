import numpy as np, time, sys
import torch

from quenv.environment.pygame.environment import Environment
from quenv.ai.steps.ai_random import AI_Random
from quenv.ai.steps.ai_greedy import AI_Greedy
from quenv.ai.steps.ai_phys   import AI_Phys
from quenv.ai.steps.ai_rl_01   import AI_RL01
from quenv.ai.steps.ai_rl_02   import AI_RL02
from quenv.ai.steps.ai_log    import AI_Log

def main():
    """ Easy launch of the environment. """            
    #ai={'car': {'ai':AI_Random(), 'num':4} }
    #ai={'car': {'ai':AI_Greedy(3), 'num':4} }
    ai={'car': {'ai':AI_Phys(f_car=True, f_seg=False), 'num':4} }        
    #ai={'car': {'ai':AI_RL01(), 'num':4} }        
    #ai={ 'phys': {'ai':AI_Phys(), 'num':4}, 'greedy': {'ai':AI_Greedy(3), 'num':3}, 'rnd': {'ai':AI_Random(10),  'num':1} }
    #ai={'car': {'ai':AI_Log(0), 'num':1} }
    
    env = Environment(ai=ai, w=60, h=40,  mt2px=22, level=1)   # small space
    #env = Environment(ai=ai, w=240, h=160,  mt2px=5, level=0) # large space
    env.set_params(car_collision = True, seg_collision = False, 
                   moving_targets=1, moving_percent=100, stop_on_target=0, tar_seg_collision = True,
                   show_target_line = True, show_actions=True,  all_targets_are_same=False)

    env.run(draw_fps=40, phys_fps=40, speed_up=False, reset_steps=-1,  steps=1_000_000);  # normal speed
    #env.run(draw_fps=1, phys_fps=40, speed_up=True, reset_steps=1000, steps=50_000);   # accelerated physics    

#-------------------------------------------------------------------------------

def main_game(fps = 40):
    """ Emulation of control of the first car in a keyboard. Other cars are controlled by AI. """
    ai = AI_Greedy(3)
    env = Environment(ai={'human':{'ai':None, 'num':1}, 'car':{'ai':ai, 'num':1}},
                      n_cars=2,  w=60, h=40, d=100, mt2px=20, level=0)     # small space
    env = Environment(n_cars=2,  w=150, h=80, d=100, mt2px=10, level=1)   # large space
    #env.set_params(car_collision = False, seg_collision = False, show_target_line = True, show_actions=True, all_targets_are_same=False)    
    
    init_state, state = env.reset()
    ai.reset( init_state, state )             # initialize the game, learn the state from the environment

    beg   = time.time()
    while True:
        dt  = time.time() - beg                # real physical time
        beg = time.time()

        dw, df = 0., 0.                        # turn the steering wheel and press the gas-brake pedal
        actions = ai.step(state, 0)        
        event = env.event()                    # receive events from the keyboard
        if  event == 'quit':  break            # exit the game
        if  event == 'tab':
            env.camera = (env.camera + 1) % 2  # change the game camera
            env.camera_dir = None
        if  event == 'r':
            state = env.reset()
            ai.reset(state)
        if 'left'  in event:  dw = -1.         # keys pressed (possibly several at the same time)
        if 'right' in event:  dw =  1.
        if 'up'    in event:  df =  1.
        if 'down'  in event:  df = -1.
        actions[0][0] = df                     # turn the steering wheel [-1...1] of the player's car
        actions[0][1] = dw                     # gas-brakes [-1...1] player's car

        state, _, _ = env.step(actions, dt)    # send to the environment the actions of all agents (0 is the player)

        env.draw()                             # scene rendering
        env.tick(fps)                          # maintain fps (frames per second)

#-------------------------------------------------------------------------------

# Choose what we will launch at startup:
print(f"python:{sys.version} torch:{torch.__version__}  numpy:{np.__version__}")    
main()
#main_game()