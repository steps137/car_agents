"""
There are two options for simulating car physics to train AI models.

1) Physics simulation takes place in the Unreal engine, which interacts with a python script.
You should select EnvironmentUnreal as the environment (see example in function main_0).

2) Physics simulation occurs directly in Python.
You don't need Unreal for this. Physics is significantly faster, but not as realistic.
The Environment class should be selected as the environment.
Various launch options are possible (below functions main1, main2 and main_game)

The logic of interaction with the environment is described in the agent class.
This class should have methods:
     reset(self, init_state, state) - getting the initial static state and current state of environment
     step (self, state, reward)     - the agent receives current state and reward (can be ignored)
The last method should return actions as a numpy array (N,2),
where N is the number of agents, 2 actions (gas/brake and steering wheel turn) for each [-1..1].

1. run "AiCar v3/NeuralPilotAI/setup.cmd" for installing libraries
2. run "AiCar v3/NeuralPilotAI/run.cmd"
When working with Unreal, the car.exe file is launched.
Then this script, in which the main1 function is activated.


Examples of agents can be found in the ai.steps directory:
     * ai_random - performs random actions
     * ai_greedy - greedy movement towards the goal

--------------------------------------------------------------------------------

The environment transmits to the agent the initial state (about static data),
the current state of the environment and the reward:

init_state = {
    'space':   np.array (3,)    dimensions in meters of space
    'n_cars':  int,             number of cars (any kind)
    'ai_kind:  int,             kind of type current AI
    'kind':    np.array (N,)    type of each car for different AI (kind==0 - human)
	'string':  np.array (M,2,3) walls positions L(x,y,z), R(x,y,z)
}

state = {
    'space': np.array (3,)    dimensions in meters of space
    'dt':    float            time since previous states update
    'pos':   np.array (N,3)   position vectors of the center of mass of all cars in meters
    'vel':   np.array (N,3)   velocity vectors of the center of mass of all cars in meters per seconds
    'dir':   np.array (N,3)   car orientation unit vector (from the center of mass forward)
    'wheels':np.array (N,2)   turning angles of the front wheels of cars
    'target_pos': np.array (N,3)   position of target points
    'target_vel': np.array (N,3)   velocity of target points
}
reward: np.array(N,) - real numbers for each machine (achieving a target, etc.)

--------------------------------------------------------------------------------

When working with a simulator in Python (without Unreal),
it is possible to configure various environment parameters:
    w,h,d - dimensions of the playing space in meters
    mt2px - conversion of meters to pixels (if w=100m and mt2px=15, then the screen width is 1500px)
    n_cars - number of cars. The first car (index=0) is considered the player's car (pink), although the AI may ignore this
    level - difficulty level (0-empty space, except borders; 1-there are partitions inside the space)

After creating the environment, you can call the env.set_params function (see example in main_1) with the arguments:
    car_collision        = True  handle car collisions
    segment_collision    = True  handle collisions with segments
    all_targets_are_same = False the coordinates of all targets are the same
    show_target_line:    = False show line from car to target
    show_actions:        = False show current car actions

If AI returns actions of dimension (N,5), then the first 2 actions are interpreted as gas and steering,
the remaining 3 are components of the vector that is drawn when the show_actions=True
You can, for example, write the direction from the Decision Making module into this vector.

In addition to standard keyboard arrow control (see main_game function), the pygame environment supports the following keys:
    Esc - exit
    p   - pause on/off (physics stops)
    i   - enable/disable action information (show_actions     = True/False)
    t   - enable/disable line to target     (show_target_line = True/False)
    tab - switch the camera (only in the game mode - see main_game)
"""

import numpy as np, time

from environment.pygame.environment import Environment
from ai.steps.ai_random import AI_Random
from ai.steps.ai_greedy import AI_Greedy
from ai.steps.ai_phys   import AI_Phys

def main_1():
    """
    Easy launch of the environment.
    Inside the run method, the state is transferred to the agent and the action is requested.
    Applicable for both the Unreal engine and the Python simulator.
    Convenient in training mode, when an accelerated calculation of the physics of the environment is needed:
        * fps - frames per second for drawing the environment (if fps=0 we do not draw);
                each step of the environment is called as often as possible
        * dt  - time in seconds that is transmitted to the environment at each step;
                if fps > 1/dt it means "environment acceleration"
        * steps - the number of steps the environment will perform;
    """
    ai={'car': {'ai':AI_Greedy(), 'num':2}, 'rnd': {'ai':AI_Random(10),  'num':2} }
    #ai={'car': {'ai':AI_Phys(), 'num':4} }

    env = Environment( ai=ai, n_cars=4,  w=60, h=40, d=100, mt2px=20, level=0)
    env.set_params(car_collision = False, show_target_line = True, show_actions=True)
    env.run(draw_fps=40, phys_fps=40, speed_up=False, steps=1_000_000)
    #env.run(draw_fps=1, phys_fps=40, speed_up=True, steps=1_000_000)
#-------------------------------------------------------------------------------

def main_2(dt = 0.02):
    """
    The interaction between the agent and the environment is implemented in the OpenAI Gym style
    Оnly available in the Python simulator.
    """
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

#-------------------------------------------------------------------------------

def main_game(fps = 40):
    """
    Emulation of control of the first car in a keyboard.
    Other cars are controlled by AI. Оnly available in the Python simulator.
    """
    ai = AI_Greedy()
    env = Environment(ai={'human':{'ai':None, 'num':1}, 'car':{'ai':ai, 'num':1}},
                      n_cars=2,  w=60, h=40, d=100, mt2px=20, level=0)     # small space
    #env = Environment(n_cars=2,  w=150, h=80, d=100, mt2px=10, level=1)   # large space

    env.all_targets_are_same = True           # the coordinates of all targets are the same
    init_state, state = env.reset()
    ai.reset( init_state, state )             # initialize the game, learn the state from the environment

    beg   = time.time()
    while True:
        dt  = time.time() - beg                # real physical time
        beg = time.time()

        dw, df = 0., 0.                        # turn the steering wheel and press the gas-brake pedal
        actions = ai.step(state, 0)
        #actions[:,:] = 0
        event = env.event()                    # receive events from the keyboard
        if  event == 'quit':  break            # exit the game
        if  event == 'tab':
            env.camera = (env.camera + 1) % 2  # change the game camera
            env.camera_dir = None
        if  event == 'r':
            state = env.reset()
            ai.reset(state)
        if 'left'  in event:  dw =  1.         # keys pressed (possibly several at the same time)
        if 'right' in event:  dw = -1.
        if 'up'    in event:  df =  1.
        if 'down'  in event:  df = -1.
        actions[0][0] = df                     # turn the steering wheel [-1...1] of the player's car
        actions[0][1] = dw                     # gas-brakes [-1...1] player's car

        state, _, _ = env.step(actions, dt)    # send to the environment the actions of all agents (0 is the player)

        env.draw()                             # scene rendering
        env.tick(fps)                          # maintain fps (frames per second)

#-------------------------------------------------------------------------------

# Choose what we will launch at startup:
main_1()
#main_2()
#main_game()