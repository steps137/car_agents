import numpy as np, time

from environment.pygame.environment import Environment
from ai.steps.ai_greedy import AI_Greedy
from ai.steps.env_model import AI_EnvModel

def main():
    """ Easy launch of the environment. """   
    ai_model = AI_EnvModel()
    ai={'car': {'ai':ai_model, 'num':16} }

    env = Environment(ai=ai, w=60, h=40,  mt2px=20, level=0)   # small space
    env.set_params(car_collision = False, seg_collision = False, show_target_line = True, show_actions=True, all_targets_are_same=False)

    #env.run(draw_fps=40, phys_fps=40, speed_up=False, steps=1_000)  # normal speed
    env.run(draw_fps=1, phys_fps=40, speed_up=True, steps=1000)   # accelerated physics
    ai_model.trainer.plot()    


#-------------------------------------------------------------------------------

main()
