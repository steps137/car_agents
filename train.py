import numpy as np, time

from environment.pygame.environment import Environment
from ai.steps.ai_greedy import AI_Greedy
from ai.steps.env_model import AI_EnvModel

import numpy as np,  matplotlib.pyplot as plt
from qunet import Data

def stat(x, names):
    n = x.shape[-1]
    print("====== x:", {x.shape}, "=======")
    for i in range(n):
        print(f"{i:2d}. {names[i]:5s}  {x[:,i].mean():7.4f} ± {x[:,i].std():7.4f}  [{x[:,i].min():7.4f} ... {x[:,i].max():7.4f}]")
    print("corr:")
    cor = np.corrcoef( [x[:, i]  for i in range(n)] )
    for j in range(n):        
        print( f"{names[j]:5s}", ", ".join([ f"{cor[j,i]:6.3f}" for i in range(n) ]))        

def main():
    """ Easy launch of the environment. """   
    ai_model = AI_EnvModel()
    ai={'car': {'ai':ai_model, 'num':16} }

    env = Environment(ai=ai, w=60, h=40,  mt2px=20, level=0)   # small space
    env.set_params(car_collision = False, seg_collision = False, show_target_line = True, show_actions=True, all_targets_are_same=False)

    #env.run(draw_fps=40, phys_fps=40, speed_up=False, steps=500)  # normal speed
    env.run(draw_fps=1, phys_fps=40, speed_up=True, steps=1000)   # accelerated physics
    
    #----------------------------------------------------------------------------------------------

    ai_model.trainer.plot()    

    data = Data( (ai_model.memo.X,  ai_model.memo.Y),  batch_size = 32)
    res = ai_model.trainer.predict(ai_model.trainer.best.score_model, data)
    y, v, x = res['y'].numpy(), res['v_true'].numpy(), res['x'].numpy()
    print(y.shape, v.shape, x.shape)

    fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(12,3), facecolor ='w')
    ax1.hist(y[:,0], bins=30, label="y1")
    ax1.hist(y[:,1], bins=30, label="y2")
    ax1.grid(ls=":"); ax1.legend() 

    ax2.hist(v, bins=30, label="v")
    ax2.grid(ls=":"); ax2.legend()

    ax3.hist(x[:,0], bins=30, label="df")
    ax3.hist(x[:,1], bins=30, label="dw")
    ax3.grid(ls=":"); ax3.legend()

    plt.show()

    print(f"v:  {v.mean():.5f} : [{v.min():.5f}...{v.max():.5f}]")
    print(f"y1: {y[:,0].mean():.5f} : [{y[:,0].min():.5f}...{y[:,0].max():.5f}]")
    print(f"y2: {y[:,1].mean():.5f} : [{y[:,1].min():.5f}...{y[:,1].max():.5f}]")
    print()
    stat(x, ['df', 'dw', 'vn', 'vd', 'vxd', 'whe'])

#-------------------------------------------------------------------------------

main()
