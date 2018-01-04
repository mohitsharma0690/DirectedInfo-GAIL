from grid_world import State, Action, TransitionFunction
from grid_world import create_obstacles, obstacle_movement, sample_start
from itertools import product
import numpy as np
import random
import sys
import os

def oned_to_onehot(action_delta, n):
    action_onehot = np.zeros(n,)
    action_onehot[action_delta] = 1.0

    return action_onehot

def gen_L(grid_width, grid_height, path='L_expert_trajectories'):
    ''' Generates trajectories of shape L, with right turn '''
    t = 3
    n = 2
    N = 200
    
    obstacles = create_obstacles(grid_width, grid_height)
    set_diff = list(set(product(tuple(range(3, grid_width-3)), tuple(range(3, grid_height-3)))) - set(obstacles))


    if not os.path.exists(path):
        os.makedirs(path)

    T = TransitionFunction(grid_width, grid_height, obstacle_movement)

    for i in range(N):
        filename = os.path.join(path, str(i) + '.txt')
        f = open(filename, 'w')
        for j in range(n):
            if j == 0:
                action = Action(random.choice(range(0,4)))
                state = State(sample_start(set_diff), obstacles)
            else: # take right turn
                if action.delta == 0:
                    action = Action(3)
                elif action.delta == 1:
                    action = Action(2)
                elif action.delta == 2:
                    action = Action(0)
                elif action.delta == 3:
                    action = Action(1)
            for k in range(t):
                f.write(' '.join([str(e) for e in state.state]) + '\n') # write state
                f.write(' '.join([str(e) for e in oned_to_onehot(action.delta, 4)]) + '\n') # write action
                f.write(' '.join([str(e) for e in oned_to_onehot(action.delta, 4)]) + '\n') # write c[t]s
                state = T(state, action, j)

        f.close()

def gen_sq_rec(grid_width, grid_height, path='SR_expert_trajectories'):
    ''' Generates squares if starting in quadrants 1 and 4, and rectangles if starting in quadransts 2 and 3 '''
    N = 200

    obstacles = create_obstacles(grid_width, grid_height)

    if not os.path.exists(path):
        os.makedirs(path)

    T = TransitionFunction(grid_width, grid_height, obstacle_movement)

    for i in range(N):
        filename = os.path.join(path, str(i) + '.txt')
        f = open(filename, 'w')
        half = random.choice(range(0,2))
        if half == 0: # left half
            set_diff = list(set(product(tuple(range(0, (grid_width/2)-3)), tuple(range(1, grid_height)))) - set(obstacles))
            start_loc = sample_start(set_diff)
        elif half == 1: # right half
            set_diff = list(set(product(tuple(range(grid_width/2, grid_width-2)), tuple(range(2, grid_height)))) - set(obstacles))
            start_loc = sample_start(set_diff)
            
        state = State(start_loc, obstacles)

        if start_loc[0] >= grid_width/2: # quadrants 1 and 4
            # generate 2x2 square clockwise
            t = 2
            n = 4
            delta = 3

            for j in range(n):
                for k in range(t):
                    action = Action(delta)
                    f.write(' '.join([str(e) for e in state.state]) + '\n') # write state
                    f.write(' '.join([str(e) for e in oned_to_onehot(action.delta, 4)]) + '\n') # write action
                    f.write(' '.join([str(e) for e in oned_to_onehot(action.delta, 4)]) + '\n') # write c[t]s
                    state = T(state, action, j*2 + k)
                
                if delta == 3:
                    delta = 1
                elif delta == 1:
                    delta = 2
                elif delta == 2:
                    delta = 0
            
        else: # quadrants 2 and 3
            # generate 3x1 rectangle anti-clockwise
            t = [1,3,1,3]
            delta = 1

            for j in range(len(t)):
                for k in range(t[j]):
                    action = Action(delta)
                    f.write(' '.join([str(e) for e in state.state]) + '\n') # write state
                    f.write(' '.join([str(e) for e in oned_to_onehot(action.delta, 4)]) + '\n') # write action
                    f.write(' '.join([str(e) for e in oned_to_onehot(action.delta, 4)]) + '\n') # write c[t]s
                    state = T(state, action, sum(t[0:j]) + k)

                if delta == 1:
                    delta = 3
                elif delta == 3:
                    delta = 0
                elif delta == 0:
                    delta = 2

def main():
    if int(sys.argv[1]) == 0:
        gen_L(12,12)
    elif int(sys.argv[1]) == 1:
        gen_sq_rec(12,12)
    else:
        print 'Undefined arguement!'

if __name__ == '__main__':
    main()
