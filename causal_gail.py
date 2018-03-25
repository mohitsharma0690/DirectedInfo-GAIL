import argparse
import sys
import os
import pickle
import math
import random
from collections import namedtuple
from itertools import count, product

import numpy as np
import scipy.optimize
from scipy.stats import norm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

from models import Policy, Posterior, Reward, Value
from grid_world import State, Action, TransitionFunction, RewardFunction, RewardFunction_SR2
from grid_world import create_obstacles, obstacle_movement, sample_start
from load_expert_traj import Expert
from replay_memory import Memory
from running_state import ZFilter
from utils import clip_grads

# from utils import *

#torch.set_default_tensor_type('torch.DoubleTensor')
dtype = torch.cuda.FloatTensor
dtype_Long = torch.cuda.LongTensor
#dtype = torch.FloatTensor
#dtype_Long = torch.LongTensor
PI = torch.DoubleTensor([3.1415926]).type(dtype)

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--expert-path', default="L_expert_trajectories/", metavar='G',
                    help='path to the expert trajectory files')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                    help='batch size (default: 2048)')
parser.add_argument('--num-episodes', type=int, default=500, metavar='N',
                    help='number of episodes (default: 500)')
parser.add_argument('--max-ep-length', type=int, default=6, metavar='N',
                    help='maximum episode length (default: 6)')
parser.add_argument('--optim-epochs', type=int, default=5, metavar='N',
                    help='number of epochs over a batch (default: 5)')
parser.add_argument('--optim-batch-size', type=int, default=64, metavar='N',
                    help='batch size for epochs (default: 64)')
parser.add_argument('--num-expert-trajs', type=int, default=5, metavar='N',
                    help='number of expert trajectories in a batch (default: 5)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='interval between saving policy weights (default: 100)')
parser.add_argument('--entropy-coeff', type=float, default=0.0, metavar='N',
                    help='coefficient for entropy cost')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='Clipping for PPO grad')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='path to checkpoint')

args = parser.parse_args()


#-----Environment-----#
width = height = 12
obstacles = create_obstacles(width, height)
set_diff = list(set(product(tuple(range(3, width-3)), repeat=2)) - set(obstacles))
start_loc = sample_start(set_diff)

s = State(start_loc, obstacles)
T = TransitionFunction(width, height, obstacle_movement)

if args.expert_path == 'SR2_expert_trajectories/':
    R = RewardFunction_SR2(-1.0,1.0,width)
else:
    R = RewardFunction(-1.0,1.0)

num_inputs = s.state.shape[0]
num_actions = 4
if args.expert_path == 'SR2_expert_trajectories/':
    num_c = 2
else:
    num_c = 4

#env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, 0, num_c, num_actions, hidden_size=64, output_activation='sigmoid').type(dtype)
old_policy_net = Policy(num_inputs, 0, num_c, num_actions, hidden_size=64, output_activation='sigmoid').type(dtype)
#value_net = Value(num_inputs+num_c, hidden_size=64).type(dtype)
reward_net = Reward(num_inputs, num_actions, num_c, hidden_size=64).type(dtype)
posterior_net = Posterior(num_inputs, num_actions, num_c, hidden_size=64).type(dtype)

opt_policy = optim.Adam(policy_net.parameters(), lr=0.0003)
#opt_value = optim.Adam(value_net.parameters(), lr=0.0003)
opt_reward = optim.Adam(reward_net.parameters(), lr=0.0003)
opt_posterior = optim.Adam(posterior_net.parameters(), lr=0.0003)


def epsilon_greedy_linear_decay(action_vector, n_episodes, n, low=0.1, high=0.9):
    if n <= n_episodes:
        eps = ((low-high)/n_episodes)*n + high
    else:
        eps = low

    if np.random.uniform() > eps:
        return np.argmax(action_vector)
    else:
        return np.random.randint(low=0, high=4)

def epsilon_greedy(action_vector, eps=0.1):
    if np.random.uniform() > eps:
        return np.argmax(action_vector)
    else:
        return np.random.randint(low=0, high=4)

def greedy(action_vector):
    return np.argmax(action_vector)

def oned_to_onehot(action_delta, n=num_actions):
    action_onehot = np.zeros(n,)
    action_onehot[int(action_delta)] = 1.0

    return action_onehot

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0).type(dtype)
    action, _, _ = policy_net(Variable(state))
    return action

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
    return log_density.sum(1)


def update_params(gen_batch, expert_batch, i_episode, optim_epochs, optim_batch_size):
    criterion = nn.BCELoss()
    #criterion_posterior = nn.NLLLoss()
    criterion_posterior = nn.MSELoss()

    #opt_value.lr = args.learning_rate*max(1.0 - float(i_episode)/args.num_episodes, 0)
    opt_policy.lr = args.learning_rate*max(1.0 - float(i_episode)/args.num_episodes, 0)
    clip_epsilon = args.clip_epsilon*max(1.0 - float(i_episode)/args.num_episodes, 0)

    # generated trajectories
    rewards = torch.Tensor(gen_batch.reward).type(dtype)
    masks = torch.Tensor(gen_batch.mask).type(dtype)
    actions = torch.Tensor(np.concatenate(gen_batch.action, 0)).type(dtype)
    states = torch.Tensor(gen_batch.state).type(dtype)
    latent_c = torch.Tensor(gen_batch.c).type(dtype)
    #values = value_net(Variable(states))

    # expert trajectories
    list_of_expert_states = []
    for i in range(len(expert_batch.state)):
        list_of_expert_states.append(torch.Tensor(expert_batch.state[i]))
    expert_states = torch.cat(list_of_expert_states,0).type(dtype)

    list_of_expert_actions = []
    for i in range(len(expert_batch.action)):
        list_of_expert_actions.append(torch.Tensor(expert_batch.action[i]))
    expert_actions = torch.cat(list_of_expert_actions, 0).type(dtype)

    list_of_expert_latent_c = []
    for i in range(len(expert_batch.c)):
        list_of_expert_latent_c.append(torch.Tensor(expert_batch.c[i]))
    expert_latent_c = torch.cat(list_of_expert_latent_c, 0).type(dtype)

    list_of_masks = []
    for i in range(len(expert_batch.mask)):
        list_of_masks.append(torch.Tensor(expert_batch.mask[i]))
    expert_masks = torch.cat(list_of_masks, 0).type(dtype)

    returns = torch.Tensor(actions.size(0),1).type(dtype)
    deltas = torch.Tensor(actions.size(0),1).type(dtype)
    advantages = torch.Tensor(actions.size(0),1).type(dtype)

    # compute advantages
    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        #deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        #advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        advantages[i] = returns[i] #changed by me to see if value function is causing problems
        prev_return = returns[i, 0]
        #prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    advantages = (advantages - advantages.mean()) / advantages.std()

    # backup params after computing probs but before updating new params
    #policy_net.backup()
    for old_policy_param, policy_param in zip(old_policy_net.parameters(), policy_net.parameters()):
        old_policy_param.data.copy_(policy_param.data)

    # update value, reward and policy networks
    optim_iters = int(math.ceil(args.batch_size/optim_batch_size))
    optim_batch_size_exp = int(math.floor(expert_actions.size(0)/(optim_iters)))

    for _ in range(optim_epochs):
        perm = np.arange(actions.size(0))
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).type(dtype_Long)
        states = states[perm]
        actions = actions[perm]
        latent_c = latent_c[perm]
        #values = values[perm]
        targets = targets[perm]
        advantages = advantages[perm]
        perm_exp = np.arange(expert_actions.size(0))
        np.random.shuffle(perm_exp)
        perm_exp = torch.LongTensor(perm_exp).type(dtype_Long)
        expert_states = expert_states[perm_exp]
        expert_actions = expert_actions[perm_exp]
        expert_latent_c = expert_latent_c[perm_exp]
        cur_id = 0
        cur_id_exp = 0

        for _ in range(optim_iters):
            cur_batch_size = min(optim_batch_size, actions.size(0) - cur_id)
            cur_batch_size_exp = min(optim_batch_size_exp, expert_actions.size(0) - cur_id_exp)
            state_var = Variable(states[cur_id:cur_id+cur_batch_size])
            action_var = Variable(actions[cur_id:cur_id+cur_batch_size])
            latent_c_var = Variable(latent_c[cur_id:cur_id+cur_batch_size])
            advantages_var = Variable(advantages[cur_id:cur_id+cur_batch_size])
            expert_state_var = Variable(expert_states[cur_id_exp:cur_id_exp+cur_batch_size_exp])
            expert_action_var = Variable(expert_actions[cur_id_exp:cur_id_exp+cur_batch_size_exp])
            expert_latent_c_var = Variable(expert_latent_c[cur_id_exp:cur_id_exp+cur_batch_size_exp])

            # update reward net
            opt_reward.zero_grad()

            # backprop with expert demonstrations
            o = reward_net(torch.cat((expert_state_var, expert_action_var, expert_latent_c_var),1))
            loss = criterion(o, Variable(torch.zeros(expert_action_var.size(0),1).type(dtype)))
            loss.backward()

            # backprop with generated demonstrations
            o = reward_net(torch.cat((state_var, action_var, latent_c_var),1))
            loss = criterion(o, Variable(torch.ones(action_var.size(0),1)).type(dtype))
            loss.backward()
    
            opt_reward.step()

            # update posterior net # We need to do this by reparameterization trick instead.
            # We should not put action_var (a_t) in the posterior net since we need c_t to 
            # predict a_t while till now we only have c_{t-1}.
            mu, _ = posterior_net(torch.cat((state_var, action_var, latent_c_var), 1))
            _, latent_c_targets = latent_c_var.max(1)
            latent_c_targets = latent_c_targets.type(dtype)
            loss = criterion_posterior(mu, latent_c_targets)
            loss.backward()

            # compute old and new action probabilities
            action_means, action_log_stds, action_stds = policy_net(torch.cat((state_var, latent_c_var), 1))
            log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

            action_means_old, action_log_stds_old, action_stds_old = old_policy_net(torch.cat((state_var, latent_c_var), 1))
            log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

            # update value net
            #opt_value.zero_grad()
            #value_var = value_net(state_var)
            #value_loss = (value_var - targets[cur_id:cur_id+cur_batch_size]).pow(2.).mean()
            #value_loss.backward()
            #opt_value.step()

            # update policy net
            opt_policy.zero_grad()
            ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
            surr1 = ratio * advantages_var[:,0]
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var[:,0]
            policy_surr = -torch.min(surr1, surr2).mean()
            policy_surr.backward()
            torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
            opt_policy.step()

            # set new starting point for batch
            cur_id += cur_batch_size
            cur_id_exp += cur_batch_size_exp


#running_state = ZFilter((num_inputs,), clip=5)
#running_reward = ZFilter((1,), demean=False, clip=10)
episode_lengths = []
optim_epochs = 5
optim_percentage = 0.05

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)

stats = {'true_reward': [], 'ep_length':[]}

expert = Expert(args.expert_path, num_inputs)
print 'Loading expert trajectories ...'
expert.push()
print 'Expert trajectories loaded.'

for i_episode in count(1):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    true_reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        c = expert.sample_c() # read c sequence from expert trajectories. Here you'll need to pass the expert trajectories through the pretrained posterior net.
        if args.expert_path == 'SR_expert_trajectories/':
            if np.argmax(c[0,:]) == 1: # left half 
                set_diff = list(set(product(tuple(range(0, (width/2)-3)), tuple(range(1, height)))) - set(obstacles))
            elif np.argmax(c[0,:]) == 3: # right half
                set_diff = list(set(product(tuple(range(width/2, width-2)), tuple(range(2, height)))) - set(obstacles))
            
        start_loc = sample_start(set_diff)
        s = State(start_loc, obstacles)
        #state = running_state(state)
        R.reset()

        reward_sum = 0
        true_reward_sum = 0
        #memory = Memory()

        for t in range(args.max_ep_length): # Don't infinite loop while learning
            ct = c[t,:]
            action = select_action(np.concatenate((s.state, ct)))
            action = epsilon_greedy_linear_decay(action.data.cpu().numpy(), args.num_episodes*0.5, i_episode, low=0.05, high=0.3)

            reward = -float(reward_net(torch.cat((Variable(torch.from_numpy(s.state).unsqueeze(0)).type(dtype), 
                    Variable(torch.from_numpy(oned_to_onehot(action)).unsqueeze(0)).type(dtype), 
                    Variable(torch.from_numpy(ct).unsqueeze(0)).type(dtype)), 1)).data.cpu().numpy()[0,0])

            if t < args.max_ep_length-1:

                mu, sigma = posterior_net(torch.cat((Variable(torch.from_numpy(s.state).unsqueeze(0)).type(dtype),
                                 Variable(torch.from_numpy(oned_to_onehot(action)).unsqueeze(0)).type(dtype),
                                 Variable(torch.from_numpy(ct).unsqueeze(0)).type(dtype)), 1))

                mu = mu.data.cpu().numpy()[0,0]
                sigma = sigma.data.cpu().numpy()[0,0]

                reward += norm.pdf(np.argmax(c[t+1,:]), loc=mu, scale=sigma) # should ideally be logpdf, but pdf may work better. Try both.
                                                                             # Also, argmax for now, but has to be c[t+1, 1:] when reverting to proper c ...

                #reward += math.exp(np.sum(np.multiply(posterior_net(torch.cat((Variable(torch.from_numpy(s.state).unsqueeze(0)).type(dtype), 
                #        Variable(torch.from_numpy(oned_to_onehot(action)).unsqueeze(0)).type(dtype), 
                #        Variable(torch.from_numpy(ct).unsqueeze(0)).type(dtype)),1)).data.cpu().numpy()[0,:], c[t+1,:])))

            next_s = T(s, Action(action), R.t)
            true_reward = R(s, Action(action), ct)
            reward_sum += reward
            true_reward_sum += true_reward

            #next_state = running_state(next_state)

            mask = 1
            if t == args.max_ep_length-1:
                R.terminal = True
                mask = 0

            memory.push(s.state, np.array([oned_to_onehot(action)]), mask, next_s.state, reward, ct)

            if args.render:
                env.render()
            
            if R.terminal:
                break

            s = next_s

        #ep_memory.push(memory)
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum
        true_reward_batch += true_reward_sum

    #optim_batch_size = min(num_episodes, max(10,int(num_episodes*optim_percentage)))
    reward_batch /= num_episodes
    true_reward_batch /= num_episodes
    gen_batch = memory.sample()
    expert_batch = expert.sample(size=args.num_expert_trajs)

    update_params(gen_batch, expert_batch, i_episode, optim_epochs, args.optim_batch_size)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward {}\tAverage reward {}\tLast true reward {}\tAverage true reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch, true_reward_sum, true_reward_batch))

    stats['true_reward'].append(true_reward_batch)

    results_path = os.path.join(args.checkpoint, 'results.pkl')
    with open(results_path,'wb') as results_f:
        pickle.dump((stats), results_f, protocol=2)

    if i_episode % args.save_interval == 0:
        f_w = open(os.path.join(args.checkpoint, 'ep_' + str(i_episode) + '.pth'), 'wb')
        checkpoint = {'policy': policy_net, 'posterior': posterior_net}
        torch.save(checkpoint, f_w)

    if i_episode == args.num_episodes:
        break
