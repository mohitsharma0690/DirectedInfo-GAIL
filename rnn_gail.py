import argparse
import sys
import os
import math
import random
import pickle
from collections import namedtuple
from itertools import count, product

import gym
import numpy as np
import scipy.optimize
from gym import wrappers

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


from models import Policy, Value
from grid_world import State, Action, TransitionFunction, RewardFunction
from grid_world import create_obstacles, obstacle_movement, sample_start
from load_expert_traj import Expert
from gru_discrete import GRU
from replay_memory import Memory, Memory_Ep
from running_state import ZFilter

# from utils import *

#torch.set_default_tensor_type('torch.DoubleTensor')
dtype = torch.cuda.FloatTensor
dtype_Long = torch.cuda.LongTensor
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
#parser.add_argument('--optim-batch-size', type=int, default=64, metavar='N',
#                    help='batch size for epochs (default: 64)')
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
R = RewardFunction(-1.0,1.0)

num_inputs = s.state.shape[0]
num_actions = 4
num_c = 4

#env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = GRU(num_inputs+num_c, num_actions, hidden_size=128, dtype=dtype).type(dtype)
old_policy_net = GRU(num_inputs+num_c, num_actions, hidden_size=128, dtype=dtype).type(dtype)
value_net = Value(num_inputs+num_c, hidden_size=128).type(dtype)
reward_net = GRU(num_inputs+num_actions+num_c, 1, hidden_size=128, policy_flag=0, activation_flag=2, dtype=dtype).type(dtype)

opt_policy = optim.Adam(policy_net.parameters(), lr=0.0003)
opt_value = optim.Adam(value_net.parameters(), lr=0.0003)
opt_reward = optim.Adam(reward_net.parameters(), lr=0.0003)
    
def create_batch_inputs(batch_states_list, batch_latent_c_list, batch_actions_list, batch_advantages_list=None):
    lengths = []
    for states in batch_states_list:
        lengths.append(states.size(0))

    max_length = max(lengths)
    batch_states = torch.zeros(len(batch_states_list), max_length, num_inputs).type(dtype)
    batch_latent_c = torch.zeros(len(batch_latent_c_list), max_length, num_c).type(dtype)
    batch_actions = torch.zeros(len(batch_actions_list), max_length, num_actions).type(dtype)
    if batch_advantages_list:
        batch_advantages = torch.zeros(len(batch_advantages_list), max_length).type(dtype)
    batch_mask = []

    if batch_advantages_list:
        sorted_lengths, sorted_batch_states_list, sorted_batch_latent_c_list, sorted_batch_actions_list, sorted_batch_advantages_list = zip(*sorted(zip(lengths, batch_states_list, batch_latent_c_list , batch_actions_list, batch_advantages_list), key=lambda x: x[0], reverse=True))
    else:
        sorted_lengths, sorted_batch_states_list, sorted_batch_latent_c_list, sorted_batch_actions_list = zip(*sorted(zip(lengths, batch_states_list, batch_latent_c_list, batch_actions_list), key=lambda x: x[0], reverse=True))

    count = 0
    if batch_advantages_list:
        for l,s,c,a,ad in zip(sorted_lengths, sorted_batch_states_list, sorted_batch_latent_c_list, sorted_batch_actions_list, sorted_batch_advantages_list):
            batch_states[count, 0:l, :] = s
            batch_latent_c[count, 0:l, :] = c
            batch_actions[count, 0:l, :] = a
            batch_advantages[count, 0:l] = ad
            batch_mask += range(count*max_length, count*max_length+l)
            count += 1
    else:
        for l,s,c,a in zip(sorted_lengths, sorted_batch_states_list, sorted_batch_latent_c_list, sorted_batch_actions_list):
            batch_states[count, 0:l, :] = s
            batch_latent_c[count, 0:l, :] = c
            batch_actions[count, 0:l, :] = a
            batch_mask += range(count*max_length, count*max_length+l)
            count += 1

    batch_mask = torch.LongTensor(batch_mask).type(dtype_Long)
    if batch_advantages_list:
        batch_advantages.transpose_(0,1)
        return batch_states, batch_latent_c, batch_actions, batch_advantages, batch_mask, sorted_lengths
    else:
        return batch_states, batch_latent_c, batch_actions, batch_mask, sorted_lengths

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
    action = policy_net(Variable(state))
    return action

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
    return log_density.sum(1)


def update_params(gen_batch_list, expert_batch_list, i_episode, optim_epochs, optim_batch_size):
    criterion = nn.BCELoss()

    opt_value.lr = args.learning_rate*max(1.0 - float(i_episode)/args.num_episodes, 0)
    opt_policy.lr = args.learning_rate*max(1.0 - float(i_episode)/args.num_episodes, 0)
    clip_epsilon = args.clip_epsilon*max(1.0 - float(i_episode)/args.num_episodes, 0)

    optim_iters = int(math.ceil(len(gen_batch_list)/optim_batch_size))
    if len(expert_batch_list) < optim_iters: # replicate expert data if not enough
        expert_batch_list *= int(math.ceil(float(optim_iters)/len(expert_batch_list)))

    optim_batch_size_exp = int(math.floor(len(expert_batch_list)/optim_iters))

    # generated trajectories
    rewards_list = []
    masks_list = []
    actions_list = []
    states_list = []
    latent_c_list = []
    values_list = []

    advantages_list = []
    targets_list = []

    for batch in gen_batch_list:
        rewards = torch.Tensor(batch.reward).type(dtype)
        rewards_list.append(rewards)
        masks = torch.Tensor(batch.mask).type(dtype)
        masks_list.append(masks)
        actions = torch.Tensor(np.concatenate(batch.action, 0)).type(dtype)
        actions_list.append(actions)
        states = torch.Tensor(batch.state).type(dtype)
        states_list.append(states)
        latent_c = torch.Tensor(batch.c).type(dtype)
        latent_c_list.append(latent_c)
        values = value_net(Variable(torch.cat((states, latent_c),1)))

        returns = torch.Tensor(actions.size(0),1).type(dtype)
        deltas = torch.Tensor(actions.size(0),1).type(dtype)
        advantages = torch.Tensor(actions.size(0),1).type(dtype)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            #deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
            #advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
            advantages[i] = returns[i] # changed by me to see if value function is creating problems
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        targets = Variable(returns)

        advantages_list.append(advantages)
        targets_list.append(targets)

    stacked_advantages = torch.cat(advantages_list, 0) # will need to compute across all ...
    advantages_mean = stacked_advantages.mean()
    advantages_std = stacked_advantages.std()

    for i in range(len(advantages_list)):
        advantages_list[i] = (advantages_list[i] - advantages_mean) / advantages_std

    # expert trajectories
    expert_states_list = []
    expert_actions_list = []
    expert_latent_c_list = []

    for batch in expert_batch_list:
        states = torch.Tensor(batch.state).type(dtype)
        expert_states_list.append(states)
        latent_c = torch.Tensor(batch.c).type(dtype)
        expert_latent_c_list.append(latent_c)
        actions = torch.Tensor(np.concatenate(batch.action, 0)).type(dtype)
        expert_actions_list.append(actions)

    # backup params after computing probs but before updating new params
    for old_policy_param, policy_param in zip(old_policy_net.parameters(), policy_net.parameters()):
        old_policy_param.data.copy_(policy_param.data)


    for _ in range(optim_epochs):
        perm = range(len(gen_batch_list))
        random.shuffle(perm)
        exp_perm = range(len(expert_batch_list))
        random.shuffle(exp_perm)
        #perm = torch.LongTensor(perm)
        #states = states[perm]
        #actions = actions[perm]
        #values = values[perm]
        #targets = targets[perm]
        #advantages = advantages[perm]
        cur_id = 0
        cur_id_exp = 0
        for _ in range(optim_iters):
            cur_batch_size = min(optim_batch_size, len(gen_batch_list) - cur_id)
            cur_batch_size_exp = min(optim_batch_size_exp, len(expert_batch_list) - cur_id_exp)

            # reset nets
            policy_net.reset(cur_batch_size)
            old_policy_net.reset(cur_batch_size)

            # zero gradients
            opt_value.zero_grad()
            opt_policy.zero_grad()
            opt_reward.zero_grad()

            batch_states_list = [states_list[ep_i] for ep_i in perm[cur_id:cur_id+cur_batch_size]]
            batch_latent_c_list = [latent_c_list[ep_i] for ep_i in perm[cur_id:cur_id+cur_batch_size]]
            batch_actions_list = [actions_list[ep_i] for ep_i in perm[cur_id:cur_id+cur_batch_size]]
            batch_advantages_list = [advantages_list[ep_i] for ep_i in perm[cur_id:cur_id+cur_batch_size]]
            batch_targets_list = [targets_list[ep_i] for ep_i in perm[cur_id:cur_id+cur_batch_size]]

            # update value net
            batch_state_var = Variable(torch.cat(batch_states_list,0))
            batch_latent_c_var = Variable(torch.cat(batch_latent_c_list, 0))
            value_var = value_net(torch.cat((batch_state_var, batch_latent_c_var),1))
            targets_var = torch.cat(batch_targets_list,0)
            value_loss = (value_var - targets_var).pow(2.).mean()
            value_loss.backward()
           
            batch_expert_states_list = [expert_states_list[ep_i] for ep_i in exp_perm[cur_id_exp:cur_id_exp+cur_batch_size_exp]]
            batch_expert_latent_c_list = [expert_latent_c_list[ep_i] for ep_i in exp_perm[cur_id_exp:cur_id_exp+cur_batch_size_exp]]
            batch_expert_actions_list = [expert_actions_list[ep_i] for ep_i in exp_perm[cur_id_exp:cur_id_exp+cur_batch_size_exp]]
            padded_exp_states, padded_exp_latent_c, padded_exp_actions, padded_exp_mask, exp_lengths = create_batch_inputs(batch_expert_states_list, batch_expert_latent_c_list, batch_expert_actions_list)
            expert_state_var = Variable(padded_exp_states)
            expert_latent_c_var = Variable(padded_exp_latent_c)
            expert_action_var = Variable(padded_exp_actions)

            padded_states, padded_latent_c, padded_actions, padded_advantages, padded_mask, lengths = create_batch_inputs(batch_states_list, batch_latent_c_list, batch_actions_list, batch_advantages_list)
            state_var = Variable(padded_states)
            latent_c_var = Variable(padded_latent_c)
            action_var = Variable(padded_actions)
            advantages_var = Variable(padded_advantages)

            # update reward net
            # backprop with expert demonstrations
            outputs = []
            reward_net.reset(cur_batch_size_exp)
            for t in range(exp_lengths[0]):
                o = reward_net(torch.cat((expert_state_var[:,t,:], expert_action_var[:,t,:], expert_latent_c_var[:,t,:]),1))
                outputs.append(o)

            outputs = torch.stack(outputs, 0).view(-1)[padded_exp_mask]
            loss = criterion(outputs, Variable(torch.zeros(outputs.size())).type(dtype))
            loss.backward()

            # backprop with generated demonstrations
            outputs = []
            reward_net.reset(cur_batch_size)
            for t in range(lengths[0]):
                o = reward_net(torch.cat((state_var[:,t,:], action_var[:,t,:], latent_c_var[:,t,:]),1))
                outputs.append(o)

            outputs = torch.stack(outputs, 0).view(-1)[padded_mask]
            loss = criterion(outputs, Variable(torch.ones(outputs.size())).type(dtype))
            loss.backward()

            opt_reward.step()

            # update policy net
            ratio_list = []
            for t in range(lengths[0]):
                log_prob_cur = torch.sum(torch.mul(policy_net(torch.cat((state_var[:,t,:], latent_c_var[:,t,:]),1)), action_var[:,t,:]),1)

                log_prob_old = torch.sum(torch.mul(old_policy_net(torch.cat((state_var[:,t,:], latent_c_var[:,t,:]),1)), action_var[:,t,:]),1)

                ratio_list.append(torch.exp(log_prob_cur - log_prob_old)) # pnew / pold


            ratio = torch.stack(ratio_list, 0)
            surr1 = (ratio * advantages_var).view(-1)[padded_mask]
            surr2 = (torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var).view(-1)[padded_mask]
            policy_surr = -torch.min(surr1, surr2).mean() # true mean depends only on actual number of timesteps in batch
            policy_surr.backward()

            torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)

            opt_value.step()
            opt_policy.step()
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
    ep_memory = Memory_Ep()

    num_steps = 0
    reward_batch = 0
    true_reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        c = expert.sample_c() # read c sequence from expert trajectories
        if np.argmax(c[0,:]) == 1: # left half
            set_diff = list(set(product(tuple(range(0, (width/2)-3)), tuple(range(1, height)))) - set(obstacles))
        elif np.argmax(c[0,:]) == 3: # right half
            set_diff = list(set(product(tuple(range(width/2, width-2)), tuple(range(2, height)))) - set(obstacles))

        start_loc = sample_start(set_diff)
        s = State(start_loc, obstacles)
        #state = running_state(state)
        policy_net.reset()
        reward_net.reset()
        R.reset()

        reward_sum = 0
        true_reward_sum = 0
        memory = Memory()

        for t in range(args.max_ep_length): # Don't infinite loop while learning
            ct = c[t,:]
            action = select_action(np.concatenate((s.state, ct)))
            action = epsilon_greedy_linear_decay(action.data.cpu().numpy(), args.num_episodes*0.5, i_episode, low=0.05, high=0.9)
            reward = -float(reward_net(torch.cat((Variable(torch.from_numpy(s.state).unsqueeze(0)).type(dtype), Variable(torch.from_numpy(oned_to_onehot(action)).unsqueeze(0)).type(dtype), Variable(torch.from_numpy(ct).unsqueeze(0)).type(dtype)), 1)).data.cpu().numpy()[0,0])
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

        ep_memory.push(memory)
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum
        true_reward_batch += true_reward_sum

    optim_batch_size = min(num_episodes, max(10,int(num_episodes*optim_percentage)))
    reward_batch /= num_episodes
    true_reward_batch /= num_episodes
    gen_batch = ep_memory.sample()
    expert_batch = expert.sample_as_list(size=args.num_expert_trajs)

    update_params(gen_batch, expert_batch, i_episode, optim_epochs, optim_batch_size)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward {}\tAverage reward {}\tLast true reward {}\tAverage true reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch, true_reward_sum, true_reward_batch))

    stats['true_reward'].append(true_reward_batch)

    results_path = os.path.join(args.checkpoint, 'results.pkl')
    with open(results_path,'wb') as results_f:
        pickle.dump((stats), results_f, protocol=2)

    if i_episode % args.save_interval == 0:
        f_w = open(os.path.join(args.checkpoint, 'ep_' + str(i_episode) + '.pth'), 'wb')
        checkpoint = {'policy': policy_net}
        torch.save(checkpoint, f_w)

    if i_episode == args.num_episodes:
        break
