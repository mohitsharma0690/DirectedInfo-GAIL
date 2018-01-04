import sys
import copy
import math
import random
import numpy as np
from models import Policy, Value, Reward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from itertools import product
import gym
from gym import wrappers
#import phase_envs.envs.AntEnv2 as AntEnv2
#from utils.normalized_env import NormalizedEnv
import matplotlib.pyplot as plt
from running_state import ZFilter

#USE_CUDA = torch.cuda.is_available()
#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dtype = torch.DoubleTensor



def select_action(state, policy_net):
	state = torch.from_numpy(state).unsqueeze(0)
	action_mean, _, action_std = policy_net(Variable(state))
	#action = torch.normal(action_mean, action_std)
	#return action
	return action_mean

def compute_friction(f, t):
	f[0,0] = 3*abs(math.sin((t)*math.pi/1000))
	return f

class Phase():
	def __init__(self):
		#self.phase_list = [0, math.pi/2, math.pi, 3*math.pi/2]
		self.n = 8
		#self.l = np.linspace(1,1.5,self.n/2) # hopper
		self.l = np.linspace(0.8,2.0,(self.n+2)/2) # walker
		#self.timer = 0
		#self.timer = 500

	#def comp_phase(self):
	#	if self.timer == 0:
	#		self.phase = random.choice(self.phase_list)
	#		self.timer += 1
	#	elif self.timer == 30:
	#		self.phase = random.choice(self.phase_list)
	#		self.timer = 1
	#	else:
	#		self.timer +=1

	#	return self.phase

	#def comp_phase(self):
	#	self.phase = (self.timer % 16)*math.pi/8
	#	self.timer += 1
	#	return self.phase

	# hopper
    	#def comp_phase(self, height, vel):
        #	if height <= 1.0:
	#        	phase = 0
        #	elif height > 1.5:
	#        	phase = math.pi
        #	else:
	#        	for i in range(self.n/2-1):
        #	        	if height > self.l[i] and height <= self.l[i+1]:
	#		                phase = (2*math.pi/self.n)*(i+1)
        #	if vel < 0:
	#	        phase = 2*math.pi - phase

	#        return phase

	# walker
	def comp_phase(self, height, vel):
        	if height <= 0.8:
        		phase = 0
	        elif height > 2.0:
        		phase = math.pi
	        else:
            		for i in range(self.n/2):
                		if height > self.l[i] and height <= self.l[i+1]:
                    			phase = (2*math.pi/self.n)*(i)
        	if vel < 0:
			phase = 2*math.pi - phase

	        return phase

    	#def comp_phase(self):
        #	phase = (2*self.timer*math.pi)/1000
        #	self.timer = (self.timer + 1) % 1000

        #	return phase

	def reset(self):
		self.timer = 0

def main():

	phase_obj = Phase()
	net_type = int(sys.argv[1])
	checkpoint = torch.load(sys.argv[2])
        actor = checkpoint['policy']
        #running_state = checkpoint['running_state']
	env = gym.make('Walker2d-v1')
	#env = wrappers.Monitor(env, './result_videos/')
	avg_total_reward = 0
	avg_step_count = 0

	print 'Using greedy policy ...'
	for ep in range(100):
		print ep
		#actor.reset()
		s = env.reset()
		#s = running_state(s)
		phase_obj.reset()
		total_reward = 0
		step_count = 0
		terminal = False
	        #list_of_height = []
        	#list_of_vel = []
		while terminal == False:
			#phase = phase_obj.comp_phase()
	        	#phase = phase_obj.comp_phase(env.env.env.model.data.qpos[1,0], env.env.env.model.data.qvel[1,0])
            		#fric = compute_friction(np.copy(env.env.env.model.geom_friction), phase_obj.timer)
		        #env.env.env.model.geom_friction = fric
	                #env.env.env.phase = phase
			#env.render()
                	#print step_count, env.env.env.model.data.qpos[1,0]
	                #list_of_height.append(env.env.env.env.model.data.qpos[1,0])
        	        #list_of_vel.append(env.env.env.model.data.qvel[1,0])
			if net_type == 0:
				a = select_action(s, actor).data.cpu().numpy()
				#x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
                                #a = actor.forward(x_a)[0].data.cpu().numpy()
				#x_c = Variable(torch.cat((a.data,torch.from_numpy(s)),1).type(dtype), requires_grad=False).unsqueeze(0)
				#q = critic.forward(x_c)
			elif net_type == 1:
				inp = np.concatenate((s,np.asarray([phase])))
				x_a = Variable(torch.from_numpy(inp).type(dtype), requires_grad=False).unsqueeze(0)
				a = actor.forward(x_a).data.cpu().numpy()
				#x_c = Variable(torch.cat((a.data,torch.from_numpy(inp)),1).type(dtype), requires_grad=False).unsqueeze(0)
				#q = critic.forward(x_c)
			elif net_type == 2:
				x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
				a = actor.forward(x_a, Variable(torch.from_numpy(np.asarray([[phase]])))).data.cpu().numpy()
				#x_c = Variable(torch.cat((a.data,torch.from_numpy(s)),1).type(dtype), requires_grad=False).unsqueeze(0)
				#q = critic.forward(x_c, phase)
			#elif net_type == 3:
			#	x_a = Variable(torch.from_numpy(s).type(dtype), requires_grad=False).unsqueeze(0)
			#	a = actor.forward(x_a).data.cpu().numpy()
			#	#x_c = Variable(torch.cat((a.data,torch.from_numpy(s)),1), requires_grad=False).unsqueeze(0)
			#	#q = critic.forward(x_c)
			#elif net_type == 4:
			#	inp = np.concatenate((s.state,np.asarray([phase])))
			#	x_a = Variable(torch.from_numpy(inp).type(dtype), requires_grad=False).unsqueeze(0)
			#	a = actor.forward(x_a).data.cpu().numpy()
			#	#x_c = Variable(torch.cat((a.data,torch.from_numpy(inp)),1).type(dtype), requires_grad=False).unsqueeze(0)
			#	#q = critic.forward(x_c)


        	        s_prime, reward, terminal, info = env.step(a)
			#s_prime = running_state(s_prime)
			total_reward += reward
			step_count += 1
        	        #print step_count
                	#raw_input()
			#if step_count >= 1000:
			#	print 'Episode length limit exceeded in greedy!'
			#	break
			s = s_prime

		print 'Total reward', total_reward
		print 'Number of steps', step_count
		avg_total_reward += total_reward
		avg_step_count += step_count


        #plt.plot(list_of_height)
        #plt.show()
        #list_of_acc = [0]
        #for i in range(1,len(list_of_vel)):
        #    list_of_acc.append(list_of_vel[i] - list_of_vel[i-1])
        #plt.plot(list_of_acc)
        #plt.show()

	print 'Average reward ', avg_total_reward/100.0
	print 'Average step count ', avg_step_count/100.0

if __name__ == '__main__':
	main()
