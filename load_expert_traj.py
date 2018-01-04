import numpy as np
from collections import namedtuple
import os
from running_state import ZFilter
import random

Trajectory = namedtuple('Trajectory', ('state', 'action', 'c', 'mask'))

class Expert(object):
    def __init__(self, folder, num_inputs):
        self.memory = []
        self.pointer = 0
        self.n = len(os.listdir(folder))
        self.folder = folder
        self.list_of_sample_c = []
        #self.running_state = ZFilter((num_inputs,), clip=5)

    def push(self):
        """Saves a (state, action, c, mask) tuple."""
        for i in range(self.n):
            f = open(self.folder + str(i) + '.txt', 'r')
            line_counter = 0
            temp_mem = []
            temp_c = []
            for line in f:
                if line_counter % 3 == 0:
                    if line_counter > 0:
                        temp_mem.append(Trajectory(s, a, c, 1))
                    #s = self.running_state(np.asarray(line.strip().split(), dtype='float'))
                    s = np.asarray(line.strip().split(), dtype='float')
                elif line_counter % 3 == 1:
                    a = np.asarray(line.strip().split(), dtype='float')
                elif line_counter % 3 == 2:
                    c = np.asarray(line.strip().split(), dtype='float')
                    temp_c.append(c)

                line_counter += 1

            f.close()
            temp_mem.append(Trajectory(s, a, c, 0))
            self.memory.append(Trajectory(*zip(*temp_mem)))
            self.list_of_sample_c.append(np.array(temp_c))


    def sample(self, size=5):
        ind = np.random.randint(self.n, size=size)
        batch_list = []
        for i in ind:
            batch_list.append(self.memory[i])

        return Trajectory(*zip(*batch_list))

    def sample_as_list(self, size=5):
        ind = np.random.randint(self.n, size=size)
        batch_list = []
        for i in ind:
            batch_list.append(self.memory[i])

        return batch_list

    def sample_c(self):
        ind = random.randint(0, self.n-1)
        return self.list_of_sample_c[ind]
        
    #def sample_batch(self, batch_size):
    #    random_batch = random.sample(self.memory, batch_size)
    #    return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)

