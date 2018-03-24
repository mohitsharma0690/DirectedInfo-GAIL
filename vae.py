import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from load_expert_traj import Expert
from grid_world import State, Action, TransitionFunction, RewardFunction, RewardFunction_SR2
from grid_world import create_obstacles, obstacle_movement, sample_start
from itertools import product

parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--expert-path', default="L_expert_trajectories/", metavar='G',
                    help='path to the expert trajectory files')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#-----Environment-----#
width = height = 12
obstacles = create_obstacles(width, height)

T = TransitionFunction(width, height, obstacle_movement)

if args.expert_path == 'SR2_expert_trajectories/':
    R = RewardFunction_SR2(-1.0,1.0,width)
else:
    R = RewardFunction(-1.0,1.0)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(10, 64)
        self.fc21 = nn.Linear(64, 1)
        self.fc22 = nn.Linear(64, 1)
        self.fc3 = nn.Linear(10, 64)
        self.fc4 = nn.Linear(64, 4)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        h1 = self.relu(self.fc1(torch.cat((x, c), 1)))
        #logvar = Variable(-4.0*torch.ones(h1.size(0),1))
        #if args.cuda:
        #    logvar = logvar.cuda()
        return self.fc21(h1), self.fc22(h1)
        #return self.fc21(h1), logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x, c):
        h3 = self.relu(self.fc3(torch.cat((x, c), 1)))
        return self.sigmoid(self.fc4(h3))
        #return self.fc4(h3)

    def forward(self, x_t0, x_t1, x_t2, x_t3, c):
        mu, logvar = self.encode(torch.cat((x_t0, x_t1, x_t2, x_t3), 1), c)
        c[:,0] = self.reparameterize(mu, logvar)
        return self.decode(torch.cat((x_t0, x_t1, x_t2, x_t3), 1), c), mu, logvar


model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    #MSE = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #KLD = 0.5 * torch.sum(mu.pow(2))

    return BCE + KLD
    #return MSE + KLD


def train(epoch, expert, Transition):
    model.train()
    train_loss = 0
    for batch_idx in range(10): # 10 batches per epoch
        batch = expert.sample(args.batch_size)
        x_data = torch.Tensor(batch.state)
        N = x_data.size(1)
        x = -1*torch.ones(x_data.size(0), 4, x_data.size(2))
        x[:,3,:] = x_data[:,0,:]

        a = Variable(torch.Tensor(batch.action))

        _, c2 = torch.Tensor(batch.c).max(2)
        c2 = c2.float()[:,0].unsqueeze(1)
        c1 = -1*torch.ones(c2.size())
        c = torch.cat((c1,c2),1)

        #c_t0 = Variable(c[:,0].clone().view(c.size(0), 1))

        if args.cuda:
            a = a.cuda()
            #c_t0 = c_t0.cuda()

        optimizer.zero_grad()
        for t in range(N):
            x_t0 = Variable(x[:,0,:].clone().view(x.size(0), x.size(2)))
            x_t1 = Variable(x[:,1,:].clone().view(x.size(0), x.size(2)))
            x_t2 = Variable(x[:,2,:].clone().view(x.size(0), x.size(2)))
            x_t3 = Variable(x[:,3,:].clone().view(x.size(0), x.size(2)))
            c_t0 = Variable(c)

            if args.cuda:
                x_t0 = x_t0.cuda()
                x_t1 = x_t1.cuda()
                x_t2 = x_t2.cuda()
                x_t3 = x_t3.cuda()
                c_t0 = c_t0.cuda()


            recon_batch, mu, logvar = model(x_t0, x_t1, x_t2, x_t3, c_t0)
            loss = loss_function(recon_batch, a[:,t,:], mu, logvar)
            loss.backward()
            train_loss += loss.data[0]

            pred_actions = recon_batch.data.cpu().numpy()

            x[:,:3,:] = x[:,1:,:]
            # get next state and update x
            for b_id in range(pred_actions.shape[0]):
                action = Action(np.argmax(pred_actions[b_id,:]))
                state = State(x[b_id,3,:].cpu().numpy(), obstacles)
                next_state = Transition(state, action, 0)
                x[b_id,3,:] = torch.Tensor(next_state.state)

            # update c
            c[:,0] = model.reparameterize(mu, logvar).data.cpu()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, 200.0,
                100. * batch_idx / 20.0,
                loss.data[0] / args.batch_size))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / 200.0))


def test(Transition):
    model.eval()
    #test_loss = 0

    for _ in range(20):
        c = expert.sample_c()
        N = c.shape[0]
        c = np.argmax(c[0,:])
        if args.expert_path == 'SR_expert_trajectories/':
            if c == 1:
                half = 0
            elif c == 3:
                half = 1
        elif args.expert_path == 'SR2_expert_trajectories/':
            half = c
        if args.expert_path == 'SR_expert_trajectories/' or args.expert_path == 'SR2_expert_trajectories/':
            if half == 0: # left half
                set_diff = list(set(product(tuple(range(0, (width/2)-3)), tuple(range(1, height)))) - set(obstacles))
            elif half == 1: # right half
                set_diff = list(set(product(tuple(range(width/2, width-2)), tuple(range(2, height)))) - set(obstacles))
        else:
            set_diff = list(set(product(tuple(range(3, width-3)), repeat=2)) - set(obstacles))

        start_loc = sample_start(set_diff)
        s = State(start_loc, obstacles)
        R.reset()
        c = torch.from_numpy(np.array([-1.0,c])).unsqueeze(0).float()

        print 'c is ', c[0,1]

        c = Variable(c)

        x = -1*torch.ones(1, 4, 2)

        if args.cuda:
            x = x.cuda()
            c = c.cuda()

        for t in range(N):

            x[:,:3,:] = x[:,1:,:]
            curr_x = torch.from_numpy(s.state).unsqueeze(0)
            if args.cuda:
                curr_x = curr_x.cuda()

            x[:,3:,:] = curr_x

            x_t0 = Variable(x[:,0,:])
            x_t1 = Variable(x[:,1,:])
            x_t2 = Variable(x[:,2,:])
            x_t3 = Variable(x[:,3,:])

            mu, logvar = model.encode(torch.cat((x_t0, x_t1, x_t2, x_t3), 1), c)
            c[:,0] = model.reparameterize(mu, logvar)
            pred_a = model.decode(torch.cat((x_t0, x_t1, x_t2, x_t3), 1), c).data.cpu().numpy()
            pred_a = np.argmax(pred_a)
            print pred_a
            next_s = Transition(s, Action(pred_a), R.t)

            s = next_s

            #test_loss += loss_function(recon_batch, data, mu, logvar).data[0]


    #test_loss /= len(test_loader.dataset)
    #print('====> Test set loss: {:.4f}'.format(test_loss))


expert = Expert(args.expert_path, 2)
expert.push()

for epoch in range(1, args.epochs + 1):
    train(epoch, expert, T)
    #test(epoch)
    #sample = Variable(torch.randn(64, 20))
    #if args.cuda:
    #    sample = sample.cuda()
    #sample = model.decode(sample).cpu()
    #save_image(sample.data.view(64, 1, 28, 28),
    #           'results/sample_' + str(epoch) + '.png')

test(T)
