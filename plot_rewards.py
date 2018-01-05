import pickle
import sys
import matplotlib.pyplot as plt

pickle_path_gail = sys.argv[1]
pickle_path_digail = sys.argv[2]

gail_stats = pickle.load(open(pickle_path_gail, 'rb'))
digail_stats = pickle.load(open(pickle_path_digail, 'rb'))

gail_rewards_smooth = []
digail_rewards_smooth = []

N = 5
for i in range(0, len(gail_stats['true_reward']), N):
    gail_rewards_smooth.append(sum(gail_stats['true_reward'][i*N : (i+1)*N])/N)

for i in range(0, len(digail_stats['true_reward']), N):
    digail_rewards_smooth.append(sum(digail_stats['true_reward'][i*N : (i+1)*N])/N)


plt.plot(range(0, len(gail_stats['true_reward']), 5), gail_rewards_smooth, label='GAIL')
plt.plot(range(0, len(digail_stats['true_reward']), 5), digail_rewards_smooth, label='DIGAIL')
plt.legend()
plt.show()
