from allocation_Almighty import Allocation,AllocationWoPre
from utils import moving_average
import argparse
from envs.Almighty import Almighty
import matplotlib.pyplot as plt
import os 

parser = argparse.ArgumentParser(description='Allocations')
parser.add_argument('--buffer_size', type=int, default=1000)
parser.add_argument('--min_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--actor_lr', type=float, default=3e-5)
parser.add_argument('--critic_lr', type=float, default=1e-4)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--decay_rate', type=float, default=0.9995)
parser.add_argument('--epsilon_low', type=float, default=0.01)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--device',default="cuda:2")
parser.add_argument('--evaluate',action='store_true')
parser.add_argument('--model_dir',default="./models/Almighty")
parser.add_argument('--manager_buffer_size', type=int, default=1000)
parser.add_argument('--manager_min_size', type=int, default=100)
parser.add_argument('--manager_batch_size', type=int, default=1)
parser.add_argument('--worker_buffer_size', type=int, default=1000)
parser.add_argument('--worker_min_size', type=int, default=100)
parser.add_argument('--worker_batch_size', type=int, default=32)
parser.add_argument('--worker_sigma', type=float, default=0.5)
parser.add_argument('--sigma_decay_rate', type=float, default=0.999)
parser.add_argument('--preassign_buffer_size', type=int, default=1000)
parser.add_argument('--preassign_min_size', type=int, default=100)
parser.add_argument('--preassign_batch_size', type=int, default=32)
parser.add_argument('--pre',action='store_true')
parser.add_argument('--random',action='store_true')
args = parser.parse_args()
env = Almighty()
args.agent_num,args.agent_dim = env.agent_type.shape
args.task_num = env.task_num
args.task_dim = env.task_dim
args.agent_type = env.agent_type
os.makedirs(args.model_dir, exist_ok=True)
if args.pre:
    print("training with pre-assign")
    alg = Allocation(args)
else:
    if args.random:
        print("training with random method")
    else:
        print("training with sequence method")
    alg = AllocationWoPre(args)
return_list = []
manager_return_list = []
for epoch in range(args.max_epoch):
    done = 0
    state = env.reset()
    returns = 0
    action = alg.select(state)
    returns = env.step(action)
    print("epoch:{} return:{}".format(epoch,returns))
    return_list.append(returns)
plt.plot(moving_average(return_list,51))
