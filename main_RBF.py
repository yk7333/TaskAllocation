from envs.RBF import RBF
import numpy as np
from allocation_RBF import Allocation,SelfishAllocation,AllocationWoPre,AllocationNormalCritic,AllocationNormal
import matplotlib.pyplot as plt
import argparse
from utils import take_action,moving_average
import os
import imageio

parser = argparse.ArgumentParser(description='Allocations')
parser.add_argument('--buffer_size', type=int, default=1000)
parser.add_argument('--min_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--actor_lr', type=float, default=1e-5)
parser.add_argument('--critic_lr', type=float, default=1e-5)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--decay_rate', type=float, default=0.9999)
parser.add_argument('--epsilon_low', type=float, default=0.01)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--device',default="cuda:5")
parser.add_argument('--evaluate',action='store_true')
parser.add_argument('--model_dir',default="./models/53_self")
parser.add_argument('--manager_buffer_size', type=int, default=1000)
parser.add_argument('--manager_min_size', type=int, default=100)
parser.add_argument('--manager_batch_size', type=int, default=32)
parser.add_argument('--worker_buffer_size', type=int, default=1000)
parser.add_argument('--worker_min_size', type=int, default=100)
parser.add_argument('--worker_batch_size', type=int, default=1)
parser.add_argument('--worker_sigma', type=float, default=0.5)
parser.add_argument('--sigma_decay_rate', type=float, default=0.9999)
parser.add_argument('--preassign_buffer_size', type=int, default=1000)
parser.add_argument('--preassign_min_size', type=int, default=100)
parser.add_argument('--preassign_batch_size', type=int, default=32)
parser.add_argument('--selfish',action='store_true')
parser.add_argument('--nopre',action='store_true')
parser.add_argument('--noattn',action='store_true')
parser.add_argument('--nomix',action='store_true')
parser.add_argument('--random',action='store_true')
parser.add_argument('--info',default='attn')
args = parser.parse_args()
env = RBF()
args.agent_num,args.agent_dim = env.agent_type.shape
args.task_num = env.task_num
args.task_dim = env.task_dim
args.agent_type = env.agent_type
information = args.info
selfish = args.selfish
preassign = not args.nopre
attn = not args.noattn
mix_critic =  not args.nomix
random = args.random
if selfish:
    print("training with selfish agents!")
    alg = SelfishAllocation(args)
elif preassign:
    if attn and mix_critic:
        print("training!")
        alg = Allocation(args)
    elif not mix_critic:
        print("training w/o AMIX critic!")
        alg = AllocationNormalCritic(args)
    else:
        print("training w/o TAM!")
        alg = AllocationNormal(args)
else:
    print(f"training w/o Pre-assign! random:{random}")
    alg = AllocationWoPre(args)


if not args.evaluate:
    return_list = []
    total_return_list = []
    for epoch in range(args.max_epoch):
        render = (epoch%50 ==49)
        if render:
            print("##########################################epoch: {} ####################################################".format(epoch+1))
        done = 0
        state,info = env.reset()
        returns = 0
        total_returns = 0
        time = 0
        avail_task = np.ones(env.task_num)
        last_allocation = [[] for _ in range(env.task_num)]
        while not done:  
            if sum(avail_task)>0:  
                allocation = alg.select(state,avail_task)
                for idx,each in enumerate(last_allocation):
                    if avail_task[idx] ==0:
                        allocation[idx] = last_allocation[idx]

            time+=1
            action = take_action(allocation,info)
            next_state,reward,done,info = env.step(action)
            for idx,each in enumerate(allocation):
                if len(each)>0 and avail_task[idx]==1:
                    avail_task[idx]=0 
                if (not np.array_equal(state[env.task_dim*idx:env.task_dim*(idx+1)], next_state[env.task_dim*idx:env.task_dim*(idx+1)])) and (avail_task[idx]==0):
                    avail_task[idx]=1

            state = next_state
            returns += reward
            total_returns += info["total_reward"]
            last_allocation = allocation
        print("episode:{} manager_return:{} total_return:{}  length:{}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(epoch,returns,total_returns,env.epoch))
        return_list.append(returns)
        total_return_list.append(total_returns)

    smooth = 501
    plt.figure(figsize=(8,6))
    plt.plot(moving_average(alg.manager.critic_loss_list,smooth))
    plt.savefig(args.model_dir+"/manager_critic_loss_{}.png".format(information))
    plt.figure(figsize=(8,6))
    plt.plot(moving_average(alg.manager.actor_loss_list,smooth))
    plt.savefig(args.model_dir+"/manager_actor_loss_{}.png".format(information))
    plt.figure(figsize=(8,6))
    plt.plot(moving_average(return_list,11))
    plt.savefig(args.model_dir+"/return_{}.png".format(information))
    plt.figure(figsize=(8,6))
    plt.plot(moving_average(total_return_list,11))
    plt.savefig(args.model_dir+"/total_return_{}.png".format(information))
    np.save(args.model_dir+"/manager_critic_loss_{}.npy".format(information),alg.manager.critic_loss_list)
    np.save(args.model_dir+"/manager_actor_loss_{}.npy".format(information),alg.manager.actor_loss_list)
    np.save(args.model_dir+"/return_{}.npy".format(information),return_list)
    np.save(args.model_dir+"/total_return_{}.npy".format(information),total_return_list)
    alg.save()
    if preassign:
        plt.figure(figsize=(8,6))
        plt.plot(moving_average(alg.preassign.critic_loss_list,smooth))
        plt.savefig(args.model_dir+"/preassign_critic_loss_{}.png".format(information))
        plt.figure(figsize=(8,6))
        plt.plot(moving_average(alg.preassign.actor_loss_list,smooth))
        plt.savefig(args.model_dir+"/preassign_actor_loss_{}.png".format(information))
    if selfish:
        for idx in range(len(alg.worker_list)):
            plt.figure(figsize=(8,6))
            plt.plot(moving_average(alg.worker_list[idx].critic_loss_list,smooth))
            plt.savefig(args.model_dir+"/worker{}_critic_loss_{}.png".format(idx,information))
            plt.figure(figsize=(8,6))
            plt.plot(moving_average(alg.worker_list[idx].actor_loss_list,smooth))
            plt.savefig(args.model_dir+"/worker{}_actor_loss_{}.png".format(idx,information))
    
else:
    alg.load()
    return_list = []
    done = 0
    state,info = env.reset()
    returns = 0
    time = 0
    avail_task = np.ones(env.task_num)
    last_allocation = [[] for _ in range(env.task_num)]
    while not done:  
        env.render()
        if sum(avail_task)>0:  
            allocation = alg.select(state,avail_task)
            for idx,each in enumerate(last_allocation):
                if avail_task[idx] ==0:
                    allocation[idx] = last_allocation[idx]

        time+=1
        action = take_action(allocation,info)
        next_state,reward,done,info = env.step(action)
        for idx,each in enumerate(allocation):
            if len(each)>0 and avail_task[idx]==1:
                avail_task[idx]=0 
            if (not np.array_equal(state[env.task_dim*idx:env.task_dim*(idx+1)], next_state[env.task_dim*idx:env.task_dim*(idx+1)])) and (avail_task[idx]==0):
                avail_task[idx]=1

        state = next_state
        returns += reward
        last_allocation = allocation

    frames = []
    
    ls = os.listdir("./img")
    ls.sort()
    for image_name in ls: 
        image_name = "./img/" + image_name 
        frames.append(imageio.imread(image_name))

    imageio.mimsave("./res_{}.gif".format(information), frames, fps=3) 
    print("zero-shot total return:{} length:{}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(returns,env.epoch))
    print("now start few-shot")
    return_list = []
    for epoch in range(args.max_epoch//5):
        done = 0
        state,info = env.reset()
        returns = 0
        time = 0
        avail_task = np.ones(env.task_num)
        last_allocation = [[] for _ in range(env.task_num)]
        while not done:  
            if sum(avail_task)>0:  
                allocation = alg.select(state,avail_task)
                for idx,each in enumerate(last_allocation):
                    if avail_task[idx] ==0:
                        allocation[idx] = last_allocation[idx]
            time+=1
            action = take_action(allocation,info)
            next_state,reward,done,info = env.step(action)
            for idx,each in enumerate(allocation):
                if len(each)>0 and avail_task[idx]==1:
                    avail_task[idx]=0 

                if (not np.array_equal(state[env.task_dim*idx:env.task_dim*(idx+1)], next_state[env.task_dim*idx:env.task_dim*(idx+1)])) and (avail_task[idx]==0):
                    avail_task[idx]=1

            state = next_state
            returns += reward
            last_allocation = allocation
        print("episode:{} total return:{} length:{}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(epoch,returns,env.epoch))
        return_list.append(returns)

    plt.figure(figsize=(8,6))
    plt.plot(moving_average(return_list,9))
    plt.savefig(args.model_dir+"/fewshot-return_{}.png".format(information))
    plt.figure(figsize=(8,6))
    np.save(args.model_dir+"/fewshot-return_{}.npy".format(information),return_list)