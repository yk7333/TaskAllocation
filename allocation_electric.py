import numpy as np
from manager import Manager,ManagerReplayBuffer
from worker import Worker
from preAssign import PreAssign,PreAssignReplayBuffer,PreAssignNormalCritic

class Allocation:
    def __init__(self,args):
        self.epoch = 0
        self.manager_buffer_size = args.manager_buffer_size
        self.manager_min_size = args.manager_min_size
        self.manager_batch_size = args.manager_batch_size
        self.preassign_buffer_size = args.preassign_buffer_size
        self.preassign_min_size = args.preassign_min_size
        self.preassign_batch_size = args.preassign_batch_size
        self.hidden_dim = args.hidden_dim
        self.max_epoch = args.max_epoch
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.tau = args.tau
        self.sigma = args.worker_sigma
        self.sigma_decay = args.sigma_decay_rate
        self.device = args.device
        self.model_dir = args.model_dir
        self.agent_num = args.agent_num
        self.agent_dim = args.agent_dim
        self.task_num = args.task_num
        self.task_dim = args.task_dim
        self.avail_agent = [np.zeros(self.agent_num,dtype=np.int8) for _ in range(self.task_num)]
        self.agent_type = args.agent_type
        self.preassign_buffer = PreAssignReplayBuffer(self.preassign_buffer_size)
        self.preassign = PreAssign(self.task_dim,self.agent_dim,self.hidden_dim,self.actor_lr,self.critic_lr,self.alpha,self.tau,self.gamma,self.device,self.model_dir)
        self.manager_buffer = ManagerReplayBuffer(self.manager_buffer_size)
        self.manager = Manager(self.task_dim,self.agent_dim,self.hidden_dim,self.actor_lr,self.critic_lr,self.alpha,self.tau,self.gamma,self.device,self.agent_type,self.model_dir)
        
    
    def chose_agents(self, task_id, task,evaluate,render=True):
        avail_agents = self.avail_agent[task_id]
        idxs = np.where(avail_agents==1)[0]
        for idx in idxs:
            self.agent_type[idx,0] = (self.convey_cost[idx]+self.convey_cost[task_id])/2 * np.linalg.norm(self.agent_type[idx,2:]-self.task[task_id,2:])

        self.manager.reset_agent(self.agent_type.copy())
        state = task.copy()
        done = 0
        returns = 0
        action_list = []
        while not done:
            action = self.manager.select(state,avail_action=avail_agents,evaluate=evaluate) 
            next_state,reward,done,avail_agents = self.manager_step(action,task_id)
            transition_dict = {'states': state.copy(),'agent_type':self.agent_type.copy(),'actions': action,'next_states': next_state.copy(),'rewards': reward,'dones': done,'avail_agents':avail_agents}
            self.manager.update(transition_dict)
            state = next_state
            returns += reward
            action_list.append(action)    
        if returns<=0:
            action_list=[]
            returns = 0
        if render:
            # if len(self.worker.actor_loss_list)>0:
                # print("worker actor loss: {}".format(self.worker.actor_loss_list[-1])," worker critic loss: {}".format(self.worker.critic_loss_list[-1]))
            if len(self.manager.actor_loss_list)>0:
                print("manager{} actor loss: {}".format(task_id,self.manager.actor_loss_list[-1])," manager{} critic loss: {}".format(task_id,self.manager.critic_loss_list[-1]))
                print("manager{} accepts this task!".format(task_id)," allocation",action_list," return:",returns)
        return action_list,returns



    def select(self,state_,avail_task_,evaluate=False,render=False):
        state = state_.copy()
        self.task = state[:self.task_num*self.task_dim].reshape(self.task_num,self.task_dim)
        self.task_copy = self.task.copy()
        avail_task = (self.task.sum(axis=1)!=0) * avail_task_
        allocation = [[] for _ in range(self.task_num)]
        if sum(avail_task)==0:
            return allocation
        self.agent_type = state[self.task_num*self.task_dim:self.task_num*self.task_dim+self.agent_num*self.agent_dim].reshape(self.agent_num,self.agent_dim)
        avail_agents = state[self.task_num*self.task_dim+self.agent_num*self.agent_dim:self.task_num*self.task_dim+self.agent_num*self.agent_dim+self.agent_num]
        self.avail_agent = np.array(self.preassign.select(self.task.copy(),self.agent_type,avail_agents,avail_task,evaluate).T)
        self.convey_cost = state[self.task_num*self.task_dim+self.agent_num*self.agent_dim+self.agent_num:]
        pre_assign = self.avail_agent.copy()
        total_returns = 0
        for task_id,task in enumerate(self.task):
            if avail_task[task_id]==1:
                action_list,returns = self.chose_agents(task_id,task,evaluate,render)
                allocation[task_id].extend(action_list)
                total_returns += returns

        self.preassign_buffer.add(self.task_copy.copy(),pre_assign.T,total_returns,self.agent_type.copy(),avail_agents,avail_task)
        if self.preassign_buffer.size() > self.preassign_min_size:
            pre_states, pre_actions, pre_rewards, pre_agent_types,pre_avail_agents,pre_avail_tasks = self.preassign_buffer.sample(self.preassign_batch_size)
            transition_dict = {'states': pre_states,'actions': pre_actions,'rewards': pre_rewards,"agent_types":pre_agent_types,"avail_agents":pre_avail_agents,"avail_tasks":pre_avail_tasks}
            self.preassign.update(transition_dict)
        return allocation


    #manager的step, action是选出某一个agent去做任务
    def manager_step(self,action,task_id):
        true_action = self.agent_type[action]
        self.task[task_id][1:] = self.task[task_id][1:] - true_action[1:]
        self.task[task_id][1:] = np.where(self.task[task_id][1:]<=0,0,self.task[task_id][1:])
        self.avail_agent[task_id][action] = 0
        done = 0
        if np.all(self.task[task_id][1:]==0) or (sum(self.avail_agent[task_id])==0):
            done = 1
        reward = self.task[task_id][0]-true_action[0] if np.all(self.task[task_id][1:]==0) else -true_action[0]
        return self.task[task_id].copy(), reward, done, self.avail_agent[task_id].copy()
    
    def save(self):
        self.manager.save()
        self.preassign.save()

    def load(self):
        self.manager.load()
        self.preassign.load()


class SelfishAllocation:
    def __init__(self,args):
        self.epoch = 0
        self.manager_buffer_size = args.manager_buffer_size
        self.manager_min_size = args.manager_min_size
        self.manager_batch_size = args.manager_batch_size
        self.preassign_buffer_size = args.preassign_buffer_size
        self.preassign_min_size = args.preassign_min_size
        self.preassign_batch_size = args.preassign_batch_size
        self.hidden_dim = args.hidden_dim
        self.max_epoch = args.max_epoch
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.tau = args.tau
        self.sigma = args.worker_sigma
        self.sigma_decay = args.sigma_decay_rate
        self.device = args.device
        self.model_dir = args.model_dir
        self.agent_num = args.agent_num
        self.agent_dim = args.agent_dim
        self.task_num = args.task_num
        self.task_dim = args.task_dim
        self.worker_list = []
        self.worker_buffer_size = args.worker_buffer_size
        self.worker_min_size = args.worker_min_size
        self.worker_batch_size = args.worker_batch_size
        self.avail_agent = [np.zeros(self.agent_num,dtype=np.int8) for _ in range(self.task_num)]
        self.agent_type = args.agent_type
        self.preassign_buffer = PreAssignReplayBuffer(self.preassign_buffer_size)
        self.preassign = PreAssign(self.task_dim,self.agent_dim,self.hidden_dim,self.actor_lr,self.critic_lr,self.alpha,self.tau,self.gamma,self.device,self.model_dir)
        self.manager_buffer = ManagerReplayBuffer(self.manager_buffer_size)
        self.manager = Manager(self.task_dim,self.agent_dim,self.hidden_dim,self.actor_lr,self.critic_lr,self.alpha,self.tau,self.gamma,self.device,self.agent_type,self.model_dir)
        for _ in self.agent_type:
            w = Worker(self.agent_dim,self.hidden_dim,self.sigma,self.actor_lr,self.critic_lr,self.tau,self.gamma,self.device,self.model_dir,self.worker_buffer_size,self.sigma_decay)
            self.worker_list.append(w)
    
    def chose_agents(self, task_id, task,evaluate,render=True):
        avail_agents = self.avail_agent[task_id]
        init_avail_agents = avail_agents.copy()
        idxs = np.where(init_avail_agents==1)[0]

        state = task.copy()
        demands = np.zeros(len(self.agent_type))
        for idx in idxs:
            demands[idx] = self.worker_list[idx].demand(task.copy())

        self.agent_type[idxs,0] = demands[idxs].copy()
        self.manager.reset_agent(self.agent_type.copy())
        state = task.copy()
        done = 0
        returns = 0
        action_list = []
        agent_rewards = np.zeros(len(self.agent_type))
        while not done:
            action = self.manager.select(state,avail_action=avail_agents,evaluate=evaluate) 
            next_state,reward,done,avail_agents = self.manager_step(action,task_id)
            transition_dict = {'states': state.copy(),'agent_type':self.agent_type.copy(),'actions': action,'next_states': next_state.copy(),'rewards': reward,'dones': done,'avail_agents':avail_agents}
            self.manager.update(transition_dict)
            state = next_state
            returns += reward
            action_list.append(action)    
            agent_rewards[action] = self.agent_type[action,0]  
        if returns<=0:
            action_list=[]
            returns = 0
            agent_rewards = np.zeros(len(self.agent_type))
        for idx in idxs:
            # 更新报价网络
            transition_dict = {'states': self.task_copy[task_id].copy(),'actions': demands[idx].copy(),'rewards': agent_rewards[idx].copy()}
            self.worker_list[idx].update(transition_dict)
        if render:
            for idx in range(len(self.worker_list)):
                if len(self.worker_list[idx].actor_loss_list)>0:
                    print("worker {} actor loss: {}".format(idx, self.worker_list[idx].actor_loss_list[-1])," worker critic loss: {}".format(self.worker_list[idx].critic_loss_list[-1]))
            if len(self.manager.actor_loss_list)>0:
                print("manager{} actor loss: {}".format(task_id,self.manager.actor_loss_list[-1])," manager{} critic loss: {}".format(task_id,self.manager.critic_loss_list[-1]))
                print("manager{} accepts this task!".format(task_id)," allocation",action_list," return:",returns)
        return action_list,returns



    def select(self,state_,avail_task_,evaluate=False,render=False):
        state = state_.copy()
        self.task = state[:self.task_num*self.task_dim].reshape(self.task_num,self.task_dim)
        self.task_copy = self.task.copy()
        avail_task = (self.task.sum(axis=1)!=0) * avail_task_
        allocation = [[] for _ in range(self.task_num)]
        if sum(avail_task)==0:
            return allocation
        self.agent_type = state[self.task_num*self.task_dim:self.task_num*self.task_dim+self.agent_num*self.agent_dim].reshape(self.agent_num,self.agent_dim)
        avail_agents = state[self.task_num*self.task_dim+self.agent_num*self.agent_dim:]
        self.avail_agent = np.array(self.preassign.select(self.task.copy(),self.agent_type,avail_agents,avail_task,evaluate).T)
        pre_assign = self.avail_agent.copy()
        total_returns = 0
        for task_id,task in enumerate(self.task):
            if avail_task[task_id]==1:
                action_list,returns = self.chose_agents(task_id,task,evaluate,render)
                allocation[task_id].extend(action_list)
                total_returns += returns

        self.preassign_buffer.add(self.task_copy.copy(),pre_assign.T,total_returns,self.agent_type.copy(),avail_agents,avail_task)
        if self.preassign_buffer.size() > self.preassign_min_size:
            pre_states, pre_actions, pre_rewards, pre_agent_types,pre_avail_agents,pre_avail_tasks = self.preassign_buffer.sample(self.preassign_batch_size)
            transition_dict = {'states': pre_states,'actions': pre_actions,'rewards': pre_rewards,"agent_types":pre_agent_types,"avail_agents":pre_avail_agents,"avail_tasks":pre_avail_tasks}
            self.preassign.update(transition_dict)
        return allocation


    #manager的step, action是选出某一个agent去做任务
    def manager_step(self,action,task_id):
        true_action = self.agent_type[action]
        self.task[task_id][1:] = self.task[task_id][1:] - true_action[1:]
        self.task[task_id][1:] = np.where(self.task[task_id][1:]<=0,0,self.task[task_id][1:])
        self.avail_agent[task_id][action] = 0
        done = 0
        if np.all(self.task[task_id][1:]==0) or (sum(self.avail_agent[task_id])==0):
            done = 1
        reward = self.task[task_id][0]-true_action[0] if np.all(self.task[task_id][1:]==0) else -true_action[0]
        return self.task[task_id].copy(), reward, done, self.avail_agent[task_id].copy()
    
    def save(self):
        self.manager.save()
        self.preassign.save()

    def load(self):
        self.manager.load()
        self.preassign.load()

class AllocationWoPre:
    def __init__(self,args):
        self.epoch = 0
        self.manager_buffer_size = args.manager_buffer_size
        self.manager_min_size = args.manager_min_size
        self.manager_batch_size = args.manager_batch_size
        self.worker_buffer_size = args.worker_buffer_size
        self.worker_min_size = args.worker_min_size
        self.worker_batch_size = args.worker_batch_size
        self.hidden_dim = args.hidden_dim
        self.max_epoch = args.max_epoch
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.tau = args.tau
        self.sigma = args.worker_sigma
        self.sigma_decay = args.sigma_decay_rate
        self.device = args.device
        self.evaluate = args.evaluate
        self.model_dir = args.model_dir
        self.agent_num = args.agent_num
        self.agent_dim = args.agent_dim
        self.task_num = args.task_num
        self.task_dim = args.task_dim
        self.avail_agent = np.zeros(self.agent_num,dtype=np.int8)
        self.worker_list = []
        self.agent_type = args.agent_type
        self.random = args.random
        self.manager_buffer = ManagerReplayBuffer(self.manager_buffer_size)
        self.manager = Manager(self.task_dim,self.agent_dim,self.hidden_dim,self.actor_lr,self.critic_lr,self.alpha,self.tau,self.gamma,self.device,self.agent_type,self.model_dir)
        
    
    def chose_agents(self, task_id, task):
        avail_agents = self.avail_agent
        state = task.copy()
        idxs = np.where(avail_agents==1)[0]
        for idx in idxs:
            self.agent_type[idx,0] = (self.convey_cost[idx]+self.convey_cost[task_id])/2 * np.linalg.norm(self.agent_type[idx,2:]-self.task[task_id,2:])

        self.manager.reset_agent(self.agent_type.copy())
        done = 0
        returns = 0
        action_list = []
        while not done:
            action = self.manager.select(state,avail_action=avail_agents) 
            next_state,reward,done,avail_agents = self.manager_step(action,task_id)
            transition_dict = {'states': state.copy(),'agent_type':self.agent_type.copy(),'actions': action,'next_states': next_state.copy(),'rewards': reward,'dones': done,'avail_agents':avail_agents}
            self.manager.update(transition_dict)
            state = next_state
            returns += reward
            action_list.append(action)    
        if returns<=0:
            action_list=[]
            returns = 0
        else:
            self.avail_agent = avail_agents
        return action_list

    def select(self,state_,avail_task):
        state = state_.copy()
        self.task = state[:self.task_num*self.task_dim].reshape(self.task_num,self.task_dim)
        self.task_copy = self.task.copy()
        self.agent_type = state[self.task_num*self.task_dim:self.task_num*self.task_dim+self.agent_num*self.agent_dim].reshape(self.agent_num,self.agent_dim)
        avail_agents = state[self.task_num*self.task_dim+self.agent_num*self.agent_dim:self.task_num*self.task_dim+self.agent_num*self.agent_dim+self.agent_num]
        avail_idx = np.where(avail_agents==1)[0]
        self.convey_cost = state[self.task_num*self.task_dim+self.agent_num*self.agent_dim+self.agent_num:]
        self.avail_agent = np.zeros(self.agent_num,dtype=np.int8)
        self.avail_agent[avail_idx] = 1
        allocation = [[] for _ in range(self.task_num)]
        if self.random:
            shuffled_indices = np.random.permutation(self.task.shape[0])
            shuffle_task = self.task[shuffled_indices]
            for t_id,task in enumerate(shuffle_task):
                task_id = shuffled_indices[t_id]
                if avail_task[task_id]==1:
                    allocation[task_id].extend(self.chose_agents(task_id,task))
        else:
            for task_id,task in enumerate(self.task):
                if avail_task[task_id]==1:
                    allocation[task_id].extend(self.chose_agents(task_id,task))
        return allocation

    #manager的step, action是选出某一个agent去做任务
    def manager_step(self,action,task_id):
        true_action = self.agent_type[action]
        self.task[task_id][1:] = self.task[task_id][1:] - true_action[1:]
        self.task[task_id] = np.where(self.task[task_id]<=0,0,self.task[task_id])
        self.avail_agent[action] = 0
        done = 0
        if np.all(self.task[task_id][1:]==0) or (sum(self.avail_agent)==0):
            done = 1
        reward = self.task[task_id][0]-true_action[0] if np.all(self.task[task_id][1:]==0) else -true_action[0]
        return self.task[task_id].copy(), reward, done, self.avail_agent.copy()
    
    def save(self):
        self.manager.save()

class AllocationNormalCritic:
    def __init__(self,args):
        self.epoch = 0
        self.manager_buffer_size = args.manager_buffer_size
        self.manager_min_size = args.manager_min_size
        self.manager_batch_size = args.manager_batch_size
        self.preassign_buffer_size = args.preassign_buffer_size
        self.preassign_min_size = args.preassign_min_size
        self.preassign_batch_size = args.preassign_batch_size
        self.hidden_dim = args.hidden_dim
        self.max_epoch = args.max_epoch
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.tau = args.tau
        self.sigma = args.worker_sigma
        self.sigma_decay = args.sigma_decay_rate
        self.device = args.device
        self.model_dir = args.model_dir
        self.agent_num = args.agent_num
        self.agent_dim = args.agent_dim
        self.task_num = args.task_num
        self.task_dim = args.task_dim
        self.avail_agent = [np.zeros(self.agent_num,dtype=np.int8) for _ in range(self.task_num)]
        self.agent_type = args.agent_type
        self.preassign_buffer = PreAssignReplayBuffer(self.preassign_buffer_size)
        self.preassign = PreAssignNormalCritic(self.task_dim,self.agent_dim,self.hidden_dim,self.actor_lr,self.critic_lr,self.alpha,self.tau,self.gamma,self.device,self.model_dir,self.agent_num,self.task_num)
        self.manager_buffer = ManagerReplayBuffer(self.manager_buffer_size)
        self.manager = Manager(self.task_dim,self.agent_dim,self.hidden_dim,self.actor_lr,self.critic_lr,self.alpha,self.tau,self.gamma,self.device,self.agent_type,self.model_dir)
        
    
    def chose_agents(self, task_id, task,evaluate,render=True):
        avail_agents = self.avail_agent[task_id]
        state = task.copy()
        idxs = np.where(avail_agents==1)[0]
        for idx in idxs:
            self.agent_type[idx,0] = (self.convey_cost[idx]+self.convey_cost[task_id])/2 * np.linalg.norm(self.agent_type[idx,2:]-self.task[task_id,2:])

        self.manager.reset_agent(self.agent_type.copy())
        done = 0
        returns = 0
        action_list = []
        while not done:
            action = self.manager.select(state,avail_action=avail_agents,evaluate=evaluate) 
            next_state,reward,done,avail_agents = self.manager_step(action,task_id)
            transition_dict = {'states': state.copy(),'agent_type':self.agent_type.copy(),'actions': action,'next_states': next_state.copy(),'rewards': reward,'dones': done,'avail_agents':avail_agents}
            self.manager.update(transition_dict)
            state = next_state
            returns += reward
            action_list.append(action)    
        if returns<=0:
            action_list=[]
            returns = 0
        if render:
            # if len(self.worker.actor_loss_list)>0:
                # print("worker actor loss: {}".format(self.worker.actor_loss_list[-1])," worker critic loss: {}".format(self.worker.critic_loss_list[-1]))
            if len(self.manager.actor_loss_list)>0:
                print("manager{} actor loss: {}".format(task_id,self.manager.actor_loss_list[-1])," manager{} critic loss: {}".format(task_id,self.manager.critic_loss_list[-1]))
                print("manager{} accepts this task!".format(task_id)," allocation",action_list," return:",returns)
        return action_list,returns

    def select(self,state_,avail_task_,evaluate=False,render=False):
        state = state_.copy()
        self.task = state[:self.task_num*self.task_dim].reshape(self.task_num,self.task_dim)
        self.task_copy = self.task.copy()
        avail_task = (self.task.sum(axis=1)!=0) * avail_task_
        allocation = [[] for _ in range(self.task_num)]
        if sum(avail_task)==0:
            return allocation
        self.agent_type = state[self.task_num*self.task_dim:self.task_num*self.task_dim+self.agent_num*self.agent_dim].reshape(self.agent_num,self.agent_dim)
        avail_agents = state[self.task_num*self.task_dim+self.agent_num*self.agent_dim:self.task_num*self.task_dim+self.agent_num*self.agent_dim+self.agent_num]
        self.avail_agent = np.array(self.preassign.select(self.task.copy(),self.agent_type,avail_agents,avail_task,evaluate).T)
        pre_assign = self.avail_agent.copy()
        self.convey_cost = state[self.task_num*self.task_dim+self.agent_num*self.agent_dim+self.agent_num:]
        total_returns = 0
        for task_id,task in enumerate(self.task):
            if avail_task[task_id]==1:
                action_list,returns = self.chose_agents(task_id,task,evaluate,render)
                allocation[task_id].extend(action_list)
                total_returns += returns

        self.preassign_buffer.add(self.task_copy.copy(),pre_assign.T,total_returns,self.agent_type.copy(),avail_agents,avail_task)
        if self.preassign_buffer.size() > self.preassign_min_size:
            pre_states, pre_actions, pre_rewards, pre_agent_types,pre_avail_agents,pre_avail_tasks = self.preassign_buffer.sample(self.preassign_batch_size)
            transition_dict = {'states': pre_states,'actions': pre_actions,'rewards': pre_rewards,"agent_types":pre_agent_types,"avail_agents":pre_avail_agents,"avail_tasks":pre_avail_tasks}
            self.preassign.update(transition_dict)
        return allocation


    #manager的step, action是选出某一个agent去做任务
    def manager_step(self,action,task_id):
        true_action = self.agent_type[action]
        self.task[task_id][1:] = self.task[task_id][1:] - true_action[1:]
        self.task[task_id][1:] = np.where(self.task[task_id][1:]<=0,0,self.task[task_id][1:])
        self.avail_agent[task_id][action] = 0
        done = 0
        if np.all(self.task[task_id][1:]==0) or (sum(self.avail_agent[task_id])==0):
            done = 1
        reward = self.task[task_id][0]-true_action[0] if np.all(self.task[task_id][1:]==0) else -true_action[0]
        return self.task[task_id].copy(), reward, done, self.avail_agent[task_id].copy()
    
    def save(self):
        self.manager.save()
        self.preassign.save()

    def load(self):
        self.manager.load()
        self.preassign.load()

class AllocationNormal:
    def __init__(self,args):
        self.epoch = 0
        self.manager_buffer_size = args.manager_buffer_size
        self.manager_min_size = args.manager_min_size
        self.manager_batch_size = args.manager_batch_size
        self.preassign_buffer_size = args.preassign_buffer_size
        self.preassign_min_size = args.preassign_min_size
        self.preassign_batch_size = args.preassign_batch_size
        self.hidden_dim = args.hidden_dim
        self.max_epoch = args.max_epoch
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.tau = args.tau
        self.sigma = args.worker_sigma
        self.sigma_decay = args.sigma_decay_rate
        self.device = args.device
        self.model_dir = args.model_dir
        self.agent_num = args.agent_num
        self.agent_dim = args.agent_dim
        self.task_num = args.task_num
        self.task_dim = args.task_dim
        self.avail_agent = [np.zeros(self.agent_num,dtype=np.int8) for _ in range(self.task_num)]
        self.agent_type = args.agent_type
        self.preassign_buffer = PreAssignReplayBuffer(self.preassign_buffer_size)
        self.preassign = PreAssignNormalCritic(self.task_dim,self.agent_dim,self.hidden_dim,self.actor_lr,self.critic_lr,self.alpha,self.tau,self.gamma,self.device,self.model_dir,self.agent_num,self.task_num)
        self.manager_buffer = ManagerReplayBuffer(self.manager_buffer_size)
        self.manager = Manager(self.task_dim,self.agent_dim,self.hidden_dim,self.actor_lr,self.critic_lr,self.alpha,self.tau,self.gamma,self.device,self.agent_type,self.model_dir)
        
    
    def chose_agents(self, task_id, task,evaluate,render=True):
        avail_agents = self.avail_agent[task_id]
        state = task.copy()
        idxs = np.where(avail_agents==1)[0]
        for idx in idxs:
            self.agent_type[idx,0] = (self.convey_cost[idx]+self.convey_cost[task_id])/2 * np.linalg.norm(self.agent_type[idx,2:]-self.task[task_id,2:])

        self.manager.reset_agent(self.agent_type.copy())
        done = 0
        returns = 0
        action_list = []
        while not done:
            action = self.manager.select(state,avail_action=avail_agents,evaluate=evaluate) 
            next_state,reward,done,avail_agents = self.manager_step(action,task_id)
            transition_dict = {'states': state.copy(),'agent_type':self.agent_type.copy(),'actions': action,'next_states': next_state.copy(),'rewards': reward,'dones': done,'avail_agents':avail_agents}
            self.manager.update(transition_dict)
            state = next_state
            returns += reward
            action_list.append(action)    
        if returns<=0:
            action_list=[]
            returns = 0
        if render:
            # if len(self.worker.actor_loss_list)>0:
                # print("worker actor loss: {}".format(self.worker.actor_loss_list[-1])," worker critic loss: {}".format(self.worker.critic_loss_list[-1]))
            if len(self.manager.actor_loss_list)>0:
                print("manager{} actor loss: {}".format(task_id,self.manager.actor_loss_list[-1])," manager{} critic loss: {}".format(task_id,self.manager.critic_loss_list[-1]))
                print("manager{} accepts this task!".format(task_id)," allocation",action_list," return:",returns)
        return action_list,returns



    def select(self,state_,avail_task_,evaluate=False,render=False):
        state = state_.copy()
        self.task = state[:self.task_num*self.task_dim].reshape(self.task_num,self.task_dim)
        self.task_copy = self.task.copy()
        avail_task = (self.task.sum(axis=1)!=0) * avail_task_
        allocation = [[] for _ in range(self.task_num)]
        if sum(avail_task)==0:
            return allocation
        self.agent_type = state[self.task_num*self.task_dim:self.task_num*self.task_dim+self.agent_num*self.agent_dim].reshape(self.agent_num,self.agent_dim)
        avail_agents = state[self.task_num*self.task_dim+self.agent_num*self.agent_dim:self.task_num*self.task_dim+self.agent_num*self.agent_dim+self.agent_num]
        self.avail_agent = np.array(self.preassign.select(self.task.copy(),self.agent_type,avail_agents,avail_task,evaluate).T)
        pre_assign = self.avail_agent.copy()
        self.convey_cost = state[self.task_num*self.task_dim+self.agent_num*self.agent_dim+self.agent_num:]
        total_returns = 0
        for task_id,task in enumerate(self.task):
            if avail_task[task_id]==1:
                action_list,returns = self.chose_agents(task_id,task,evaluate,render)
                allocation[task_id].extend(action_list)
                total_returns += returns

        self.preassign_buffer.add(self.task_copy.copy(),pre_assign.T,total_returns,self.agent_type.copy(),avail_agents,avail_task)
        if self.preassign_buffer.size() > self.preassign_min_size:
            pre_states, pre_actions, pre_rewards, pre_agent_types,pre_avail_agents,pre_avail_tasks = self.preassign_buffer.sample(self.preassign_batch_size)
            transition_dict = {'states': pre_states,'actions': pre_actions,'rewards': pre_rewards,"agent_types":pre_agent_types,"avail_agents":pre_avail_agents,"avail_tasks":pre_avail_tasks}
            self.preassign.update(transition_dict)
        return allocation


    #manager的step, action是选出某一个agent去做任务
    def manager_step(self,action,task_id):
        true_action = self.agent_type[action]
        self.task[task_id][1:] = self.task[task_id][1:] - true_action[1:]
        self.task[task_id][1:] = np.where(self.task[task_id][1:]<=0,0,self.task[task_id][1:])
        self.avail_agent[task_id][action] = 0
        done = 0
        if np.all(self.task[task_id][1:]==0) or (sum(self.avail_agent[task_id])==0):
            done = 1
        reward = self.task[task_id][0]-true_action[0] if np.all(self.task[task_id][1:]==0) else -true_action[0]
        return self.task[task_id].copy(), reward, done, self.avail_agent[task_id].copy()
    
    def save(self):
        self.manager.save()
        self.preassign.save()

    def load(self):
        self.manager.load()
        self.preassign.load()