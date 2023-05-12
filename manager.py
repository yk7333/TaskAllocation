import numpy as np
import torch as th
import torch.nn as nn
import itertools
import collections
import random
import matplotlib.pyplot as plt

class ManagerReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  

    def add(self, state, agent_type, action, reward, next_state, done,avail_agents):  
        self.buffer.append((state, agent_type, action, reward, next_state, done, avail_agents))

    def sample(self, batch_size):  
        transitions = random.sample(self.buffer, batch_size)
        state, agent_type, action, reward, next_state, done, avail_agents = zip(*transitions)
        return np.array(state), np.array(agent_type), action, reward, np.array(next_state), done, avail_agents

    def size(self): 
        return len(self.buffer)

class Embedding(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Embedding,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,hidden_dim)
        self.relu = th.relu
        self.hidden_dim = hidden_dim

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc4(x)
        return x.reshape(-1,self.hidden_dim)
    
class ManagerNet(nn.Module):
    def __init__(self,agent_dim,task_dim,hidden_dim,agent_type,device):
        super(ManagerNet,self).__init__()
        self.agent_type = th.tensor(agent_type,dtype=th.float,device=device).squeeze(0)
        self.agent_embed = Embedding(agent_dim,hidden_dim)
        self.task_embed = Embedding(task_dim,hidden_dim)
        self.fc_task = nn.Sequential(nn.Linear(task_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim))
        self.fc_agent = nn.Sequential(nn.Linear(agent_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim))
        self.hidden_dim = hidden_dim
        self.device = device
    
    def reset_agent(self,agent_type):
        self.agent_type = th.tensor(agent_type,dtype=th.float,device=self.device).squeeze(0)

    def actor(self,task,avail_action):
        avail_action = th.tensor(avail_action,dtype=th.float,device=self.device) #batch_size, num_agents
        if len(avail_action.shape)==1:
            avail_action = avail_action.unsqueeze(0)
        task = th.tensor(task,dtype=th.float,device=self.device)    #batch_size, task_dim
        agent_embedding = self.agent_embed(self.agent_type) #num_agents,hidden_dim
        task_embedding = self.task_embed(task)  #batch_size, hidden_dim
        score = agent_embedding @ task_embedding.T /th.tensor(self.hidden_dim,dtype=th.long,device=self.device)    #num_agents,batch_size
        probs = th.softmax(th.where(avail_action.T==1,score,-999999.),dim=0)    #num_agents,batch_size
        return probs.T #batch_size,num_agents

    def critic(self,task):
        Q_value = self.fc_agent(self.agent_type) @ self.fc_task(task).reshape(-1,1)  #agent_num,hidden_dim @ hidden_dim,1 -> agent_num,1
        return Q_value.T


class Manager:
    def __init__(self,state_dim,agent_dim,hidden_dim,actor_lr,critic_lr,alpha,tau,gamma,device,agent_type,model_dir):
        self.net = ManagerNet(agent_dim,state_dim,hidden_dim,agent_type,device).to(device)
        self.target_net = ManagerNet(agent_dim,state_dim,hidden_dim,agent_type,device).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.loss = nn.MSELoss() 
        actor_params = itertools.chain(self.net.agent_embed.parameters(), self.net.task_embed.parameters())
        critic_params = itertools.chain(self.net.fc_task.parameters(), self.net.fc_agent.parameters())
        self.actor_optimizer = th.optim.Adam(actor_params,lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(critic_params,lr=critic_lr)
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.device=device
        self.critic_loss_list = []
        self.actor_loss_list = []
        self.model_dir = model_dir
    
    def reset_agent(self,agent_type):
        self.net.reset_agent(agent_type)
        self.target_net.reset_agent(agent_type)
    
    def select(self,state, avail_action,evaluate=False):
        probs = self.net.actor(state,avail_action)
        if evaluate:
            action = probs.argmax()
        else:
            action_dist = th.distributions.Categorical(probs)
            action = action_dist.sample() 
        return action.cpu().detach().item()
    
    def update(self,transition_dict):
        states = th.tensor(transition_dict['states'],dtype=th.float).flatten().to(self.device)
        agent_type = th.tensor(transition_dict['agent_type'],dtype=th.float).to(self.device)
        actions = th.tensor(transition_dict['actions'],dtype=th.long).to(self.device)  
        rewards = th.tensor(transition_dict['rewards'],dtype=th.float).to(self.device)
        next_states = th.tensor(transition_dict['next_states'],dtype=th.float).flatten().to(self.device)
        dones = th.tensor(transition_dict['dones'],dtype=th.float).to(self.device)
        avail_agents = th.tensor(transition_dict['avail_agents'],dtype=th.float).flatten().to(self.device)

        self.net.reset_agent(agent_type)
        self.target_net.reset_agent(agent_type)
        next_probs = self.net.actor(next_states, avail_agents)
        next_log_probs = th.log(next_probs + 1e-8)
        entropy = -th.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q_value = self.net.critic(next_states)
        v_value = th.sum(next_probs * q_value,dim=1,keepdim=True).flatten()
        next_value = v_value + 0.01 * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        critic_q_values = self.net.critic(states).reshape(1,-1).gather(1, actions.reshape(1,1))
        critic_loss = self.loss(td_target,critic_q_values)
        self.critic_loss_list.append(critic_loss.cpu().detach().item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        avail_agents.scatter_(0,actions,1)  #保存的是next_state的avail_agents,因此在当前state下的avail_agents需要补上被选中的agents
        probs = self.net.actor(states,avail_agents)
        log_probs = th.log(probs + 1e-8)
        entropy = -th.sum(probs * log_probs, dim=1, keepdim=True)  
        q_value = self.target_net.critic(states)
        v_value = th.sum(probs * q_value,dim=1,keepdim=True)  
        actor_loss = th.mean(-0.01 * entropy - v_value)
        self.actor_loss_list.append(actor_loss.cpu().detach().item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 
        self.soft_update(self.net, self.target_net)
        
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


    def save(self):
        model_path = self.model_dir + "/model.pt"
        th.save(self.net.state_dict(),model_path) 
    
    def load(self):
        model_path = self.model_dir + "/model.pt"
        self.net.load_state_dict(th.load(model_path))
        self.target_net.load_state_dict(th.load(model_path))
