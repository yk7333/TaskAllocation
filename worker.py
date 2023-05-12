import numpy as np
import torch as th
import torch.nn as nn
import collections
import random

class WorkerReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  

    def add(self, state, action, reward):  
        self.buffer.append((state, action, reward))

    def sample(self, batch_size):  
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward = zip(*transitions)
        return np.array(state), action, reward

    def size(self): 
        return len(self.buffer)
    
    
class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = th.relu

    def forward(self, x):
        x = self.fc1(x)
        x = th.sigmoid(x)
        x = self.fc3(x)
        x = th.sigmoid(x)
        return x

class CriticNet(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)
        self.relu = th.relu
        
    def forward(self, x, a):
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        if len(a.shape)==1:
            a = a.unsqueeze(0)
        cat = th.cat([x, a], dim=-1) 
        x = self.fc1(cat)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x


class Worker:
    def __init__(self, state_dim, hidden_dim, sigma, actor_lr, critic_lr, tau, gamma, device, model_dir, buffer_size,sigma_decay):  
        self.actor = ActorNet(state_dim, hidden_dim).to(device)
        self.critic = CriticNet(state_dim+1, hidden_dim).to(device)
        self.target_actor = ActorNet(state_dim, hidden_dim).to(device)
        self.target_critic = CriticNet(state_dim+1, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma 
        self.tau = tau 
        self.device = device
        self.model_dir = model_dir
        self.loss = nn.MSELoss()
        self.actor_loss_list = []
        self.critic_loss_list = []
        self.buffer = WorkerReplayBuffer(buffer_size)
        self.sigma_decay = sigma_decay

    def demand(self, task):
        self.sigma = self.sigma * self.sigma_decay  if self.sigma>0.05 else 0.05
        task = th.tensor(task, dtype=th.float, device=self.device).reshape(1,-1)
        action = self.actor(task).item()
        action = action + self.sigma * np.random.randn()
        action = action if action>0 else 0 
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = th.tensor(transition_dict['states'], dtype=th.float, device=self.device)
        actions = th.tensor(transition_dict['actions'], dtype=th.float, device=self.device).reshape(-1, 1)
        rewards = th.tensor(transition_dict['rewards'], dtype=th.float, device=self.device).reshape(-1, 1)

        q_targets = rewards 
        critic_loss = self.loss(self.critic(states, actions), q_targets)
        self.critic_loss_list.append(critic_loss.cpu().detach().item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -th.mean(self.critic(states, self.actor(states)))
        self.actor_loss_list.append(actor_loss.cpu().detach().item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor) 
        self.soft_update(self.critic, self.target_critic)

    def save(self):
        actor_path = self.model_dir + "/worker_actor.pt"
        critic_path = self.model_dir + "/worker_critic.pt"
        th.save(self.actor.state_dict(),actor_path)
        th.save(self.critic.state_dict(),critic_path) 
        
    
    def load(self):
        actor_path = self.model_dir + "/worker_actor.pt"
        critic_path = self.model_dir + "/worker_critic.pt"
        self.actor.load_state_dict(th.load(actor_path))
        self.critic.load_state_dict(th.load(critic_path)) 
        

