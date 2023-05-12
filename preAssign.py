import numpy as np
import torch as th
import torch.nn as nn
import collections
import random

class PreAssignReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  

    def add(self, state, action, reward, agent_type,avail_agent,avail_task):  
        self.buffer.append((state, action, reward, agent_type,avail_agent,avail_task))

    def sample(self, batch_size):  
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, agent_type,avail_agent,avail_task = zip(*transitions)
        return np.array(state), action, reward, agent_type,avail_agent,avail_task

    def size(self): 
        return len(self.buffer)

class Embedding(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Embedding,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.relu = th.relu
        self.fc4 = nn.Linear(hidden_dim,hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Actor(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.relu = th.relu
        self.hidden_dim = hidden_dim

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Critic(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.relu = th.relu
        self.hidden_dim = hidden_dim

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PreAssignActor(nn.Module):
    def __init__(self,task_dim,agent_dim,hidden_dim,device):
        super(PreAssignActor,self).__init__()
        self.agent_embed = Embedding(agent_dim,hidden_dim)
        self.task_embed = Embedding(task_dim,hidden_dim)
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, tasks, agent_type, avail_task):
        avail_task = th.tensor(avail_task,device=self.device)
        agent_embedding = self.agent_embed(agent_type).reshape(-1,agent_type.shape[-2],self.hidden_dim) #batch_size,agent_num,agent_dim -> batch_size,agent_num,hidden_dim
        task_embedding = self.task_embed(tasks).reshape(-1,tasks.shape[-2],self.hidden_dim)  # batch_size,task_num,hidden_dim
        # (batch_size,agent_num,hidden_dim) @ (task_num,task_num,hidden_dim).T ->(batch_size,agent_num,task_num) ->(batch_size,task_num,agent_num) avail_action:(batch_size,task_num,agent_num)
        logits = th.softmax(th.where(avail_task.unsqueeze(-1).repeat(1,1,agent_type.shape[-2])==1,th.bmm(agent_embedding, task_embedding.permute(0,2,1)).permute(0,2,1)/th.tensor(self.hidden_dim,dtype=th.long,device=self.device).flatten(),-999999.),dim=1)
        return logits.permute(0,2,1)  # (batch_size,agent_num,task_num)
    
class PreAssignCritic(nn.Module):
    def __init__(self,task_dim,agent_dim,hidden_dim,device):
        super(PreAssignCritic,self).__init__()
        self.agent_embed = Embedding(agent_dim,hidden_dim)
        self.task_embed = Embedding(task_dim,hidden_dim)
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self,tasks,agent_type):
        # batch_size,agent_num,hidden_dim @ batch_size,hidden_dim,task_num -> batch_size,agent_num,task_num
        return th.bmm(self.agent_embed(agent_type),self.task_embed(tasks).permute(0,2,1))

class MIX(nn.Module):
    def __init__(self,agent_dim,hidden_dim):
        super(MIX,self).__init__()
        self.q =  nn.Sequential(nn.Linear(agent_dim,hidden_dim),nn.LeakyReLU(0.1),nn.Linear(hidden_dim,hidden_dim))
        self.k =  nn.Sequential(nn.Linear(agent_dim,hidden_dim),nn.LeakyReLU(0.1),nn.Linear(hidden_dim,hidden_dim))
        self.v1 = nn.Sequential(nn.Linear(agent_dim,hidden_dim),nn.LeakyReLU(0.1),nn.Linear(hidden_dim,hidden_dim))
        self.v2 = nn.Sequential(nn.Linear(agent_dim,hidden_dim),nn.LeakyReLU(0.1),nn.Linear(hidden_dim,1))
        self.fc = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.LeakyReLU(0.1),nn.Linear(hidden_dim,1))
        self.hidden_dim = hidden_dim
        self.relu = nn.LeakyReLU(0.1)

    def forward(self,agent_type,value,avail_agents):     #agent_type:batch_size,agent_num,agent_dim  value:batch_size,agent_num
        agent_type = agent_type*avail_agents.unsqueeze(-1)  #batch_size,agent_num,agent_dim
        key = self.k(agent_type)            #batch_size,agent_num,hidden_dim
        query = self.q(agent_type)          #batch_size,agent_num,hidden_dim
        score = th.softmax(th.bmm(key,query.permute(0,2,1))/th.sqrt(th.tensor(self.hidden_dim,dtype=th.int,device=agent_type.device)),dim=-1) #batch_size,agent_num,agent_num
        value_w = self.v1(agent_type)          #batch_size,agent_num,hidden_dim
        w = th.bmm(score,value_w)        #batch_size,agent_num,hidden_dim
        x = th.bmm(value.unsqueeze(1),w)       #x:batch_size,1,hidden_dim
        x = self.relu(x)
        x = self.fc(x)                         #x:batch_size,1,1
        return x.reshape(-1,1)

class PreAssign:
    def __init__(self,task_dim,agent_dim,hidden_dim,actor_lr,critic_lr,alpha,tau,gamma,device,model_dir):
        self.actor = PreAssignActor(task_dim,agent_dim,hidden_dim,device).to(device)
        self.critic = PreAssignCritic(task_dim,agent_dim,hidden_dim,device).to(device)
        self.mix = MIX(agent_dim,hidden_dim).to(device)
        self.target_critic = PreAssignCritic(task_dim,agent_dim,hidden_dim,device).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.loss = nn.MSELoss() 
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.mix_optimizer = th.optim.Adam(self.mix.parameters(),lr=critic_lr)
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.critic_loss_list = []
        self.actor_loss_list = []
        self.model_dir = model_dir
        self.device = device
    

    def select(self, tasks, agent_type, avail_agent, avail_task,evaluate=False):
        avail_agent = th.tensor(avail_agent)
        tasks = th.tensor(tasks,dtype=th.float,device=self.device)
        agent_type = th.tensor(agent_type,dtype=th.float,device=self.device)
        probs = self.actor(tasks, agent_type, avail_task).squeeze(0)
        na,nt = probs.shape
        if evaluate:
            action = probs.argmax(axis=-1)
        else:
            action_dist = th.distributions.Categorical(probs)
            action = action_dist.sample()
        action = th.eye(nt)[action.cpu()]   #na,nt
        action = action * avail_agent.unsqueeze(-1) #na,nt
        return action.cpu().detach()
    
    def update(self,transition_dict):
        rewards = th.tensor(transition_dict['rewards'],dtype=th.float).flatten().to(self.device)
        batch_size = len(rewards)
        states = th.tensor(transition_dict['states'],dtype=th.float).reshape(batch_size,-1,self.task_dim).to(self.device)
        actions = th.tensor(transition_dict['actions'],dtype=th.long).to(self.device)  
        agent_types = th.tensor(transition_dict['agent_types'],dtype=th.float).to(self.device)
        avail_agents = th.tensor(transition_dict['avail_agents'],dtype=th.float).reshape(batch_size,-1).to(self.device)
        avail_tasks = th.tensor(transition_dict['avail_tasks'],dtype=th.float).reshape(batch_size,-1).to(self.device)

        td_target = rewards
        critic_q_values = (self.critic(states, agent_types)*actions).sum(-1) # batch_size,agent_num,task_num-> batch_size,agent_num
        critic_total_q_values = self.mix(agent_types,critic_q_values,avail_agents).reshape(td_target.shape) # batch_size,1
        critic_loss = self.loss(td_target,critic_total_q_values)
        self.critic_loss_list.append(critic_loss.cpu().detach().item())
        self.critic_optimizer.zero_grad()
        self.mix_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        self.mix_optimizer.step()
        
        probs = self.actor(states,agent_types,avail_tasks)
        log_probs = th.log(probs + 1e-8)
        entropy = -th.sum(probs.sum(-1) * log_probs.sum(-1), dim=1, keepdim=True)  
        q_values = (self.target_critic(states, agent_types)*actions)
        v_value = th.sum(probs * q_values, dim=-1)  
        total_v_value = self.mix(agent_types,v_value,avail_agents)
        actor_loss = th.mean(-0.01 * entropy - total_v_value)
        self.actor_loss_list.append(actor_loss.cpu().detach().item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 
        self.soft_update(self.critic, self.target_critic)
        
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


    def save(self):
        actor_path = self.model_dir + "/actor.pt"
        critic_path = self.model_dir + "/critic.pt"
        mix_path = self.model_dir + "/mix.pt"
        th.save(self.actor.state_dict(),actor_path)
        th.save(self.critic.state_dict(),critic_path) 
        th.save(self.mix.state_dict(),mix_path) 
        
    
    def load(self):
        actor_path = self.model_dir + "/actor.pt"
        critic_path = self.model_dir + "/critic.pt"
        mix_path = self.model_dir + "/mix.pt"
        self.actor.load_state_dict(th.load(actor_path))
        self.critic.load_state_dict(th.load(critic_path))
        self.mix.load_state_dict(th.load(mix_path))

class PreAssignNormalCritic:
    def __init__(self,task_dim,agent_dim,hidden_dim,actor_lr,critic_lr,alpha,tau,gamma,device,model_dir,agent_num,task_num):
        self.actor = PreAssignActor(task_dim,agent_dim,hidden_dim,device).to(device)
        self.critic = Critic(task_dim*task_num+agent_dim*agent_num+agent_num*task_num,hidden_dim,1).to(device)
        self.target_critic = Critic(task_dim*task_num+agent_dim*agent_num+agent_num*task_num,hidden_dim,1).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.loss = nn.MSELoss() 
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.critic_loss_list = []
        self.actor_loss_list = []
        self.model_dir = model_dir
        self.device = device
        self.bias = 0.5
    

    def select(self, tasks, agent_type, avail_agent, avail_task,evaluate=False):
        avail_agent = th.tensor(avail_agent)
        tasks = th.tensor(tasks,dtype=th.float,device=self.device)
        agent_type = th.tensor(agent_type,dtype=th.float,device=self.device)
        probs = self.actor(tasks, agent_type, avail_task).squeeze(0)
        na,nt = probs.shape
        if evaluate:
            action = probs.argmax(axis=-1)
        else:
            action_dist = th.distributions.Categorical(probs)
            action = action_dist.sample()
        action = th.eye(nt)[action.cpu()]   #na,nt
        action = action * avail_agent.unsqueeze(-1) #na,nt
        return action.cpu().detach()
    
    def update(self,transition_dict):
        rewards = th.tensor(transition_dict['rewards'],dtype=th.float).flatten().to(self.device)
        batch_size = len(rewards)
        states = th.tensor(transition_dict['states'],dtype=th.float).reshape(batch_size,-1,self.task_dim).to(self.device)
        actions = th.tensor(transition_dict['actions'],dtype=th.long).to(self.device)  
        agent_types = th.tensor(transition_dict['agent_types'],dtype=th.float).to(self.device)
        avail_tasks = th.tensor(transition_dict['avail_tasks'],dtype=th.float).reshape(batch_size,-1).to(self.device)

        td_target = rewards
        critic_q_values = self.critic(th.cat((states.reshape(batch_size,-1), agent_types.reshape(batch_size,-1),actions.reshape(batch_size,-1)),dim=-1)) # batch_size,agent_num,task_num-> batch_size,agent_num
        critic_loss = self.loss(td_target,critic_q_values)
        self.critic_loss_list.append(critic_loss.cpu().detach().item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        
        probs = self.actor(states,agent_types,avail_tasks)+self.bias
        probs[probs>1]=1
        prob_action = (probs*actions).sum(-1)
        non_zero_rows = [row[row != 0] for row in prob_action]
        non_zero_rows = [row if row.numel() > 0 else th.tensor([1.]) for row in non_zero_rows]
        row_products = [th.prod(row).to(self.device) for row in non_zero_rows]
        prob = th.stack(row_products).reshape(-1, 1)
        log_probs = th.log(prob + 1e-8)
        entropy = -th.sum(prob * log_probs, dim=1, keepdim=True)  
        q_values = (self.target_critic(th.cat((states.reshape(batch_size,-1), agent_types.reshape(batch_size,-1),actions.reshape(batch_size,-1)),dim=-1)))
        v_value = th.sum(prob * q_values, dim=-1)  
        actor_loss = th.mean(-0.01 * entropy - v_value)
        self.actor_loss_list.append(actor_loss.cpu().detach().item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 
        self.soft_update(self.critic, self.target_critic)
        
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


    def save(self):
        actor_path = self.model_dir + "/actor_normal_critic.pt"
        critic_path = self.model_dir + "/critic_normal_critic.pt"
        mix_path = self.model_dir + "/mix_normal_critic.pt"
        th.save(self.actor.state_dict(),actor_path)
        th.save(self.critic.state_dict(),critic_path) 
        th.save(self.mix.state_dict(),mix_path) 
        
    
    def load(self):
        actor_path = self.model_dir + "/actor_normal_critic.pt"
        critic_path = self.model_dir + "/critic_normal_critic.pt"
        mix_path = self.model_dir + "/mix_normal_critic.pt"
        self.actor.load_state_dict(th.load(actor_path))
        self.critic.load_state_dict(th.load(critic_path))
        self.mix.load_state_dict(th.load(mix_path))

class PreAssignNormal:
    def __init__(self,task_dim,agent_dim,hidden_dim,actor_lr,critic_lr,alpha,tau,gamma,device,model_dir,agent_num,task_num):
        self.actor = Actor(task_dim*task_num+agent_dim*agent_num,agent_num*task_num,hidden_dim,device).to(device)
        self.critic = Critic(task_dim*task_num+agent_dim*agent_num,hidden_dim,task_num*agent_num).to(device)
        self.mix = MIX(agent_dim,hidden_dim).to(device)
        self.target_critic = Critic(task_dim*task_num+agent_dim*agent_num,hidden_dim,task_num*agent_num).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.loss = nn.MSELoss() 
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.mix_optimizer = th.optim.Adam(self.mix.parameters(),lr=critic_lr)
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.critic_loss_list = []
        self.actor_loss_list = []
        self.model_dir = model_dir
        self.device = device
    

    def select(self, tasks, agent_type, avail_agent, avail_task,evaluate=False):
        avail_agent = th.tensor(avail_agent)
        tasks = th.tensor(tasks,dtype=th.float,device=self.device)
        agent_type = th.tensor(agent_type,dtype=th.float,device=self.device)
        probs = self.actor(tasks, agent_type, avail_task).reshape(agent_type.shape[0],-1)
        na,nt = probs.shape
        if evaluate:
            action = probs.argmax(axis=-1)
        else:
            action_dist = th.distributions.Categorical(probs)
            action = action_dist.sample()
        action = th.eye(nt)[action.cpu()]   #na,nt
        action = action * avail_agent.unsqueeze(-1) #na,nt
        return action.cpu().detach()
    
    def update(self,transition_dict):
        rewards = th.tensor(transition_dict['rewards'],dtype=th.float).flatten().to(self.device)
        batch_size = len(rewards)
        states = th.tensor(transition_dict['states'],dtype=th.float).reshape(batch_size,-1,self.task_dim).to(self.device)
        actions = th.tensor(transition_dict['actions'],dtype=th.long).to(self.device)  
        agent_types = th.tensor(transition_dict['agent_types'],dtype=th.float).to(self.device)
        avail_agents = th.tensor(transition_dict['avail_agents'],dtype=th.float).reshape(batch_size,-1).to(self.device)
        avail_tasks = th.tensor(transition_dict['avail_tasks'],dtype=th.float).reshape(batch_size,-1).to(self.device)

        td_target = rewards
        critic_q_values = (self.critic(th.cat((states.reshape(batch_size,-1), agent_types.reshape(batch_size,-1)),dim=-1)).reshape(actions.shape)*actions).sum(-1) # batch_size,agent_num,task_num-> batch_size,agent_num
        critic_total_q_values = self.mix(agent_types,critic_q_values,avail_agents).reshape(td_target.shape) # batch_size,1
        critic_loss = self.loss(td_target,critic_total_q_values)
        self.critic_loss_list.append(critic_loss.cpu().detach().item())
        self.critic_optimizer.zero_grad()
        self.mix_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        self.mix_optimizer.step()
        
        probs = self.actor(states,agent_types,avail_tasks)
        log_probs = th.log(probs + 1e-8)
        entropy = -th.sum(probs.sum(-1) * log_probs.sum(-1), dim=1, keepdim=True)  
        q_values = (self.target_critic(th.cat((states.reshape(batch_size,-1), agent_types.reshape(batch_size,-1)),dim=-1)).reshape(actions.shape)*actions)
        v_value = th.sum(probs * q_values, dim=-1)  
        total_v_value = self.mix(agent_types,v_value,avail_agents)
        actor_loss = th.mean(-0.01 * entropy - total_v_value)
        self.actor_loss_list.append(actor_loss.cpu().detach().item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 
        self.soft_update(self.critic, self.target_critic)
        
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


    def save(self):
        actor_path = self.model_dir + "/actor_normal_critic.pt"
        critic_path = self.model_dir + "/critic_normal_critic.pt"
        mix_path = self.model_dir + "/mix_normal_critic.pt"
        th.save(self.actor.state_dict(),actor_path)
        th.save(self.critic.state_dict(),critic_path) 
        th.save(self.mix.state_dict(),mix_path) 
        
    
    def load(self):
        actor_path = self.model_dir + "/actor_normal_critic.pt"
        critic_path = self.model_dir + "/critic_normal_critic.pt"
        mix_path = self.model_dir + "/mix_normal_critic.pt"
        self.actor.load_state_dict(th.load(actor_path))
        self.critic.load_state_dict(th.load(critic_path))
        self.mix.load_state_dict(th.load(mix_path))


if __name__ =="__main__":
    agent_type_init =  np.array([
                               [0.2,4,5,3],
                               [0.3,7,8,4],
                               [0.1,15,10,7],
                               [0.6,4,12,3],
                               [0.4,5,3,1],
                               [0.2,4,6,2],
                               [0.3,2,8,3],
                               [0.5,7,9,3],
                               [0.8,15,16,10],
                               [0.8,7,3,8],
                               [0.6,2,4,3],
                               [0.7,3,5,4],
                               [0.9,10,8,9],
                               [0.6,3,5,7],
                               [0.5,10,6,12],
                               [0.1,3.,1,1],
                               [0.2,3,4,2],
                               [0.1,1,4,3],
                               [0.4,8,5,4],
                               [0.3,6,4,1],
                               [0.2,3,6,2],
                               [0.3,5,6,3],
                               [0.1,2,1,4],
                               [0.6,6,10,1],
                               [0.4,5,7,2],
                               [0.2,3,3,3],
                               [0.3,4,5,4],
                               [0.5,6,9,1],
                               [0.8,12,10,2],
                               [0.8,16,6,3],
                               [0.1,9.,2,1],
                               [0.2,2,8,3],
                               [0.1,1,1,1],
                               [0.4,7,8,5],
                               [0.3,16,4,10],
                               [0.2,3,4,12],
                               [0.3,7,2,1],
                               [0.1,3,6,4],
                               [0.6,1,8,9],
                               [0.4,4,6,12],
                               [0.6,4,2,4],
                               [0.7,9,5,1],
                               [0.9,11,18,1],
                               [0.6,6,6,8],
                               [0.4,16,8,7],
                               [0.1,4.,2,5],
                               [0.2,6,8,4],
                               [0.1,1,3,3],
                               [0.4,3,6,2],
                               [0.3,9,1,4],
                               [0.2,3,6,2],
                               [0.3,7,5,9],
                               [0.1,3,5,8],
                               [0.6,16,3,5],
                               [0.4,5,7,12],
                               [0.6,1,1,4],
                               [0.7,7,9,12],
                               [0.9,6,2,5],
                               [0.6,7,8,6],
                               [0.5,11,14,14],
                               ])
    total_state =  np.array([[10.,20,30,50],[10.,30,20,30],[10.,30,20,30],[10.,30,20,30]])
    agent_type = [[] for _ in range(len(total_state))]
    agent_list = []
    a = PreAssign(4,4,64,1e-3,1e-3,0.01,0.005,0.98,"cuda:5",None,10000)
    allo = a.select(th.tensor(total_state,dtype=th.float,device="cuda:5"),th.tensor(agent_type_init,dtype=th.float,device="cuda:5"),th.tensor([[1,1,1,1]],device="cuda:5"))
    print(allo)