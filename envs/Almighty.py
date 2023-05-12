import numpy as np

class Almighty:
    def __init__(self):
        self.task_num = 9
        self.task_dim = 9
        self.agent_type = np.array([[1.,1,0,0,0,0,0,0,0],
                           [1,1,0,0,0,0,0,0,0],
                           [1,1,0,0,0,0,0,0,0],
                           [1,0,1,0,0,0,0,0,0],
                           [1,0,1,0,0,0,0,0,0],
                           [1,0,1,0,0,0,0,0,0],
                           [1,0,0,1,0,0,0,0,0],
                           [1,0,0,1,0,0,0,0,0],
                           [1,0,0,1,0,0,0,0,0],
                           [1,0,0,0,1,0,0,0,0],
                           [1,0,0,0,1,0,0,0,0],
                           [1,0,0,0,1,0,0,0,0],
                           [1,0,0,0,0,1,0,0,0],
                           [1,0,0,0,0,1,0,0,0],
                           [1,0,0,0,0,1,0,0,0],
                           [1,0,0,0,0,0,1,0,0],
                           [1,0,0,0,0,0,1,0,0],
                           [1,0,0,0,0,0,1,0,0],
                           [1,0,0,0,0,0,0,1,0],
                           [1,0,0,0,0,0,0,1,0],
                           [1,0,0,0,0,0,0,1,0],
                           [1,0,0,0,0,0,0,0,1],
                           [1,0,0,0,0,0,0,0,1],
                           [1,0,0,0,0,0,0,0,1],
                           [1,1,1,1,1,1,1,1,1],])
        self.task = np.r_[np.insert(np.identity(self.task_num-1),0,3,axis=1),np.insert(np.ones([1,self.task_num-1]),0,20,axis=1)]
        
    def reset(self):
        self.epoch = 0
        self.task = np.r_[np.insert(np.identity(self.task_num-1),0,3,axis=1),np.insert(np.ones([1,self.task_num-1]),0,20,axis=1)]
        self.agent_type = np.array([[1.,1,0,0,0,0,0,0,0],
                           [1,1,0,0,0,0,0,0,0],
                           [1,1,0,0,0,0,0,0,0],
                           [1,0,1,0,0,0,0,0,0],
                           [1,0,1,0,0,0,0,0,0],
                           [1,0,1,0,0,0,0,0,0],
                           [1,0,0,1,0,0,0,0,0],
                           [1,0,0,1,0,0,0,0,0],
                           [1,0,0,1,0,0,0,0,0],
                           [1,0,0,0,1,0,0,0,0],
                           [1,0,0,0,1,0,0,0,0],
                           [1,0,0,0,1,0,0,0,0],
                           [1,0,0,0,0,1,0,0,0],
                           [1,0,0,0,0,1,0,0,0],
                           [1,0,0,0,0,1,0,0,0],
                           [1,0,0,0,0,0,1,0,0],
                           [1,0,0,0,0,0,1,0,0],
                           [1,0,0,0,0,0,1,0,0],
                           [1,0,0,0,0,0,0,1,0],
                           [1,0,0,0,0,0,0,1,0],
                           [1,0,0,0,0,0,0,1,0],
                           [1,0,0,0,0,0,0,0,1],
                           [1,0,0,0,0,0,0,0,1],
                           [1,0,0,0,0,0,0,0,1],
                           [1,1,1,1,1,1,1,1,1],])
        self.avail_agents = np.ones(len(self.agent_type))
        self.total_state = np.r_[self.task.flatten(),self.agent_type.flatten(),self.avail_agents.flatten()]
        return self.total_state.copy()

    def step(self,action):
        self.epoch += 1
        reward= 0
        manager_cost = 0
        for task_id,allocation in enumerate(action):
            if allocation==[]:
                continue
            total = self.agent_type[np.array(allocation)].sum(axis=0)
            if np.all(total[1:] >= self.task[task_id][1:]):
                reward += self.task[task_id][0]
                manager_cost += total[0]
                self.task[task_id] *= 0
                self.avail_agents[allocation] = 0
        self.total_state = np.r_[self.task.flatten(),self.agent_type.flatten(),self.avail_agents.astype(np.int8).flatten()]
        return reward - manager_cost
        
if __name__ == "__main__":
    E = Almighty()
    E.reset()
    done = 0
    print(E.step([[0],[3],[6],[9],[12],[15],[18],[21],[24]]))





