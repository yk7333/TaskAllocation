import numpy as np

class Electric:
    def __init__(self):
        # self.task_property = np.array([[3.,30,10.,50],[3.,20,14,15],[5.,50,16,78],[4.,40,22,36],[5,50,28,51],
        #                     [3.,40,32,26],[2.,20,33,40],[8.,100,34,92],[3.,20,35,73],[7,80,39,50],
        #                     [7.,60,41,62],[4.,40,53,20],[10.,140,55,44],[3.,25,56,51],[4,40,61,79],
        #                     [6.,55,73,12],[4,35,76,28],[8.,85,82,65],[2.,25,90,20],[6,70,94,30]])
        self.task_property = np.array([[5.,40,20.,80],[4.,30,16,18],[9.,50,39,98],[6.,22,56,18],[4,40,18,31],
                            [5.,26,46,53],[2.,40,33,20],[5.,50,62,43],[6.,50,64,43],[9,80,39,50],
                            [8.,60,51,92],[4.,26,43,40],[10.,80,60,65],[3.,12,24,41],[2,20,21,39],
                            [4.5,35,53,42],[3.5,26,46,52],[6.,45,52,65],[2.,35,30,18],[5,50,74,50]])
        
        self.agent_type = self.task_property.copy()
        self.task_num = 5
        self.task_dim = 4
        self.agent_num = 20
        self.agent_dim = 4

    def reset(self):
        self.epoch = 0
        self.time = 0 
        self.fresh = np.zeros(self.task_num)-1
        
        self.task_property = np.array([[5.,40,20.,80],[4.,30,16,18],[9.,50,39,98],[6.,22,56,18],[4,40,18,31],
                            [5.,26,46,53],[2.,40,33,20],[5.,50,62,43],[6.,50,64,43],[9,80,39,50],
                            [8.,60,51,92],[4.,26,43,40],[10.,80,60,65],[3.,12,24,41],[2,20,21,39],
                            [4.5,35,53,42],[3.5,26,46,52],[6.,45,52,65],[2.,35,30,18],[5,50,74,50]])
        self.agent_type = self.task_property.copy()
        self.agent_type[:,0] = 0   # 在后面会根据距离计算cost
        self.task_list = [0,1,2,3,4,0,5,1,6,7,8,9,10,11,12,4,13,14,15,16,17,18,6,19,0,5,12,7,9,12]
        self.convey_cost = np.array([0.04,0.03,0.04,0.05,0.05,0.02,0.03,0.03,0.03,0.04,0.04,0.03,0.02,0.04,0.04,0.03,0.02,0.03,0.03,0.02])
        self.task = []
        self.avail_agents = np.ones(len(self.agent_type))
        for _ in range(self.task_num):
            task_id = self.task_list.pop(0)
            self.avail_agents[task_id] = 0
            self.task.append(self.task_property[task_id]*2)

        self.task =  np.array(self.task)
        
        self.total_state = np.r_[self.task.flatten(),self.agent_type.flatten(),self.avail_agents.flatten(),self.convey_cost]
        return self.total_state.copy(), {}

    def compute_cost(self,agent_id,task_id):
        convey_cost = (self.convey_cost[agent_id] + self.convey_cost[task_id])/2
        distance = np.linalg.norm(self.agent_type[agent_id,2:] - self.task[task_id,2:])
        return convey_cost * distance

    def step(self,action):
        self.epoch += 1
        reward= 0
        total_reward = 0
        self.total_type = np.zeros([self.task_num,self.task_dim])
        for task_id,each_alloc in enumerate(action):
            if len(each_alloc) == 0:
                continue
            for each in each_alloc: 
                self.agent_type[each,0] = self.compute_cost(each,task_id)
                self.total_type[task_id] += self.agent_type[each]
        for task_id in range(self.task_num):
            if (self.task[task_id]).sum()==0:
                continue
            if np.all(self.total_type[task_id][1] >= self.task[task_id][1]):
                reward += self.task[task_id][0] - self.total_type[task_id][0]
                total_reward += self.task[task_id][0]
                agent_id = (self.task[task_id,2:] == self.agent_type[:,2:]).sum(axis=1).argmax()
                self.task[task_id] *= 0
                self.fresh[task_id] = 1
                
                self.avail_agents[agent_id] = 1
        self.task[np.where(self.fresh<=-10)] *= 0.9
        done = 1 if ((len(self.task_list)==0) and ((self.task).sum()==0)) or self.epoch>=100 else 0
 
        self.fresh -= 1
        empty_task_idx = np.where(self.task.sum(axis=1)==0)[0]
        if self.epoch%5==0:
            for idx in empty_task_idx:
                if idx==2:
                    break
                if len(self.task_list)>0:
                    task_id = self.task_list.pop(0)
                    self.avail_agents[task_id] = 0
                    self.task[idx] = self.task_property[task_id]*2

        self.total_state = np.r_[self.task.flatten(),self.agent_type.flatten(),self.avail_agents.astype(np.int8).flatten(),self.convey_cost.flatten()]
        return self.total_state.copy(), reward, done, {"total_reward":total_reward}

if __name__ == "__main__":
    E = Electric()
    state,info = E.reset()
    done = 0
    for _ in range(100):
        allocation=[[1,2,3],[4,5,6],[7,8],[10,11,12],[14,15]]
        _,reward,_,info = E.step(allocation)
        print(reward)




