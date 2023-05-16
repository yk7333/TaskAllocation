import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

class RBF:
    def __init__(self):
        # self.agent_type = np.array([
        #                     [0.2,4,5,3],
        #                     [0.3,7,8,4],
        #                     [0.6,15,10,7],
        #                     [0.2,4,12,3],
        #                     [0.4,5,3,1],
        #                     [0.2,4,6,2],
        #                     [0.3,2,8,3],
        #                     [0.3,7,9,3],
        #                     [0.8,15,16,10],
        #                     [0.5,7,3,8],
        #                     [0.6,2,4,3],
        #                     [0.3,3,5,4],
        #                     [0.5,10,8,9],
        #                     [0.6,3,5,7],
        #                     [0.5,10,6,12],
        #                     [0.1,3.,1,1],
        #                     [0.2,3,4,2],
        #                     [0.1,1,4,3],
        #                     [0.4,8,5,4],
        #                     [0.3,6,4,1],
        #                     [0.2,3,6,2],
        #                     [0.3,5,6,3],
        #                     [0.1,2,1,4],
        #                     [0.6,6,10,1],
        #                     [0.4,5,7,2],
        #                     [0.2,3,3,3],
        #                     [0.3,4,5,4],
        #                     [0.5,6,9,1],
        #                     [0.8,12,10,2],
        #                     [0.8,16,6,3],
        #                     [0.1,9.,2,1],
        #                     [0.2,2,8,3],
        #                     [0.1,1,1,1],
        #                     [0.4,7,8,5],
        #                     [0.5,16,4,10],
        #                     [0.2,3,4,12],
        #                     [0.3,7,2,1],
        #                     [0.1,3,6,4],
        #                     [0.6,1,8,9],
        #                     [0.4,4,6,12],
        #                     [0.6,4,2,4],
        #                     [0.7,9,5,1],
        #                     [0.8,11,18,10],
        #                     [0.4,6,6,8],
        #                     [0.4,16,8,7],
        #                     [0.2,4.,2,5],
        #                     [0.3,6,8,4],
        #                     [0.1,1,3,3],
        #                     [0.2,3,6,2],
        #                     [0.3,9,1,4],
        #                     [0.2,3,6,2],
        #                     [0.3,7,5,9],
        #                     [0.2,3,5,8],
        #                     [0.6,16,3,5],
        #                     [0.4,5,7,12],
        #                     [0.1,1,1,4],
        #                     [0.7,7,9,12],
        #                     [0.4,6,2,5],
        #                     [0.6,7,8,6],
        #                     [0.8,11,14,14],
        #                     [0.2,5,3,4],
        #                     [0.3,4,7,8],
        #                     [0.6,7,15,10],
        #                     [0.2,4,3,12],
        #                     [0.4,1,5,3],
        #                     [0.2,2,6,4],
        #                     [0.3,8,3,2],
        #                     [0.3,9,3,7],
        #                     [0.8,10,15,16],
        #                     [0.5,8,7,3],
        #                     [0.6,12,4,3],
        #                     [0.3,1,6,6],
        #                     [0.5,7,8,3],
        #                     [0.4,5,3,9],
        #                     [0.5,6,12,7],
        #                     [0.1,3.,3,1],
        #                     [0.2,6,1,2],
        #                     [0.1,3,1,3],
        #                     [0.4,4,6,8],
        #                     [0.3,2,6,5],
        #                     [0.2,2,4,3],
        #                     [0.2,4,6,3],
        #                     [0.1,2,1,3],
        #                     [0.6,12,3,10],
        #                     [0.4,4,8,6],
        #                     [0.2,3,3,3],
        #                     [0.3,2,4,7],
        #                     [0.5,3,10,6],
        #                     [0.8,14,10,6],
        #                     [0.8,3,16,8],
        #                     [0.3,1.,9,2],
        #                     [0.3,8,3,2],
        #                     [0.1,1,1,4],
        #                     [0.4,5,6,8],
        #                     [0.7,4,16,10],
        #                     [0.2,3,8,6],
        #                     [0.3,1,2,7],
        #                     [0.2,3,6,4],
        #                     [0.6,9,2,10],
        #                     [0.4,4,12,2],
        #                     ])
        self.agent_type = np.array([
                           [0.2,5,3,4],
                           [0.3,4,7,8],
                           [0.6,7,15,10],
                           [0.2,4,3,12],
                           [0.4,1,5,3],
                           [0.2,2,6,4],
                           [0.3,8,3,2],
                           [0.3,9,3,7],
                           [0.8,10,15,16],
                           [0.5,8,7,3],
                           [0.6,12,4,3],
                           [0.3,1,6,6],
                           [0.5,7,8,3],
                           [0.4,5,3,9],
                           [0.5,6,12,7],
                           [0.1,3.,3,1],
                           [0.2,6,1,2],
                           [0.1,3,1,3],
                           [0.4,4,6,8],
                           [0.3,2,6,5],
                           [0.2,2,4,3],
                           [0.2,4,6,3],
                           [0.1,2,1,3],
                           [0.6,12,3,10],
                           [0.4,4,8,6],
                           [0.2,3,3,3],
                           [0.3,2,4,7],
                           [0.5,3,10,6],
                           [0.8,14,10,6],
                           [0.8,3,16,8],
                           [0.3,1.,9,2],
                           [0.3,8,3,2],
                           [0.1,1,1,4],
                           [0.4,5,6,8],
                           [0.7,4,16,10],
                           [0.2,3,8,6],
                           [0.3,1,2,7],
                           [0.2,3,6,4],
                           [0.6,9,2,10],
                           [0.4,4,12,2],
                           [0.6,10,2,10],
                           [0.6,6,8,12],
                           [0.8,12,18,10],
                           [0.4,8,8,4],
                           [0.6,16,4,9],
                           [0.2,2.,5,2],
                           [0.3,8,8,2],
                           [0.1,1,3,3],
                           [0.2,6,3,2],
                           [0.3,1,9,6],
                           [0.2,2,3,6],
                           [0.3,6,4,9],
                           [0.2,3,5,4],
                           [0.6,16,6,4],
                           [0.5,5,7,12],
                           [0.1,2,4,1],
                           [0.7,9,10,12],
                           [0.4,2,6,8],
                           [0.5,4,10,6],
                           [0.8,14,10,12],
                           ])
        self.agent_num,self.agent_dim = self.agent_type.shape
        self.task_num = 5  
        self.task_dim = 4
        self.mapsize = 7
        self.agent_pos = np.ones([self.agent_num,2]) * (self.mapsize//2)
        
    def reset(self):
        self.agent_pos = np.ones([self.agent_num,2]) * (self.mapsize//2)
        self.epoch = 0
        self.time = 0 
        self.fresh = np.zeros(self.task_num)-1
        self.task_list = [[3.,20,30,30],[3.,30,20,20],[5.,50,40,20],[4.,20,50,40],[5,30,50,50],
                            [3.,40,50,20],[2.,20,20,10],[8.,100,120,30],[3.,20,50,20],[7,60,80,40],
                            [7.,80,60,20],[4.,40,50,35],[10.,150,140,50],[3.,30,20,30],[4,40,60,40],
                            [6.,20,80,50],[4,30,40,40],[8.,100,70,40],[2.,30,30,10],[6,30,70,30],
                            [15.,120,150,80],[6.,70,50,40],[5.,20,50,50],[3.,10,30,20],[2,10,10,10],
                            [10.,80,130,60],[5.,60,10,30],[5.,50,40,40],[4.,10,60,30],[8,80,80,40]]
        # self.task_list = [[4.,30,30,40],[2.,15,20,15],[6.,40,50,20],[7.,30,50,60],[3,30,20,30],
        #                     [5.,60,50,40],[3.,20,20,30],[10.,120,100,30],[3.,50,20,15],[9,80,40,80],
        #                     [3.,40,20,20],[6.,60,30,50],[8.,60,100,60],[4.,30,40,40],[5,40,60,40],
        #                     [6.,50,80,20],[4,40,20,50],[5.,50,70,60],[3.,30,20,40],[8,70,40,70],
        #                     [10.,100,80,50],[5.,40,60,80],[3.,20,30,50],[2.,10,30,20],[2,15,20,15],
        #                     [10.,40,60,120],[3.,40,30,10],[4.,40,40,40],[4.,30,50,30],[6,40,60,40]]
        self.task_pos = [[1,2],[5,6],[5,3],[0,6],[6,2],
                         [4,6],[3,4],[2,5],[1,4],[2,3],
                         [3,4],[5,3],[6,3],[6,0],[5,5],
                         [5,1],[1,2],[0,4],[3,5],[4,2],
                         [4,6],[0,1],[6,3],[6,5],[4,4],
                         [3,2],[6,4],[1,1],[3,3],[1,5]]
        self.task = []
        self.pos = []
        for _ in range(self.task_num):
            self.task.append(self.task_list.pop(0))
            self.pos.append(self.task_pos.pop(0))
        self.task =  np.array(self.task)
        self.pos = np.array(self.pos)
        self.avail_agents = np.ones(len(self.agent_type))
        self.total_state = np.r_[self.task.flatten(),self.agent_type.flatten(),self.avail_agents.flatten()]
        return self.total_state.copy(), {"task_pos":self.pos,"agent_pos":self.agent_pos}

    def step(self,action):
        self.epoch += 1
        reward= 0
        total_reward = 0
        self.agent_pos[np.where(action==1),0] = np.where(self.agent_pos[np.where(action==1),0]>0,self.agent_pos[np.where(action==1),0]-1,self.agent_pos[np.where(action==1),0])
        self.agent_pos[np.where(action==2),0] = np.where(self.agent_pos[np.where(action==2),0]<self.mapsize-1,self.agent_pos[np.where(action==2),0]+1,self.agent_pos[np.where(action==2),0])   
        self.agent_pos[np.where(action==3),1] = np.where(self.agent_pos[np.where(action==3),1]>0,self.agent_pos[np.where(action==3),1]-1,self.agent_pos[np.where(action==3),1])  
        self.agent_pos[np.where(action==4),1] = np.where(self.agent_pos[np.where(action==4),1]<self.mapsize-1,self.agent_pos[np.where(action==4),1]+1,self.agent_pos[np.where(action==4),1])    
        self.total_type = np.zeros([self.task_num,self.task_dim])
        for each in np.where(action==5)[0]:
            self.total_type[((self.agent_pos[each] == self.pos).sum(axis=1))==2] += self.agent_type[each]
        for task_id in range(self.task_num):
            if (self.task[task_id]).sum()==0:
                continue
            if np.all(self.total_type[task_id][1:] >= self.task[task_id][1:]):
                reward += self.task[task_id][0] - self.total_type[task_id][0]
                total_reward += self.task[task_id][0]
                self.task[task_id] *= 0
                self.pos[task_id] *= -1
                self.fresh[task_id] = 1
        self.task[np.where(self.fresh<=-10)] *= 0.99
        self.avail_agents = (action==0).astype(np.int8)
        done = 1 if ((len(self.task_list)==0) and ((self.task).sum()==0)) or self.epoch>=100 else 0
 
        self.fresh -= 1
        empty_task_idx = np.where(self.task.sum(axis=1)==0)[0]
        if self.epoch%5==0:
            for idx in empty_task_idx:
                if len(self.task_list)>0:
                    self.task[idx] = self.task_list.pop(0) 
                    self.pos[idx] = self.task_pos.pop(0) 

        self.total_state = np.r_[self.task.flatten(),self.agent_type.flatten(),self.avail_agents.astype(np.int8).flatten()]
        return self.total_state.copy(), reward, done, {"task_pos":self.pos.copy(),"agent_pos":self.agent_pos.copy(),"total_reward":total_reward}
    
    def render(self):
        plt.figure(figsize=(5,5))
        plt.gca().set_facecolor('black')
        plt.grid(color='white', linewidth=1)
        for task_id,each_pos in enumerate(self.pos):
            if np.all(self.task[task_id]>0):
                task_img = mpimg.imread('./icon/apple.png')
                task_size = min(sum(self.task[task_id][1:])/500,0.45)
                task_left = each_pos[1] + 0.5 - task_size 
                task_right = each_pos[1] + 0.5 + task_size
                task_bottom = each_pos[0] + 0.5 - task_size
                task_top = each_pos[0] + 0.5 + task_size
                plt.gca().add_artist(plt.imshow(task_img, extent=[task_left, task_right, task_bottom, task_top]))

        for each_pos in self.agent_pos:
            number = sum(np.all(np.equal(self.agent_pos, each_pos),axis=1)) 
            agent_img = mpimg.imread('./icon/agent.png')
            plt.gca().add_artist(plt.imshow(agent_img, extent=[each_pos[1]+0.2, each_pos[1]+0.8, each_pos[0]+0.2, each_pos[0]+0.8]))
            plt.text(each_pos[1]+0.8, each_pos[0]+0.2, str(number), color='gray', fontsize=12, ha='center', va='center')
        plt.xlim(0, 7)
        plt.ylim(0, 7)
        plt.savefig("./img/img_%04d.jpg" % self.time)
        self.time+=1
        plt.show()


        
def take_action(allocation,info):
    task_pos = info["task_pos"]
    agent_pos = info["agent_pos"]
    action = np.zeros(len(agent_pos))
    for task_id,alloc in enumerate(allocation):
        if alloc ==[] or np.all(task_pos[task_id]<=0):
            continue
        alloc = np.array(alloc)
        error = task_pos[task_id] - agent_pos[alloc]
        action[alloc[np.where(error[:,0]<0)[0]]] = 1
        action[alloc[np.where(error[:,0]>0)[0]]] = 2
        action[alloc[np.where(error[:,1]<0)[0]]] = 3
        action[alloc[np.where(error[:,1]>0)[0]]] = 4
        action[alloc[(error==0).sum(axis=1)==2]] = 5
    return action

if __name__ == "__main__":
    agent_type= np.array([[0.2,40,50,30],
                    [0.3,70,80,40],
                    [0.6,150,100,70],
                    [0.2,40,120,30],
                    [0.4,50,30,10],
                    [0.2,40,60,20],
                    [0.3,20,80,30],
                    [0.3,70,30,90],
                    [0.8,150,160,100],
                    [0.5,70,30,80],
                    [0.6,20,40,30],
                    [0.3,30,50,40],
                    [0.5,100,80,90],
                    [0.6,30,50,70],
                    [0.8,150,160,100],
                    [0.5,70,30,80],
                    [0.6,20,40,30],
                    [0.3,30,50,40],
                    [0.5,100,80,90],
                    [0.6,30,50,70],
                               ])
    E = RBF(agent_type)
    state,info = E.reset()
    done = 0
    for _ in range(100):
        E.render()
        allocation=[[1,2,3],[4,5,6],[7,8,9],[10,11,12,13],[0,14,15],[16,17,18]]
        action = take_action(allocation,info)
        _,reward,_,info = E.step(action)




