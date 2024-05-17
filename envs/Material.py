import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

class Material:
    def __init__(self):
        rng = np.random.RandomState(0)
        col1 = rng.randint(1, 15, size=100)
        sum_col1 = col1 * 5
        col2 = rng.randint(1, sum_col1 - 3, size=100)
        col3 = rng.randint(1, sum_col1 - col2 - 2, size=100)
        col4 = sum_col1 - col2 - col3
        self.agent_type = np.column_stack((col1, col2, col3, col4))
        self.agent_num,self.agent_dim = self.agent_type.shape
        new_cols = rng.randint(0, 100, size=(self.agent_num, 2))
        self.agent_type = np.concatenate((self.agent_type, new_cols), axis=1) #random initial position
        self.agent_num,self.agent_dim = self.agent_type.shape
        self.init_agent_type = self.agent_type.copy()
        
        self.task_num = 5  # maximum task number
        self.task_dim = self.agent_dim
        self.mapsize = 100
        
    def reset(self):
        self.epoch = 0
        self.time = 0 
        self.fresh = np.zeros(self.task_num)-1
        self.task_list = [[30.,20,30,30,10,20],[40.,30,20,20,50,60],[50.,50,40,20,50,30],[40.,20,50,40,10,60],[50,30,50,50,60,20],
                            [30.,40,50,20,40,60],[25.,20,20,10,30,40],[80.,100,120,30,20,50],[30.,20,50,20,10,40],[80,60,80,40,20,30],
                            [75.,80,60,20,30,40],[45.,40,50,35,50,30],[100.,150,140,50,60,30],[30.,30,20,30,60,0],[40,40,60,40,50,50],
                            [65.,20,80,50,50,10],[45,30,40,40,10,20],[80.,100,70,40,0,40],[20.,30,30,10,3,5],[60,30,70,30,40,20],
                            [150.,120,150,80,40,60],[65.,70,50,40,0,10],[55.,20,50,50,60,30],[30.,10,30,20,60,50],[20,10,10,10,40,40],
                            [105.,80,130,60,30,20],[50.,60,10,30,60,40],[55.,50,40,40,10,10],[40.,10,60,30,30,30],[80,80,80,40,10,50]]
        self.agent_type = self.init_agent_type.copy()
        self.task = []
        for _ in range(self.task_num):
            self.task.append(self.task_list.pop(0))
        self.task =  np.array(self.task)
        self.avail_agents = np.ones(len(self.agent_type))
        self.total_state = np.r_[self.task.flatten(),self.agent_type.flatten(),self.avail_agents.flatten()]
        self.pos = self.task[:,-2:]
        self.agent_pos = self.agent_type[:,-2:]
        return self.total_state.copy(), {"task_pos":self.pos.copy(),"agent_pos":self.agent_type[:,-2:].copy()}

    def step(self,action):
        self.epoch += 1
        self.agent_pos = self.agent_type[:,-2:]
        reward= 0
        total_reward = 0
        self.agent_pos[np.where(action==1),0] = np.where(self.agent_pos[np.where(action==1),0]>0,self.agent_pos[np.where(action==1),0]-1,self.agent_pos[np.where(action==1),0])
        self.agent_pos[np.where(action==2),0] = np.where(self.agent_pos[np.where(action==2),0]<self.mapsize-1,self.agent_pos[np.where(action==2),0]+1,self.agent_pos[np.where(action==2),0])   
        self.agent_pos[np.where(action==3),1] = np.where(self.agent_pos[np.where(action==3),1]>0,self.agent_pos[np.where(action==3),1]-1,self.agent_pos[np.where(action==3),1])  
        self.agent_pos[np.where(action==4),1] = np.where(self.agent_pos[np.where(action==4),1]<self.mapsize-1,self.agent_pos[np.where(action==4),1]+1,self.agent_pos[np.where(action==4),1])    
        self.agent_pos[np.where(action==5),0] = np.where(self.agent_pos[np.where(action==5),0]>0,self.agent_pos[np.where(action==5),0]-1,self.agent_pos[np.where(action==5),0])
        self.agent_pos[np.where(action==5),1] = np.where(self.agent_pos[np.where(action==5),1]>0,self.agent_pos[np.where(action==5),1]-1,self.agent_pos[np.where(action==5),1])  
        self.agent_pos[np.where(action==6),0] = np.where(self.agent_pos[np.where(action==6),0]>0,self.agent_pos[np.where(action==6),0]-1,self.agent_pos[np.where(action==6),0])
        self.agent_pos[np.where(action==6),1] = np.where(self.agent_pos[np.where(action==6),1]<self.mapsize-1,self.agent_pos[np.where(action==6),1]+1,self.agent_pos[np.where(action==6),1])    
        self.agent_pos[np.where(action==7),0] = np.where(self.agent_pos[np.where(action==7),0]<self.mapsize-1,self.agent_pos[np.where(action==7),0]+1,self.agent_pos[np.where(action==7),0])   
        self.agent_pos[np.where(action==7),1] = np.where(self.agent_pos[np.where(action==7),1]>0,self.agent_pos[np.where(action==7),1]-1,self.agent_pos[np.where(action==7),1])  
        self.agent_pos[np.where(action==8),0] = np.where(self.agent_pos[np.where(action==8),0]<self.mapsize-1,self.agent_pos[np.where(action==8),0]+1,self.agent_pos[np.where(action==8),0])   
        self.agent_pos[np.where(action==8),1] = np.where(self.agent_pos[np.where(action==8),1]<self.mapsize-1,self.agent_pos[np.where(action==8),1]+1,self.agent_pos[np.where(action==8),1])    
        
        self.total_type = np.zeros([self.task_num,self.task_dim])
        choose_agents = [[] for _ in range(self.task_num)]
        for each in np.where(action==9)[0]:
            current_task = (((self.agent_pos[each] == self.pos).sum(axis=1))==2).argmax()
            self.total_type[current_task] += self.agent_type[each]
            choose_agents[current_task].append(each)
        for task_id in range(self.task_num):
            if (self.task[task_id]).sum()==0:
                continue
            if np.all(self.total_type[task_id][1:-2] >= self.task[task_id][1:-2]):
                for each in choose_agents[task_id]:
                    self.avail_agents[each] = 0
                reward += self.task[task_id][0] - self.total_type[task_id][0]
                total_reward += self.task[task_id][0]
                self.task[task_id] *= 0
                self.fresh[task_id] = 1
        self.task[np.where(self.fresh<=-15)][:-2] *= 0.95
        self.avail_agents = (action==0).astype(np.int8)
        done = 1 if ((len(self.task_list)==0) and ((self.task).sum()==0)) or self.epoch>=150 else 0
 
        self.fresh -= 1
        empty_task_idx = np.where(self.task.sum(axis=1)==0)[0]
        if self.epoch%5==0:
            for idx in empty_task_idx:
                if len(self.task_list)>0:
                    self.task[idx] = self.task_list.pop(0) 

        self.total_state = np.r_[self.task.flatten(),self.agent_type.flatten(),self.avail_agents.astype(np.int8).flatten()]
        return self.total_state.copy(), reward, done, {"task_pos":self.task[:,-2:].copy(),"agent_pos":self.agent_type[:,-2:].copy(),"total_reward":total_reward}
    
    def render(self):
        plt.figure(figsize=(5,5))
        plt.gca().set_facecolor('white')
        for task_id,each_pos in enumerate(self.pos):
            if np.all(self.task[task_id]>0):
                task_img = mpimg.imread('./icon/warehouse.jpeg')
                task_size = min(sum(self.task[task_id][1:])/50,3)
                task_left = each_pos[1] + 0.5 - task_size 
                task_right = each_pos[1] + 0.5 + task_size
                task_bottom = each_pos[0] + 0.5 - task_size
                task_top = each_pos[0] + 0.5 + task_size
                plt.gca().add_artist(plt.imshow(task_img, extent=[task_left, task_right, task_bottom, task_top]))

        for each_pos in self.agent_pos:
            number = sum(np.all(np.equal(self.agent_pos, each_pos),axis=1)) 
            agent_img = mpimg.imread('./icon/car.jpeg')
            plt.gca().add_artist(plt.imshow(agent_img, extent=[each_pos[1]-2, each_pos[1]+2, each_pos[0]-2, each_pos[0]+2]))
            plt.text(each_pos[1]+0.8, each_pos[0]+0.2, str(number), color='gray', fontsize=5, ha='center', va='center')
        plt.xlim(0, self.mapsize)
        plt.ylim(0, self.mapsize)
        plt.savefig("./img/img_%04d.jpg" % self.time)
        self.time+=1
        plt.show()


        


