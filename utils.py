import numpy as np

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

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

def take_action_material(allocation,info):
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
        action[alloc[np.intersect1d(np.where(error[:,0]<0)[0], np.where(error[:,1]<0)[0])]]=5
        action[alloc[np.intersect1d(np.where(error[:,0]<0)[0], np.where(error[:,1]>0)[0])]]=6
        action[alloc[np.intersect1d(np.where(error[:,0]>0)[0], np.where(error[:,1]<0)[0])]]=7
        action[alloc[np.intersect1d(np.where(error[:,0]>0)[0], np.where(error[:,1]>0)[0])]]=8
        action[alloc[(error==0).sum(axis=1)==2]] = 9
    return action