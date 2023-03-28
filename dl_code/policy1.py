import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import math, random 
import gym 
import numpy as np 

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import copy
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = torch.set_grad_enabled(True)
def Transfer_cost(v,action):
  return(torch.tensor(np.sum(v)-v[action]))

class State:
    def __init__(self, s):
        self.s=s #set state (w0_0, w0_1 ... w7_0, w7_0)



    
    def reset(self, s0):
        self.s=s0 #reset back to initial state

        
    def getState(self):
        res=[]
        for i in range(len(s)):
          res.append(self.s[i])

        return res

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        #self.bn1=nn.BatchNorm2d(128)
        self.layer2 = nn.Linear(128, 64)
        #self.bn2=nn.BatchNorm2d(64)
        self.layer3 = nn.Linear(64, 32)
        #self.bn3=nn.BatchNorm2d(32)
        self.layer4 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x=self.layer4(x)
        return x



class Agent:
    def __init__(self, state_size, action_size, model, device, is_eval=False, model_name=''):
        self.state_size = state_size 
        self.action_size = action_size 
        self.memory = deque(maxlen=5000) #increase memory
        self.device=device
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95 #gamma is the discount factor. It quantifies how much importance we give for future rewards.
        self.epsilon = 1.0 #Exploration and Exploitation — Epsilon (e)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        #self.model = load_model("models/" + model_name) if is_eval else self._model()
        self.model = model.double() if is_eval else DQN(self.state_size, self.action_size).double() #if in test mode, self.model = model, else: self.model=DQN()
        self.model=nn.DataParallel(self.model).to(self.device)
        
        self.target_model=copy.deepcopy(self.model)
        self.target_model=nn.DataParallel(self.target_model).to(self.device)
        
        self.batch_loss=0
        self.update=0
      
    def act(self, state):
        state=torch.tensor(state).to(self.device)
        if not self.is_eval and random.random() <= self.epsilon:
            #print("random action")
            return random.randrange(self.action_size)
        #print("Calculating using model")
        
        return torch.argmax(self.model(state))

    def update_target_network(self):
      
 
      self.target_model=copy.deepcopy(self.model).to(self.device)

      #self.target_model.load_state_dict(self.model.state_dict()) 
      print('updated the target model')

    def expReplay(self, batch_size, update_target_net_frequency, loss_function, optimizer):
        mini_batch = []
        l = len(self.memory)
        #batch_predicted=torch.empty([batch_size], requires_grad=True)
        #batch_targets=torch.empty([batch_size], requires_grad=True)
        batch_targets=[]
        batch_predicted=[]
        mini_batch = random.sample(self.memory, batch_size)
        i=0
        for state, action, reward, next_state, done in mini_batch:
            state, action, reward, next_state, done = torch.tensor(state).to(self.device), torch.tensor(action).to(self.device), torch.tensor(reward).to(self.device), torch.tensor(next_state).to(self.device), torch.tensor(done).to(self.device)
            target = reward
            
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)) #have a target network (is a deep copy of our main network, deep copy it every update frequency)
            
            target_f = self.model(state)
            target_f[action] = target
            batch_targets.append(target_f)
            
            batch_predicted.append(self.model(state))
            
            i+=1
            
        
        batch_targets=torch.stack(batch_targets)
        batch_predicted=torch.stack(batch_predicted)
        batch_targets.requires_grad_()
        batch_predicted.requires_grad_()
        loss=loss_function(batch_predicted, batch_targets)

        self.batch_loss=loss
        optimizer.zero_grad()
        loss.backward() 
        #torch.nn.utils.clip_grad_norm(self.model.parameters(), 10)
        optimizer.step()
        self.update+=1
        if self.update % update_target_net_frequency==0:
          self.update_target_network()
        

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 
        
model0 = DQN(16,8).double()
agent = Agent(16,8,model0,device)
batch_losses=[]
mse=torch.nn.MSELoss()
optimizer=torch.optim.Adam(agent.model.parameters(), lr=0.0001) #Adam
transfer_costs=[]
alpha_std=[]
rewards=[]
alpha=100
batch_size=64
iteration=0
parent_load_index=[i for i in range(16) if i%2==1]
queue_load_index=[i for i in range(16) if i%2==0]
s0= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
def take_action():
  
  
  
  if iteration==0:
    s=s0
    action=2
  else:
    s=agent.memory[-1][2]
    action=agent.act(s)
  iteration+=1

  return action, torch.tensor(s)

def record_transition_data_experience_replay(action,s, next_state,done):
  reward=-(alpha*torch.std(next_state[queue_load_index]) + Transfer_cost(s[parent_load_index], action))
  #the reward is the std of the queue loads of the next state and the transfer cost
  rewards.append(reward)
  alpha_std.append(alpha*torch.std(next_state[queue_load_index]))
  transfer_costs.append(Transfer_cost(s[parent_load_index], action))          
  agent.memory.append((s, action, reward, next_state, done)) #storing transition data to use them for training with experience replay
  if len(agent.memory) > batch_size: #if we have enough data points in the agent's memory (more than the batch size), we train using experience replay
    agent.expReplay(batch_size, 500, mse, optimizer)
    batch_losses.append(agent.batch_loss)
    print('exp replay')


file_path = "../build/action_state.txt"

def read_line(line):

  line_list = line.split(" ")

  str_to_float = list(map(float, line_list))
  return torch.tensor(str_to_float)
  


def read_state(s,action):
  while 1:
    
    with open(file_path, "r") as f:
      lines = f.readlines()
      #print(lines)
      if len(lines) > 0 and lines[0] == "DONE":
        done=True
        next_state=s
        record_transition_data_experience_replay(action,s,next_state,done)
        return 1
      elif len(lines) > 0 and lines[-1] == "STATE_READY\n":
        #print (lines)
        done=False
        next_state=read_line(lines[0])                
        record_transition_data_experience_replay(action,s,next_state,done)
        return 0
      else:
        continue

def write_action(action):
  with open(file_path, "w") as f:
    f.write("{}\n".format(action))
    f.write("1\n")
    f.write("ACTION_READY\n")

def main():
  
  while 1:
    action, s = take_action()
    rtn = read_state(s,action)
    if rtn == 1:
      return
    write_action(action)






if __name__ == "__main__":
  main()