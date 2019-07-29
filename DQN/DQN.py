import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# For saving the results as gif
import matplotlib
import matplotlib.pyplot as plt
import imageio

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   
# Greedy policy, sometimes the computer won't choose action which has the highest value in Q table, instead it chooses randomly
EPSILON = 0.9
# Reward discount
GAMMA = 0.9
# Target neural network update frequency
TARGET_REPLACE_ITER = 100   
MEMORY_CAPACITY = 2000
# Physical model
env = gym.make('CartPole-v0') 
# env = env.unwrapped
observation = env.reset()
frames = []

# Number of actions
N_ACTIONS = env.action_space.n
# Dimension of state
N_STATES = env.observation_space.shape[0] 
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
# Verify the data type of what is inside action_space
# In this case, there are two types of actions, and their values are 0 and 1 in the type of int

# Define network structure
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50) # Input state
        self.fc1.weight.data.normal_(0, 0.1)   # Sample from normal distribution to initailize the neural value
        self.out = nn.Linear(50, N_ACTIONS) # Output action value (evaluate the value for each action)
        self.out.weight.data.normal_(0, 0.1)   # Initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# Use class to integrate multiple functions
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()   # Declare two similar networks

        self.learn_step_counter = 0     # For target updating
        self.memory_counter = 0     # For storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # Initialize memory, (2000, current state + action + reward + next step),
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # Target net is not updated here, since at here we will update every iteration
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # Model expects at least 2D tensor as input, the first dimension is mini batch 
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # Input only one sample
        if np.random.uniform() < EPSILON:   # Greedy
            actions_value = self.eval_net.forward(x)    # From this state find next step action value
            action = torch.max(actions_value, 1)[1].data.numpy()    # Returns action (index) which has the maximum probability (value)
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)    # from 0 to number of action, randomly choose a value
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) # Package together
        # Replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY  # Over 2000 we will rewrite from the beginning 
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) # Copy parameters
        self.learn_step_counter += 1

        # Sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)    # = sample BATCH_SIZE (number, ex. 1 or 3) values from range(MEMORY_CAPACITY) 
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        # b_a is a tensor with size [32, 1] and represents the chosen action index
        # Tensor.gather(dim, index), the index size must be the same as the dim dimention element of Tensor
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1), pick the chosen action's evaluation value
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate, it will eliminate the property: grad_fn=<AddmmBackward>, return pure tensor
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # reshape, since shape (q_next.max(1)[0]) = [32], shape (b_r) = [32,1]
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # It'll make eval_net be updated every iteration while target_net be updated every 100 iteration

dqn = DQN()

print('\nStart simulation...')
for i_episode in range(60):
    s = env.reset() # Returns an initial observation
    ep_r = 0
    # For every experiment
    while True:
        # Refresh frame
        # env.render() can replace the code below
        frames.append(env.render(mode = 'rgb_array'))
        a = dqn.choose_action(s)

        # Take action, done means this time, the experiment is over
        s_, r, done, info = env.step(a)

        # Modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        
        # If dqn.memory_counter > MEMORY_CAPACITY:
        dqn.learn()
        if done:
            print('Ep: ', i_episode,
                  '| Ep_r: ', round(ep_r, 2))               

        if done:
            break
        s = s_

print('Finish!!')
env.close()

# Code below is written for saving result, can be notated
"""
print("frames: ", np.shape(frames))
gif = []
def display_frames_as_gif(frames):
    fig, ax = plt.subplots(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    for i in range(len(frames)):
        print("i: ", i, "/", len(frames))
        ax.imshow(frames[i])
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif.append(image)

display_frames_as_gif(frames)
kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave('./powers.gif', gif, fps=40)
"""

