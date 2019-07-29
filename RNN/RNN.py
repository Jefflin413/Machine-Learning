import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Hyper parameters
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

# Show data
"""
# float32 for converting to torch FloatTensor
steps = np.linspace(0, np.pi*2, 100, dtype = np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label = 'target (cos)')
plt.plot(steps, x_np, 'b-', label = 'input (sin)')
plt.legend(loc='best')
plt.show()
"""

# Define RNN structure
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size = INPUT_SIZE,
            hidden_size = 32, # Number of hidden neuron for one layer
            num_layers = 1, # Number of hidden layer
            batch_first = True,
            )
        
        self.out = nn.Linear(32,1)
        
    # Because in this demo, the prediction is continuous, so the hidden_​​state of current training can be the input hidden_​​state of the next training

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        # save all prediction, calculate output for each time step
        """
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim = 1), h_state
        """
        # or can be done by simpily
        outs = self.out(r_out)
        return outs, h_state
        
        # since linear can accept any number of additional dimensions, if and only if the last dimension is correct

rnn = RNN()


optimizer = torch.optim.Adam(rnn.parameters(), lr = LR)
loss_func = nn.MSELoss()

# Initialize hidden state
h_state = None

f = plt.figure(1, figsize = (12,5))
plt.ion()
gif = []
for step in range(150):
    start, end = step * np.pi, (step+1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype = np.float32, endpoint = False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)
    # repack the hidden state, break the connection from last iteration
    h_state = h_state.data

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plot
    plt.plot(steps, y_np, 'r-')
    # x and y must have same first dimension, steps has shape (10,) and prediction.data.numpy() has shape (1, 10, 1)
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.05)
    f.canvas.draw()
    image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
    gif.append(image)

imageio.mimsave('./RNN.gif', gif, fps=10)
plt.ioff()
plt.show()
    
    
        
