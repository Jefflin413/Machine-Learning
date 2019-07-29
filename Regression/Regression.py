import torch
import matplotlib.pyplot as plt

import imageio
import numpy as np

# Create data, 100 points evenly distributed from -2 to 2
# Use unsqueeze to add one more dimension, torch needs at least 2D input (batch, data)
x = torch.unsqueeze(torch.linspace(-2,2,100), dim=1)
# y = x^2 + b
y = x.pow(2) + 0.5 * torch.rand(x.size()) 

# Plot data 
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# Define NN class
class Net(torch.nn.Module):

    # 官方必備步驟
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() 
        self.hidden = torch.nn.Linear(n_feature, n_hidden) 
        self.predict = torch.nn.Linear(n_hidden, n_output)
        
    def forward(self,x): 
        x = torch.relu(self.hidden(x))
        x = self.predict(x) 
        return x

# Construct a NN
net = Net(1, 10, 1)

# Can use print to see the structure of the network
# print(net)

# Stochastic gradient descent (SGD)
optimizer = torch.optim.SGD(net.parameters(), lr = 0.2)
# Mean squared error
loss_func = torch.nn.MSELoss()

# Continuously plot
plt.ion()   
f = plt.figure()
gif = []
for t in range(200):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        # Plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
        f.canvas.draw()
        image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
        gif.append(image)

imageio.mimsave('./regression.gif', gif, fps=5)       
plt.ioff()
plt.show()
