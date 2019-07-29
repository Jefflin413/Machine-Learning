import os

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import imageio

# Hyperparameters
EPOCH = 5
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
# Number of  images shown in figure
N_TEST_IMG = 10

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

# Load dataset
train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
    )

# DataLoader
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# Show specific image in training dataset
"""
print(train_data.data.size())
print(train_data.targets.size())
plt.imshow(train_data.data[2].numpy(), cmap = 'gray')
plt.title('%i' % train_data.targets[2])
plt.show()
"""

# Structure of the Autoencoder
# Constructing a single NN model to represent an Autoencoder is possible, but in this case we separately built encoder and decoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3),
            )
        
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
            )
        
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded

encoder = Encoder()
decoder = Decoder()
        
optimizer_en = torch.optim.Adam(encoder.parameters(), lr = LR)
optimizer_de = torch.optim.Adam(decoder.parameters(), lr = LR)

# Mean square error
loss_func = nn.MSELoss()

# Initialize figure
# The first two paramters represent the number of rows and columns
f, a = plt.subplots(2, N_TEST_IMG, figsize=(10, 2))
# Continuously plot
plt.ion()
# A container that is used to save images in order to produce gif file 
gif = []

# Original data (first row) for viewing
view_data = train_data.data[:N_TEST_IMG]
# Assign handwriting digits 0-9 to location 0-9 repectively
view_data[0] = train_data.data[1]
view_data[1] = train_data.data[3]
view_data[2] = train_data.data[5]
view_data[3] = train_data.data[7]
view_data[4] = train_data.data[9] 
view_data[5] = train_data.data[11] 
view_data[6] = train_data.data[13]
view_data[7] = train_data.data[15]
view_data[8] = train_data.data[17]
view_data[9] = train_data.data[19] 

# Plot N_TEST_IMG testing data on the first row of figure and hide the x and y labels
for i in range(N_TEST_IMG):
    a[0][i].imshow(view_data[i].numpy(), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

# Standard procedure of training NN
for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        # Since it's an unsupervised learning, the label data won't be used
        # b_x is processed through encoder and decoder while b_y remains the same
        # Reshape the dimension of x and y to (batch, 28*28)
        b_x = x.view(-1, 28*28)   
        b_y = x.view(-1, 28*28)   

        encoded = encoder(b_x)
        decoded = decoder(encoded)
        loss = loss_func(decoded, b_y)
        optimizer_en.zero_grad()               
        optimizer_de.zero_grad()
        loss.backward() 
        optimizer_en.step() 
        optimizer_de.step()

        # Visualization process
        # Every 100 steps
        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            # Plot decoded images at the second row of the figure
            # Flatten from shape (batch, 28, 28) to (batch, 28*28) and convert from [0-255] to [0-1]
            # Plot N_TEST_IMG reconstructed data on the second row of figure and hide the x and y labels
            # Actually the data value between both 0 to 1 or 0 to 255 can be successfully plotted in the mode of cmap = 'gray' 
            en = encoder(view_data.view(-1, 28*28).type(torch.FloatTensor)/255.) 
            decoded_data = decoder(en) 
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data[i].data.numpy(), (28, 28)), cmap='gray') # 0-1
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())

            # Show in figure
            plt.draw()
            # Used to return the plot as an image rray
            # Draw the canvas, cache the renderer
            # Save every images in the list 'gif'
            f.canvas.draw()
            image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
            gif.append(image)
            
            plt.pause(0.05)

# Saved as a gif file under the root folder
imageio.mimsave('./autoencoder.gif', gif, fps=2.5)

plt.ioff()
plt.show()
                
# Visualization in 3D plot
view_data = train_data.data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data = encoder(view_data)
fig = plt.figure(2);
# 3D figure
ax = Axes3D(fig)
# 3 features as coordinate
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
# Label
values = train_data.targets[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    # Colors for every label
    c = cm.rainbow(int(255*s/9));
    # Set point coordinate, value being shown, color
    ax.text(x, y, z, s, backgroundcolor=c)
# Set x, y, z labels
ax.set_xlim(X.min(), X.max());
ax.set_ylim(Y.min(), Y.max());
ax.set_zlim(Z.min(), Z.max())
plt.show()

# Separate to two nn models and save them respectively
torch.save(encoder, 'encoder.pkl')
torch.save(decoder, 'decoder.pkl')

