import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

import numpy as np

import matplotlib.pyplot as plt
import imageio

# Set GPU as device if it is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# Obtain datat from MNIST digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

# Training data
# Transform data type to Tensor from pixel or numpy array
# torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
    )

# Plot one example
"""
print(train_data.data.size())                 # (60000, 28, 28)
print(train_data.targets.size())               # (60000)
plt.imshow(train_data.data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()
"""

# Testing data
test_data = torchvision.datasets.MNIST(root = './mnist/', train = False)
# For matching weights of shape([nb_filters, in_channels, kernel_h, kernel_w])
# Reshape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# Range of gray scale is 0-255
# a = torch.unsqueeze(b, c) Returns a new tensor 'a' with a dimension of size 'c' inserted at the specified position of 'b'.
test_x = torch.unsqueeze(test_data.data, dim = 1).type(torch.FloatTensor)[:2000]/255
test_y = test_data.targets[:2000]
# Send testing data into device
test_x = test_x.to(device)
test_y = test_y.to(device)
# Dataloader for mini-batch
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)

# Define CNN structure
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # Convolution (BatchSize, 1, 28, 28)
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride = 1,
                # If want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
                padding = 2,
                ),
            # After convolution (BatchSize, 16, 28, 28)
            # Activation function
            nn.ReLU(),
            # Pooling, output width and length = input width and length/pooling kernel size  (output BatchSize, 16, 14, 14)
            nn.MaxPool2d(kernel_size = 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2), # (BatchSize, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2), # (BatchSize, 32, 7, 7)
            )
        # Fully connected layer as classifier
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten x, reshape from (BatchSize,32,7,7) to (BatchSize,32*7*7)
        # x.view(c,-1) -1 represent the remain dimension will be the product of all dimensions except c
        x = x.view(x.size(0), -1)
        output = self.out(x)
        # For visualization, we make the CNN return both input and output of classifier
        return output, x

# Construct CNN
cnn = CNN()
cnn = cnn.to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()

# The following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    
    # Used to return the plot as an image rray
    # Draw the canvas, cache the renderer
    # Save every images in the list 'gif'  
    f.canvas.draw()
    image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
    gif.append(image)
    
    plt.show()
    plt.pause(0.01)

plt.ion()
# For the purpose of storing gif
f = plt.figure()
gif = []

# Training process
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        # Send training data to GPU
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        # For visualization, we make the CNN return both input and output of classifier
        output = cnn(b_x)[0] 
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Every element used to plot figure is passed back to cpu by the command .cpu().detach().numpy()
        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].cpu().detach().numpy()
            accuracy = float((pred_y == test_y.cpu().detach().numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().detach().numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                # Using principal component analysis to decrease the dimension of input of classifier to 2D
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.cpu().detach().numpy()[:plot_only, :])
                labels = test_y.cpu().detach().numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)

plt.ioff()

# Testing
test_output, _ =  cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1]
imageio.mimsave('./CNN.gif', gif, fps=2.5)
print(pred_y.cpu().detach().numpy(), 'prediction number')
print(test_y[:10].cpu().detach().numpy(), 'real number')

    


        
                

