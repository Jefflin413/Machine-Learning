import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Hyper parameters
BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001

# The example we use to demonstrate the performance of GAN is there exists two person,
# one is an artist, and another is a critic. The generator is the artist who is responsible for using ideas to create paintings.
# The discriminator is the critic whose job is to discriminate the works came from generator and the works came from master.
# We choose a range bounded by two quadratic equation (Y = 2*X^2 + 1 and Y = X^2) that represents the works from master.
N_IDEAS = 5
ART_COMPONENTS = 15
# Stack arrays in sequence vertically (row wise). → (64,15)
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

#                           X                                    Y = 2*X^2 + 1
#plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
#                           X                                       Y = X^2
#plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
#plt.legend(loc='upper right')
#plt.show()

# define real data
def artist_works():
    # Randomly pick value between 1 and 2, resulting a [BATCH_SIZE] array,
    # and then add one more dim for it. (64,) → (64,1)
    a = np.random.uniform(1, 2, size = BATCH_SIZE)[:, np.newaxis]
    # Create points y between y1 = x^2 and y2 = 2*x^2 + 1 
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1) # x = PAINT_POINTS = (64, 15), y = paintings = (64,15)
    paintings = torch.from_numpy(paintings).float()
    return paintings

# Construct Generator
G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),
    )

# Construct Discriminator
D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid(),
    )

opt_D = torch.optim.Adam(D.parameters(), lr = LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr = LR_G)

plt.ion() # Continous plotting
f = plt.figure()
gif = []
for step in range(10000):
    artist_paintings = artist_works() # (64,15)
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS) # Randomly sample and form (64,5)
    G_paintings = G(G_ideas) # Fake painting

    prob_artist0 = D(artist_paintings) # Score for real painting, which D should give as high as possible
    prob_artist1 = D(G_paintings) # Score for fake painting,  which D should give as low as possible

    # Pytorch can only decease loss
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True) # Reusing computational graph,
    # It's like lock the D parameters while updating G
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw = 3, label = 'Generated painting')
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw = 3, label = 'upper bound')
        plt.plot(PAINT_POINTS[0], np.power(PAINT_POINTS[0], 2), c='#FF9359', lw = 3, label = 'lower bound')
        plt.text(0, 0.6, 'D accuracy = %2f \n(0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        # 2*log(0.5)=-1.3862
        plt.text(0, 0.2, 'D score = %2f \n(-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3)); plt.legend(loc = 'upper right', fontsize = 10); plt.draw(); plt.pause(0.01)
        f.canvas.draw()
        image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
        gif.append(image)

imageio.mimsave('./GAN.gif', gif, fps=5)
plt.ioff()
plt.show()


        
    



