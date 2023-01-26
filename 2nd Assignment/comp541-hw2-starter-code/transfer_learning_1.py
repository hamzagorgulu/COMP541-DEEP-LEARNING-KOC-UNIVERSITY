from os import walk
import os
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import time

BATCH_SIZE = 32
NUM_EPOCHS = 90
PATIENCE = 35
LR = 1e-3    # the learning rate should be proportional to the batch size
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
# other hyperparameters: dropout value, optimizer, loss function, weight initialization, etc.

path = "data"

# change device to mps if you have a GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device: ", device)
#print(os.listdir(path))  #classes

# calculate the mean and std of each channel on the dataset


transform = transforms.Compose([transforms.Resize(28),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)])


dataset = datasets.ImageFolder(path, transform=transform)
dataset.classes


# information about the dataset
#print("Number of images: ", len(dataset))  #2718 images
#print("Classes: ", dataset.classes)  #40 classes including male and female actors. 68 images per class


# split the dataset into train, validation and test
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

test_dataset.dataset.classes


# create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

import torchvision
#imshow(torchvision.utils.make_grid(images))

# print labels
#print(' - '.join('%5s' % dataset.classes[labels[j]] for j in range(BATCH_SIZE)))


# Multi-layer perceptron
class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28 * 28 *3 , 300),
      nn.ReLU(),
      nn.Dropout(0.6),
      #nn.Linear(300, 300),
      #nn.ReLU(),
      #nn.Dropout(0.5),
      nn.Linear(300, 40) # 40 class
    )        
  def forward(self, x):
    '''Forward pass'''
    # x = x.reshape(x.size(0),-1)
    return self.layers(x)


if __name__ == '__main__':
    start = time.time()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    mlp = MLP().to(device)

    # weight initialization
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    criterion = nn.CrossEntropyLoss()
    # hinge loss
    #criterion = nn.MultiMarginLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)  # 5e-4  is better than 1e-4
    # sgd with momentum
    #optimizer = torch.optim.SGD(mlp.parameters(), lr=LR, momentum=0.9)
    # sgd
    #optimizer = torch.optim.SGD(mlp.parameters(), lr=LR)
    # nestrov
    #optimizer = torch.optim.SGD(mlp.parameters(), lr=LR, momentum=0.9, nesterov=True)
    # adagrad
    # optimizer = torch.optim.Adagrad(mlp.parameters(), lr=LR)
    # rmsprop
    # optimizer = torch.optim.RMSprop(mlp.parameters(), lr=LR)  #62.4

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6, verbose=True)

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
    
        mlp.train()

        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = mlp(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (output.argmax(1) == label).sum().item() 

        mlp.eval()
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device)
                label = label.to(device)
                output = mlp(image)
                loss = criterion(output, label)
                val_loss += loss.item()
                val_acc += (output.argmax(1) == label).sum().item()

        train_losses.append(train_loss / len(train_loader)) 
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_acc / len(train_dataset))
        val_accs.append(val_acc / len(val_dataset))

        # apply early stopping
        if epoch > PATIENCE:
            if val_losses[-1] > val_losses[-PATIENCE]:
                print("Early stopping")
                break
        # scheduler
        scheduler.step(val_loss / len(val_loader))

        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.3f}, Train Acc: {train_accs[-1]:.3f}, Val Loss: {val_losses[-1]:.3f}, Val Acc: {val_accs[-1]:.3f}')
    end = time.time()

    print("Computation time: ", end - start)

    # test
    test_loss = 0
    test_acc = 0
    mlp.eval()
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = mlp(image)
            loss = criterion(output, label)
            test_loss += loss.item()
            test_acc += (output.argmax(1) == label).sum().item()

    print(f'Test Loss: {test_loss / len(test_loader):.3f}, Test Acc: {test_acc / len(test_dataset):.3f}')

    # save model
    torch.save(mlp.state_dict(), 'mlp.pth')

    # plot
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.legend()
    plt.show()

    plt.plot(train_accs, label='train acc')
    plt.plot(val_accs, label='val acc')
    plt.legend()
    plt.show()




# show an image
import matplotlib.pyplot as plt
import numpy as np
"""
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
"""
# get some random training images
dataiter = iter(dataset)
images, labels = next(dataiter)

