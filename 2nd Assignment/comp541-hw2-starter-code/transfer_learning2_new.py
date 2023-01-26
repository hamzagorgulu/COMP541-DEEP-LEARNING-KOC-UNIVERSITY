import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings

BATCH_SIZE = 32
EPOCHS = 100
LR = 5e-5


path = "data"
alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', weights="DEFAULT")  
warnings.filterwarnings("ignore", category=UserWarning) 

class GetConvLayer(torch.nn.Module):
    def __init__(self, model):
        super(GetConvLayer, self).__init__()
        self.model = model
        self.features = model.features
        self.conv4layer = self.features[-3:-1]
        self.conv4layer[0] = nn.Conv2d(3, 300, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4layer[1] = nn.ReLU(inplace=True)
        #self.conv4layer[2] = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv4layer[2] = nn.Dropout(p=0.5, inplace=False)


    def get_conv4layer(self):
        return self.conv4layer

class FeatureExtractor(torch.nn.Module):
    def __init__(self, layer):
        super(FeatureExtractor, self).__init__()
        self.layer = layer
    def forward(self, x):
        #print(f"before feature extractor: {x.shape}")
        x = self.layer(x)
        #print(f"before feature extractor: {x.shape}")
        return x

class Classifier(torch.nn.Module):
    def __init__(self, input_size, output_size): 
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.7, inplace=False),
            nn.Flatten(),
            nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.layer(x)

class AlexNet(torch.nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(AlexNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class Dataloader:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = ImageFolder(self.path, transform=self.transform)
        
        self.train_size = int(0.8 * len(self.dataset))
        self.val_size = int(0.1 * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.val_size

        self.trainset, self.valset, self.testset = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size, self.test_size])

        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.valloader = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def lens(self):
        return len(self.trainset), len(self.valset), len(self.testset)

    def get_loaders(self):
        return self.trainloader, self.valloader, self.testloader

if __name__ == "__main__":
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = [] 

    warnings.filterwarnings("ignore")
    dataloader = Dataloader(path, BATCH_SIZE)
    trainloader, valloader, testloader = dataloader.get_loaders()
    len_train, len_val, len_test = dataloader.lens()

    conv4layer = GetConvLayer(alexnet).get_conv4layer()
    feature_extractor = FeatureExtractor(conv4layer)
    classifier = Classifier(43200, 40)
    alexnet = AlexNet(feature_extractor, classifier)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(alexnet.parameters(), lr=LR, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        alexnet.train()
        for input, label in trainloader:
            optimizer.zero_grad()
            output = alexnet(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (output.argmax(1) == label).sum().item() 
        
        alexnet.eval()
        with torch.no_grad():
            for image, label in valloader:
                output = alexnet(image)
                loss = criterion(output, label)
                val_loss += loss.item()
                val_acc += (output.argmax(1) == label).sum().item() 
        
        lr_scheduler.step()
        
        train_losses.append(train_loss / len(trainloader)) 
        val_losses.append(val_loss / len(valloader))
        train_accs.append(train_acc / len_train)
        val_accs.append(val_acc / len_val)

        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.3f}, Train Acc: {train_accs[-1]:.3f}, Val Loss: {val_losses[-1]:.3f}, Val Acc: {val_accs[-1]:.3f}')

    print('Finished Training')

    plt.title("Loss")
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

    plt.title("Accuracy")
    plt.plot(train_accs, label='Training accuracy')
    plt.plot(val_accs, label='Validation accuracy')
    plt.legend(frameon=False)
    plt.show()

    alexnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in testloader:
            output = alexnet(image)
            total += label.size(0)
            correct += (output.argmax(1) == label).sum().item()
    print(f"Test Accuracy: {correct / total}")