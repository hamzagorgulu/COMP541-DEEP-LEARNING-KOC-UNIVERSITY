# alexnet as fixed feature extractor
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
model

BATCH_SIZE = 32
EPOCHS = 10

# start out with only last conv activation as feature
layer = model.features[-3:-1]
layer[0] = nn.Conv2d(3, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))



class FeatureExtractor(torch.nn.Module):
    def __init__(self, layer):
        super(FeatureExtractor, self).__init__()
        self.layer = layer
    def forward(self, x):
        print(f"before feature extractor: {x.shape}")
        x = self.layer(x)
        print(f"before feature extractor: {x.shape}")
        return x

feature_extractor = FeatureExtractor(layer)
feature_extractor

# feature axtractor as input, fully connected layer as output
class Classifier(torch.nn.Module):
    def __init__(self, input_size, output_size): 
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
    def forward(self, x):
        print(f"before fc: {x.shape}")
        x = x.view(x.size(0), -1) # flatten
        print(f"after view: {x.shape}")
        x = self.fc(x)
        print(f"after fc: {x.shape}")
        return x


# create classifier object considering the output of the feature extractor
classifier = Classifier(256*256*256, 40)

# merge feature extractor and classifier
class AlexNet(torch.nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(AlexNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
    def forward(self, x):
        x = self.feature_extractor(x)
        # print the shape of the feature extractor output
        x = self.classifier(x)
        return x

# create an example image
example = torch.rand(32, 3, 256, 256)
# insert into AlexNet
alexnet = AlexNet(feature_extractor, classifier)
alexnet(example)




# load data


transform = transforms.Compose(
    [transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = ImageFolder(root='./data', transform=transform)

train_len = int(len(dataset) * 0.8)
val_len = int(len(dataset) * 0.1)
test_len = len(dataset) - train_len - val_len

trainset, val_set, testset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=2)
valloader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle=True, num_workers=2)

# info about trainloader
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(images.shape) # 1 batch: [32, 3, 256, 256]  : 32 images, 3 channels, 256x256 pixels
print(labels.shape) # [32] : 32 labels

# train model


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

model = AlexNet(feature_extractor, classifier).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(model, trainloader, valloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))
    print('Finished Training')

# test model
def test(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

correct = 0
total = 0


train(model, trainloader, valloader, criterion, optimizer, EPOCHS)
test(model, testloader)











