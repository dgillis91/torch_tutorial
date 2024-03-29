# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 21:35:27 2019

@author: barre
"""
import os
# os.chdir('DATA_PATH')
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Random Seeds
torch.manual_seed(42)

# Setup transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Downlaod data
trainset = datasets.MNIST('./data/', download=True,
                          train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

testset = datasets.MNIST('./data/', download=True,
                         train=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=True)

# Shapes
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.shape)
print(labels.shape)
testiter = iter(testloader)
test_image, test_label = testiter.next()
print(test_image.shape)
print(test_label.shape)

# Display first image
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Build conv layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8,
                                     kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=32,
                                     kernel_size=5)
        
        # Dense layers
        self.dense1 = torch.nn.Linear(in_features=512, out_features=64)
        self.dense2 = torch.nn.Linear(in_features=64, out_features=10)
        
    def forward(self, x):
        # Push x throuh conv layer 1, relu, max pool
        x = torch.nn.functional.relu(input=self.conv1(x))
        x = torch.nn.functional.max_pool2d(input=x, kernel_size=(2, 2))
        
        # Push x through conv layer 2, relu, max pool
        x = torch.nn.functional.relu(input=self.conv2(x))
        x = torch.nn.functional.max_pool2d(input=x, kernel_size=(2, 2))
        
        # Flatten
        x = x.view(-1, self.__num_flat_features(x))
        
        # Push through fully connected 1, relu
        x = torch.nn.functional.relu(input=self.dense1(x))
        
        # Push through final connected layer
        x = self.dense2(x)
        
        return x
    
    def __num_flat_features(self, x):
        """Helper method to get shape of individual comeponents of tensor"""
        # The shape without num samples
        size = x.size()[1:]
        # print(size)
        # Shapes multiplied to give the number of features
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetTrainer:
    def __init__(self):
        # Initialze Net
        self.net = Net()
        print('Model Structure:\n')
        print(self.net)
        
        # Cuda/CPU Selection
        self.__select_device()
    
    def __select_device(self):
        if torch.cuda.is_available():
            print('cuda available')
            self.__device = torch.device('cuda')
        else:
            print('cuda not available, using cpu')
            self.__device = torch.device('cpu')
        self.net.to(self.__device)
        
    def __train_step(self, inputs, targets):
        # Zero out gradients
        self.__optimizer.zero_grad()
        
        # Forward pass
        outputs = self.net(inputs)
        
        # Calc loss
        loss = self.__loss_metric(outputs, targets)
        
        # Grad and update
        loss.backward()
        self.__optimizer.step()
        
        # Update loss
        self.__running_loss += loss.item()
        
    def train_network(self, num_epochs=3, learning_rate=0.001, print_mod=100):
        # Model in training mode
        self.net.train()
        
        # Optimizer init
        self.__optimizer = torch.optim.Adam(self.net.parameters(),
                                            lr=learning_rate)
        
        # Loss metric init
        self.__loss_metric = torch.nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(num_epochs):
            print('Running epoch: ' + str(epoch + 1))
            self.__running_loss = 0
            
            for i, batch in enumerate(trainloader):
                
                # Bind batch to device
                inputs = batch[0].to(self.__device)
                targets = batch[1].to(self.__device)
                
                # Run train step
                self.__train_step(inputs, targets)
                
                # Evaluation/accuracy printing
                if (i % print_mod == 0) and (i >= print_mod):
                    normalized_loss = self.__running_loss / (print_mod * 1.0)
                    print('Loss: ' + str(normalized_loss))
                    self.__running_loss = 0
                    
    def score(self, validation_or_test_loader):
        # Init accuracy metrics
        validation_or_test_loss = 0
        correct = 0
        
        # Eval Mode
        self.net.eval()
        with torch.no_grad():
            
            for batch in validation_or_test_loader:
                
                # Bind batch to device
                inputs = batch[0].to(self.__device)
                targets = batch[1].to(self.__device)
                
                # Make prediction
                outputs = self.net(inputs)
                value, index = torch.max(outputs, 1)
                
                # Calc loss
                validation_or_test_loss += \
                    self.__loss_metric(outputs, targets).item()
                
                # Number correct
                correct += index.eq(targets.data.view_as(index)).sum()
                
        # Final calc of metrics
        num_observations = (len(validation_or_test_loader.dataset) * 1.0)
        normalized_loss = validation_or_test_loss / num_observations
        accuracy = (correct.item() * 100.0) / num_observations
        print('Loss: ' + str(normalized_loss))
        print('Accuracy: ' + str(accuracy))
                    
    def predict(self, image_data):
        # Show image
        print('Image to be predicted:\n')
        plt.imshow(image_data.numpy().squeeze(), cmap='gray_r')
        
        # Eval Model
        self.net.eval()
        with torch.no_grad():
            predictions = self.net(image_data.unsqueeze(0).to(self.__device))
            value, index = torch.max(predictions, 1)
            print('Predicted class: ' + str(index.item()))
        
    
# Init net
covnet = NetTrainer()

# Train net
covnet.train_network()

# Score Model
covnet.score(testloader)

# Test Single Prediction
covnet.predict(images[0])
