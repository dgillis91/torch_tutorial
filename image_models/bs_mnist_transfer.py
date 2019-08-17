# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 21:35:27 2019

@author: barre
"""
import os
# os.chdir('DATA PATH')
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# Random Seeds
torch.manual_seed(42)

# Setup transforms
transform = transforms.Compose([transforms.Grayscale(3),
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

# Downlaod data
trainset = datasets.MNIST('./data4/', download=True,
                          train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)

testset = datasets.MNIST('./data4/', download=True,
                         train=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
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
plt.imshow(images[0][0, :, :].numpy(), cmap='gray_r')


def get_pretrained_vgg_net(num_classes):
    # Load pretrained model
    pretrained_model = models.vgg11_bn(pretrained=True)
    
    # Freeze training for all layers
    for param in pretrained_model.features.parameters():
        param.require_grad = False
    
    # Get number of input features in last classifier layer
    num_last_layer_input_features = pretrained_model.classifier[-1].in_features
    
    # Grab model up to last layer
    features = list(pretrained_model.classifier.children())[:-1]
    new_last_layer = torch.nn.Linear(num_last_layer_input_features,
                                     num_classes)
    features.extend([new_last_layer])
    
    # Overwrite classifier with new structure
    pretrained_model.classifier = torch.nn.Sequential(*features)
    
    return pretrained_model


def get_pretrained_squeeze_net(num_classes):
    # Load pretrained model
    pretrained_model = models.squeezenet1_1(pretrained=True)
    
    # Freeze training for all layers
    for param in pretrained_model.features.parameters():
        param.require_grad = False
    
    # Get number of input features in last classifier layer
    cov_last_input = pretrained_model.classifier[1].in_channels
    cov_last_output = pretrained_model.classifier[1].out_channels
    
    # Grab model layer to update
    features = list(pretrained_model.classifier.children())
    features[1] = torch.nn.Conv2d(cov_last_input, num_classes,
                                  kernel_size=(1, 1), stride=(1, 1))
    
    # Overwrite classifier with new structure
    pretrained_model.classifier = torch.nn.Sequential(*features)
    pretrained_model.num_classes = num_classes
    
    return pretrained_model


class NetTrainer:
    def __init__(self):
        # Initialze Net
        self.net = get_pretrained_squeeze_net(num_classes=10)
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
                    
                # Clean up cuda cache
                torch.cuda.empty_cache()
                    
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
        plt.imshow(image_data[0, :, :].numpy().squeeze(), cmap='gray_r')
        
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
