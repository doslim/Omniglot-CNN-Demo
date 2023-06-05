import torch.nn as nn
import torch

class Vanilla_CNN(nn.Module):
    def __init__(self):
        super(Vanilla_CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), # input channel, ouput channel, filter size, stride, padding
            nn.BatchNorm2d(32), # num_features
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # kernel size, stride, padding
        
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,50),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # resize, first axis is batch_num
        return self.fc(out)


class FC_Network(nn.Module):
    def __init__(self):
        super(FC_Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,50),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.fc(x)
        return out


class CNN_classifier(nn.Module):
    def __init__(self):
        super(CNN_classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), # input channel, ouput channel, filter size, stride, padding
            nn.BatchNorm2d(64), # num_features
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # kernel size, stride, padding
        
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 50)
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # resize, first axis is batch_num
        return self.fc(out)


class Vanilla_CNN_no_dropout(nn.Module):
    def __init__(self):
        super(Vanilla_CNN_no_dropout, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), # input channel, ouput channel, filter size, stride, padding
            nn.BatchNorm2d(32), # num_features
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # kernel size, stride, padding
        
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,50),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # resize, first axis is batch_num
        return self.fc(out)
    

class Vanilla_CNN_no_BN(nn.Module):
    def __init__(self):
        super(Vanilla_CNN_no_BN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), # input channel, ouput channel, filter size, stride, padding
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # kernel size, stride, padding
        
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512,50),
            nn.Dropout(0.3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # resize, first axis is batch_num
        return self.fc(out)
    

class CNN_classifier_no_dropout(nn.Module):
    def __init__(self):
        super(CNN_classifier_no_dropout, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), # input channel, ouput channel, filter size, stride, padding
            nn.BatchNorm2d(64), # num_features
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # kernel size, stride, padding
        
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 50)
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # resize, first axis is batch_num
        return self.fc(out)
    

class CNN_classifier_no_BN(nn.Module):
    def __init__(self):
        super(CNN_classifier_no_BN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), # input channel, ouput channel, filter size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # kernel size, stride, padding
        
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 50)
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # resize, first axis is batch_num
        return self.fc(out)
    

class CNN_classifier_60(nn.Module):
    def __init__(self):
        super(CNN_classifier_60, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), # input channel, ouput channel, filter size, stride, padding
            nn.BatchNorm2d(64), # num_features
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # kernel size, stride, padding
        
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 60)
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # resize, first axis is batch_num
        return self.fc(out)
    

class Vanilla_CNN_60(nn.Module):
    def __init__(self):
        super(Vanilla_CNN_60, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), # input channel, ouput channel, filter size, stride, padding
            nn.BatchNorm2d(32), # num_features
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # kernel size, stride, padding
        
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,60),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # resize, first axis is batch_num
        return self.fc(out)
    