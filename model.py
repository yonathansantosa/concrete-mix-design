import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class feedforward(nn.Module):
    def __init__(self):
        super(feedforward, self).__init__()

        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 8)
        self.fc5 = nn.Linear(8, 8)
        self.fc6 = nn.Linear(8, 8)
        self.fc7 = nn.Linear(8, 8)
        self.fc8 = nn.Linear(8, 8)
        self.fc9 = nn.Linear(8, 8)
        self.fc10 = nn.Linear(8, 5)
        self.fc11 = nn.Linear(5, 3)
        self.fc12 = nn.Linear(3, 1)
    
    def forward(self, x):
        out_fc1 = F.relu(self.fc1(x))
        out_fc2 = F.relu(self.fc2(out_fc1))
        out_fc3 = F.relu(self.fc3(out_fc2))
        out_fc4 = F.relu(self.fc4(out_fc3))
        out_fc5 = F.relu(self.fc5(out_fc4) + out_fc2)
        out_fc6 = F.relu(self.fc6(out_fc5))
        out_fc7 = F.relu(self.fc7(out_fc6) + out_fc5)
        out_fc8 = F.relu(self.fc8(out_fc7))
        out_fc9 = F.relu(self.fc9(out_fc8) + out_fc7)
        out_fc10 = F.relu(self.fc10(out_fc9))
        out_fc11 = F.relu(self.fc11(out_fc10))
        output = self.fc12(out_fc11)

        return output

class feedforward_50(nn.Module):
    def __init__(self):
        super(feedforward_50, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(8,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,1)
        )
    
    def forward(self, x):
        output = self.fc(x)

        return output

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, (2, 4)),
            nn.Sigmoid()
        )

        self.ff = nn.Linear(8,1)
        
    def forward(self, x):
        out_conv = self.conv(x).view(x.shape[0],8)
        out = self.ff(out_conv)
        return out

class MyEnsemble(nn.Module):
    def __init__(self, models, b=1):
        super(MyEnsemble, self).__init__()
        self.models = models
        self.aggregate = nn.Sequential(
            nn.Linear(b,b),
            nn.ReLU(),
            nn.Linear(b,1)
        )
        self.b = b

        self.divisor = nn.Parameter(torch.rand(1, b), requires_grad=True)
        
    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_prime = torch.zeros(self.b, x.shape[0]).to(device)
        for i in range(self.b):
            x_prime[i] = self.models[i].forward(x).squeeze()

        x_prime_t = x_prime.view(x.shape[0], self.b)
        # out = torch.sum(x_prime_t * self.divisor, dim=1, keepdim=True) 
        # out = self.aggregate(x_prime.view(x.shape[0], self.b))
        out = torch.mean(x_prime.view(x.shape[0], self.b), dim=1, keepdim=True)
        return out

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        # loss = self.mse(yhat, y) + self.eps
        loss = torch.sqrt(torch.mean((yhat-y)**2) + self.eps)
        return loss