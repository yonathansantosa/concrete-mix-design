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
        out_fc1 = torch.tanh(self.fc1(x))
        out_fc2 = torch.tanh(self.fc2(out_fc1))
        out_fc3 = torch.tanh(self.fc3(out_fc2))
        out_fc4 = torch.tanh(self.fc4(out_fc3))
        out_fc5 = torch.tanh(self.fc5(out_fc4) + out_fc2)
        out_fc6 = torch.tanh(self.fc6(out_fc5))
        out_fc7 = torch.tanh(self.fc7(out_fc6) + out_fc5)
        out_fc8 = torch.tanh(self.fc8(out_fc7))
        out_fc9 = torch.tanh(self.fc9(out_fc8) + out_fc7)
        out_fc10 = torch.tanh(self.fc10(out_fc9))
        out_fc11 = torch.tanh(self.fc11(out_fc10))
        output = self.fc12(out_fc11)

        return output

class MyEnsemble(nn.Module):
    def __init__(self, models, b=1):
        super(MyEnsemble, self).__init__()
        self.models = models
        self.classifier = nn.Sequential(
                            nn.Linear(b, b),
                            nn.Tanh(),
                            nn.Linear(b,1)
        )
        self.b = b
        
    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_prime = torch.zeros(self.b, x.shape[0]).to(device)
        for i in range(self.b):
            x_prime[i] = self.models[i].forward(x).squeeze()/1000

        out = self.classifier(x_prime.view(x.shape[0], self.b))
        return out