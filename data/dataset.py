import numpy as np
import torch

class Concrete:
    def __init__(self, dataset_file, model_name):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = np.genfromtxt(dataset_file, delimiter=';', skip_header=1)
        # self.X = torch.from_numpy(dataset[:, :-1] / dataset[:, :-1].max(0)).float().to(device)
        self.X = torch.from_numpy(dataset[:, :-1]).float().to(device)
        self.X_mean = 0.
        self.X_std = 0.
        # self.x_mean = self.X.mean(dim=0)
        # self.x_std = self.X.std(dim=0)
        # self.X = (self.X - self.x_mean)/self.x_std
        
        self.y = torch.from_numpy(dataset[:, -1]).float().to(device)
        # self.y_mean = self.y.mean(dim=0)
        # self.y_std = self.y.std(dim=0)
        self.y_mean = 0.
        self.y_std = 0.
        # self.y = (self.y - self.y_mean)/self.y_std
        self.model_name = model_name

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.model_name == 'cnn':
            return (((self.X[index] - self.X_mean)/self.X_std).view(1,2,4), self.y[index])
        else:
            return ((self.X[index] - self.X_mean)/self.X_std, self.y[index])