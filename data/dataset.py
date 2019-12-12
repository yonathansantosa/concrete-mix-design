import numpy as np
import torch

class Concrete:
    def __init__(self, dataset_file, model_name):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = np.genfromtxt(dataset_file, delimiter=';', skip_header=1)
        # self.X = torch.from_numpy(dataset[:, :-1] / dataset[:, :-1].max(0)).float().to(device)
        self.X = torch.from_numpy(dataset[:, :-1]).float().to(device)
        self.x_bar = self.X.mean(dim=0)
        self.std = self.X.std(dim=0)
        self.X = (self.X - self.x_bar)/self.std
        self.y = torch.from_numpy(dataset[:, -1]).float().to(device)
        self.model_name = model_name

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.model_name == 'cnn':
            return (self.X[index].view(1,2,4), self.y[index])
        else:
            return (self.X[index], self.y[index])