import numpy as np
import torch

class Concrete:
    def __init__(self, dataset_file):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = np.genfromtxt(dataset_file, delimiter=';', skip_header=1)
        mu = 0
        std = 0.5
        self.X = torch.from_numpy(dataset[:, :-1] / dataset[:, :-1].max(0)).float().to(device)
        # self.X = torch.from_numpy(dataset[:, :-1]).float().to(device)
        self.y = torch.from_numpy(dataset[:, -1]).float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.X[index].view(1,2,4), self.y[index])