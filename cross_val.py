import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.autograd import Variable

import numpy as np

from tqdm import trange, tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cross_val(
    train_indices,
    optimizer,
    criterion,
    model,
    data=torch.zeros(1,8), 
    batch_size=32, 
    k=1, 
    max_epoch=10,
    ):
    val_loss_min = np.inf
    best_params = None
    np.random.seed(0)
    torch.manual_seed(0)
    # np.random.shuffle(train_indices)
    cross_split = int(np.floor(len(train_indices) / k))

    for fold in range(k):
        train_cross_indices = np.delete(train_indices, np.s_[fold*cross_split:(fold+1)*cross_split])
        train_sampler = SubsetRandomSampler(train_cross_indices)
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)

        validation_cross_indices = train_indices[fold*cross_split:(fold+1)*cross_split]
        validation_sampler = SubsetRandomSampler(validation_cross_indices)
        validation_loader = DataLoader(data, batch_size=batch_size, sampler=validation_sampler)
        
        model.reset()
        model.to(device)
        train_loss = np.zeros(max_epoch)
        validation_loss = np.zeros(max_epoch)

        optimizer = optim.Adadelta(model.parameters(), lr=1, rho=0.99, eps=1.0e-8)

        for epoch in trange(0, max_epoch, total=max_epoch, initial=0):
            model.train()
            for it, (X, y) in enumerate(train_loader):
                model.zero_grad()
                inputs = Variable(X, requires_grad=True).to(device)
                output = model.forward(inputs)
                target = Variable(y.unsqueeze(1)).to(device)
                loss = criterion(output, target)
                loss.backward()
                
                if it==0: train_loss[epoch] = float(loss.data.cpu().item())

                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            val_loss = 0.
            for it, (X, y) in enumerate(validation_loader):
                model.zero_grad()
                inputs = Variable(X, requires_grad=True).to(device)
                output = model.forward(inputs)
                target = Variable(y.unsqueeze(1)).to(device)
                val_loss += F.mse_loss(output, target, reduction='sum').data.cpu().item()
            validation_loss[epoch] = float(val_loss/(len(validation_cross_indices))

        if validation_loss[-1] < val_loss_min:
            val_loss_min = validation_loss[-1]
            np.savetxt("train.csv", train_loss, delimiter=",")
            np.savetxt("validation.csv", validation_loss, delimiter=",")
            best_params = model.state_dict()
            None
    print(val_loss_min)
    return best_params