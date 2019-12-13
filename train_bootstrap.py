import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.autograd import Variable

import argparse
from tqdm import trange, tqdm
import os
import wandb

import numpy as np

from data.dataset import Concrete
from model import feedforward, MyEnsemble, cnn, feedforward_50, RMSELoss

# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--maxepoch', default=1000, help='maximum iteration (default=1000)')
parser.add_argument('--agr_maxepoch', default=1000, help='maximum aggregate iteration (default=1000)')
parser.add_argument('--lr', default=0.1, help='learing rate (default=0.1)')
parser.add_argument('--batch', default=32, help='minibatch size (default=32)')
parser.add_argument('--b', default=2, help='number of bag (default=2)')
parser.add_argument('--seed', default=2, help='number of bag (default=2)')
parser.add_argument('--bsize', default=100, help='number of bag size sample (default=100)')
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--wandb', default=False, action='store_true')
parser.add_argument('--trainable_bag', default=False, action='store_true')
parser.add_argument('--b_max', default=2)
parser.add_argument('--model', default='feedforward')
parser.add_argument('--quiet', default=False, action='store_true')


args = parser.parse_args()
cloud_dir = '/content/gdrive/My Drive/'
saved_model_path = f'trained'
if not args.local: saved_model_path = cloud_dir + saved_model_path
if not os.path.exists(saved_model_path): os.makedirs(saved_model_path)
b = int(args.b)
b_max = int(args.b_max)
random_seed = int(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(random_seed)
torch.manual_seed(random_seed)

random_seed = np.random.randint(100, size=b_max)

# Creating dataset
batch_size = int(args.batch)
val_batch_size = int(args.batch)
validation_split = .8
dataset_folder = 'data/'
data = Concrete(dataset_folder+'concrete.csv', args.model)

# Creating dataset split
data_size = len(data)
indices = list(range(data_size))
split = int(np.floor(validation_split * data_size))
np.random.shuffle(indices)

if args.wandb: wandb.init(project="concrete-mix-design", name=f'bootstrap', notes=f'{b}')

# Creating PT data samplers and loaders:
train_indices, val_indices = indices[:split], indices[split:]
np.random.seed(random_seed[int(args.b)])

data.X_mean = data.X[train_indices[:]].mean(dim=0)
data.X_std = data.X[train_indices[:]].std(dim=0)

data.y_mean = data.y[train_indices[:]].mean(dim=0)
data.y_std = data.y[train_indices[:]].std(dim=0)

train_indices = np.random.choice(train_indices, size=int(np.floor(len(train_indices) * .7)))

np.random.shuffle(val_indices)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(data, batch_size=val_batch_size, sampler=valid_sampler)

torch.autograd.set_detect_anomaly(True)
# Hyperparameter
learning_rate = float(args.lr)
if args.model == 'feedforward':
    model = feedforward()
elif args.model == 'feedforward_50':
    model = feedforward_50()
else:
    model = cnn()
model.to(device)
max_epoch = int(args.maxepoch)
momentum=0.1

if args.wandb: wandb.watch(model)
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.99, eps=1.0e-8)
criterion = nn.MSELoss()

for epoch in trange(0, max_epoch, total=max_epoch, initial=0):
    model.train()
    for it, (X, y) in enumerate(train_loader):
        model.zero_grad()
        inputs = Variable(X, requires_grad=True).to(device)
        output = model.forward(inputs)
        target = Variable(y.unsqueeze(1)).to(device)
        loss = criterion(output, target)
        # l1_norm = 0.
        # for p in model.parameters():
        #     l1_norm += 1.0e-5*torch.norm(p, p=1)
        # loss += l1_norm
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), 10)
        if args.wandb:
            wandb.log({"Train Loss": loss.data.cpu().item()}, step=epoch)
        if it == 0 and not args.quiet:
            tqdm.write(f'{float(output[0].cpu().data)} ==> {float(target[0].cpu().data)}')

        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    val_loss = 0.
    for it, (X, y) in enumerate(validation_loader):
        model.zero_grad()
        inputs = Variable(X, requires_grad=True).to(device)
        output = model.forward(inputs)
        target = Variable(y.unsqueeze(1)).to(device)
        val_loss += F.mse_loss(output, target, reduction='sum').sum().data.cpu().item()
    if args.wandb: wandb.log({"Validation Loss": val_loss/len(val_indices)}, step=epoch)

torch.save(model.state_dict(), f'{saved_model_path}/model-{b}.pth')

