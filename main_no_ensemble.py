import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.autograd import Variable

from cross_val import cross_val

import argparse
from tqdm import trange, tqdm
import os

import numpy as np

from data.dataset import Concrete
from model import feedforward, MyEnsemble, cnn, feedforward_50, RMSELoss

# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--maxepoch', default=1000, help='maximum iteration (default=1000)')
parser.add_argument('--lr', default=0.1, help='learing rate (default=0.1)')
parser.add_argument('--batch', default=32, help='minibatch size (default=32)')
parser.add_argument('--seed', default=2, help='number of bag (default=2)')
parser.add_argument('--bsize', default=100, help='number of bag size sample (default=100)')
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--wandb', default=False, action='store_true')
parser.add_argument('--model', default='feedforward')


args = parser.parse_args()
cloud_dir = '/content/gdrive/My Drive/'
saved_model_path = f'trained'
if not args.local: saved_model_path = cloud_dir + saved_model_path
if not os.path.exists(saved_model_path): os.makedirs(saved_model_path)
random_seed = int(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(random_seed)
torch.manual_seed(random_seed)


# Creating dataset
batch_size = int(args.batch)
val_batch_size = int(args.batch)
test_batch_size = int(args.batch)
test_split = .8
k = 5
dataset_folder = 'data/'
data = Concrete(dataset_folder+'concrete.csv', model_name=args.model)

# Creating dataset split
data_size = len(data)
indices = list(range(data_size))
split = int(np.floor(test_split * data_size))
np.random.shuffle(indices)

if args.wandb: wandb.init(project="concrete-mix-design", name="no bagging")

# Creating PT data samplers and loaders:
train_indices, test_indices = indices[:split], indices[split:]
# train_indices = np.random.choice(train_indices, size=(int(args.bsize)))
test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(data, batch_size=test_batch_size, sampler=test_sampler)
data.X_mean = data.X[train_indices[:]].mean(dim=0)
data.X_std = data.X[train_indices[:]].std(dim=0)

data.y_mean = data.y[train_indices[:]].mean(dim=0)
data.y_std = data.y[train_indices[:]].std(dim=0)

# Hyperparameter
learning_rate = float(args.lr)
max_epoch = int(args.maxepoch)
momentum=0.1

if args.model == 'feedforward':
    model = feedforward()
elif args.model == 'feedforward_50':
    model = feedforward_50()
else:
    model = cnn()

model.to(device)
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.99, eps=1.0e-8)
criterion = nn.MSELoss()

best_params = cross_val(train_indices, optimizer, criterion, model, data, batch_size, k, max_epoch)
model.load_state_dict(best_params)
if args.wandb: wandb.watch(model)

model.eval()
test_loss = 0.

for it, (X, y) in enumerate(test_loader):
    model.zero_grad()
    inputs = Variable(X, requires_grad=True).to(device)
    output = model.forward(inputs)
    target = Variable(y.unsqueeze(1)).to(device)
    test_loss += F.mse_loss(output, target, reduction='sum').sum().data.cpu().item()/len(test_indices)

if args.wandb:
    train_loss = np.loadtxt('train.csv')
    validation_loss = np.loadtxt('validation.csv')
    wandb.log({'Test Loss': test_loss})
    for i, (t, v) in enumerate(zip(train_loss, validation_loss)):
        wandb.log({"Train Loss": t}, step=i+1)
        wandb.log({"Validation Loss": v}, step=i+1)
