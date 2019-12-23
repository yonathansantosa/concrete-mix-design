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
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--maxepoch', default=1000, help='maximum aggregate iteration (default=1000)')
parser.add_argument('--lr', default=0.1, help='aggregate learing rate (default=0.1)')
parser.add_argument('--batch', default=32, help='minibatch size (default=32)')
parser.add_argument('--b', default=2, help='number of bag (default=2)')
parser.add_argument('--seed', default=2, help='number of bag (default=2)')
parser.add_argument('--bsize', default=100, help='number of bag size sample (default=100)')
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--wandb', default=False, action='store_true')
parser.add_argument('--trainable_bag', default=False, action='store_true')
parser.add_argument('--model', default='feedforward')
parser.add_argument('--b_max', default=2)
parser.add_argument('--quiet', default=False, action='store_true')
parser.add_argument('--usebest', default=None)


args = parser.parse_args()
cloud_dir = '/content/gdrive/My Drive/'
saved_model_path = f'trained'
if not args.local: saved_model_path = cloud_dir + saved_model_path
if not os.path.exists(saved_model_path): os.makedirs(saved_model_path)
b_max = int(args.b)
random_seed = int(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if args.usebest is not None:
    usebest=int(args.usebest)


# Creating dataset
batch_size = int(args.batch)
val_batch_size = int(args.batch)
validation_split = .8
dataset_folder = 'data/'
# data = Concrete(dataset_folder+'concrete.csv', args.model)
data = Concrete(dataset_folder+'concrete_no_day1.csv', args.model)


# Creating dataset split
data_size = len(data)
indices = list(range(data_size))
split = int(np.floor(validation_split * data_size))
np.random.shuffle(indices)

# Creating PT data samplers and loaders:
train_indices, test_indices = indices[:split], indices[split:]

data.X_mean = data.X[train_indices[:]].mean(dim=0)
data.X_std = data.X[train_indices[:]].std(dim=0)

data.y_mean = data.y[train_indices[:]].mean(dim=0)
data.y_std = data.y[train_indices[:]].std(dim=0)

np.random.shuffle(train_indices)
np.random.shuffle(test_indices)
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(data, batch_size=val_batch_size, sampler=test_sampler)

# Hyperparameter
learning_rate = float(args.lr)
max_epoch = int(args.maxepoch)
momentum=0.1

# Creating dataset split
data_size = len(data)
indices = list(range(data_size))
split = int(np.floor(validation_split * data_size))
np.random.shuffle(indices)

models = []
models_param = []
for b in range(b_max):
    if args.model == 'feedforward':
        m = feedforward()
    elif args.model == 'feedforward_50':
        m = feedforward_50()
    else:
        m = cnn()
    m.load_state_dict(torch.load(f'{saved_model_path}/model-{b}.pth'))
    # m.parameters(require_grads=False)
    m.to(device)
    if args.trainable_bag: 
        m.train()
        models_param += list(m.parameters())
    else:
        m.eval()
    models += [m]

aggregate = MyEnsemble(models, b_max)
aggregate.to(device)
if args.trainable_bag:
    optimizer = optim.Adadelta(list(aggregate.parameters()) + models_param, lr=learning_rate, rho=0.99, eps=1.0e-8)
else:
    optimizer = optim.Adadelta(list(aggregate.parameters()), lr=learning_rate, rho=0.99, eps=1.0e-8)

if args.wandb: wandb.init(project="concrete-mix-design", name='aggregate')
if args.wandb: wandb.watch(aggregate)


criterion = nn.MSELoss()
'''
for epoch in trange(0, max_epoch, total=max_epoch, initial=0):
    aggregate.train()
    for it, (X, y) in enumerate(train_loader):
        aggregate.zero_grad()
        inputs = Variable(X, requires_grad=True).to(device)
        models_out, output = aggregate.forward(inputs)
        target = Variable(y.unsqueeze(1)).to(device)
        loss = criterion(output, target)
        # l1_norm = 0.
        # for p in aggregate.parameters():
        #     l1_norm += 1.0e-5*torch.norm(p, p=1)
        # loss += l1_norm
        loss.backward()
        # nn.utils.clip_grad_value_(aggregate.parameters(), 1)
        if args.wandb and it==0: 
            wandb.log({"Aggregate Train Loss": loss.data.cpu().item()}, step=epoch)
        # if it==0 and not args.quiet: tqdm.write(f'{models_out[0].detach().cpu().numpy()} =====> {output[0].data.cpu().item()} || {y[0].data.cpu().item()}')
        optimizer.step()
        optimizer.zero_grad()

    
    aggregate.eval()
    val_loss = 0.
    # table = wandb.Table(columns=["id", "Predicted Label", "True Label"])
#     c = 0

    for it, (X, y) in enumerate(test_loader):
        aggregate.zero_grad()
        inputs = Variable(X).to(device)
        _, output = aggregate.forward(inputs)
        target = Variable(y.unsqueeze(1)).to(device)
        val_loss += F.mse_loss(output, target, reduction='sum').sum().data.cpu().item()/len(test_indices)

        if it == 0 and not args.quiet and epoch % 5 == 0:
            X_test = (torch.tensor([139.6,209.4,0.0,192.0,0.0,1047.0,806.9,3]).to(device) - data.X_mean)/data.X_std
            y_test = torch.tensor([[8.06]])
            inputs_test = Variable(X_test)
            models_out, output = aggregate.forward(inputs_test)
            tqdm.write(f'{models_out[0].detach().cpu().numpy()} =====> {output[0].data.cpu().item():.2f} || {y_test[0].data.cpu().item():.2f}')
            # tqdm.write(f'{float(output[0].cpu().data)} ==> {float(target[0].cpu().data)}')
    if args.wandb:
        wandb.log({"Aggregate Validation Loss": val_loss}, step=epoch)
        # for o, t in zip(output.data.cpu().squeeze(), y.data):
        #     table.add_data(c, float(o), float(t))
        #     c += 1
        # wandb.log({"examples": table})
'''
X_test = (torch.tensor([139.6,209.4,0.0,192.0,0.0,1047.0,806.9,3]).to(device) - data.X_mean)/data.X_std
y_test = torch.tensor([[8.06]])
inputs_test = Variable(X_test)
models_out, output = aggregate.forward(inputs_test)
tqdm.write(f'{models_out[0].detach().cpu().numpy()} =====> {output[0].data.cpu().item():.2f} || {y_test[0].data.cpu().item():.2f}')
test_loss = 0.
for it, (X, y) in enumerate(test_loader):
    aggregate.zero_grad()
    inputs = Variable(X, requires_grad=True).to(device)
    _, output = aggregate.forward(inputs)
    target = Variable(y.unsqueeze(1)).to(device)
    test_loss += F.mse_loss(output, target, reduction='sum').sum().data.cpu().item()/len(test_indices)

    if it == 0 and not args.quiet:
        tqdm.write(f'{float(output[0].cpu().data)} ==> {float(target[0].cpu().data)}')

if args.wandb:
    x = np.arange(b_max)
    y = aggregate.divisor.squeeze().data.detach().cpu().numpy()
    fig, ax = plt.subplots()
    plt.bar(x, y)
    wandb.log({"Divisor": plt})
    wandb.log({"Test Loss": test_loss})

print(test_loss)