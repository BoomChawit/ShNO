import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy

from argparse import ArgumentParser
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer

import random
import math
from typing import List, Optional
from tqdm import tqdm

from function import *

input_x_file = "./datasets/1.07g/Input_1.07g1_x.csv"
input_y_file = "./datasets/1.07g/Input_1.07g1_v.csv"
output_disp_file = "./datasets/1.07g/Output_1.07g1_disp.csv"
output_acc_file = "./datasets/1.07g/Output_1.07g1_acc.csv"

input_x_df = pd.read_csv(input_x_file).dropna(axis=1, how='all')
input_y_df = pd.read_csv(input_y_file).dropna(axis=1, how='all')
output_acc_df = pd.read_csv(output_acc_file).dropna(axis=1, how='all')
output_disp_df = pd.read_csv(output_disp_file).dropna(axis=1, how='all')

input_acceleration_x = torch.tensor(input_x_df.iloc[8:].T.values, dtype=torch.float32)[:, ::2]
input_acceleration_x = input_acceleration_x.reshape(input_acceleration_x.shape[0], 1, input_acceleration_x.shape[1])

input_acceleration_y = torch.tensor(input_y_df.iloc[8:].T.values, dtype=torch.float32)[:, ::2]
input_acceleration_y = input_acceleration_y.reshape(input_acceleration_y.shape[0], 1, input_acceleration_y.shape[1])

parameter_dataset = torch.tensor(input_x_df.iloc[:7].T.values, dtype=torch.float32)
parameter_dataset = parameter_dataset.unsqueeze(2).repeat(1, 1, 1000)

displacement_dataset = torch.tensor(output_disp_df.iloc[8:].T.values, dtype=torch.float32)[:, ::2] 
displacement_dataset = displacement_dataset.reshape(displacement_dataset.shape[0], 1, displacement_dataset.shape[1])

acceleration_dataset = torch.tensor(output_acc_df.iloc[8:].T.values, dtype=torch.float32)[:, ::2] 
acceleration_dataset = acceleration_dataset.reshape(acceleration_dataset.shape[0], 1, acceleration_dataset.shape[1])

x_dataset = torch.cat([input_acceleration_x / 1000, input_acceleration_y, parameter_dataset], axis = 1)

y_dataset = torch.cat([displacement_dataset, acceleration_dataset], axis = 1)

x_dataset = x_dataset.permute(0, 2, 1)
y_dataset = y_dataset.permute(0, 2, 1)

ntrain = 600

x_trainset, x_testset = x_dataset[:ntrain, ::2, :], x_dataset[ntrain:, ::2, :]
y_trainset, y_testset = y_dataset[:ntrain, ::2, :], y_dataset[ntrain:, ::2, :]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_trainset, y_trainset), batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_testset, y_testset), batch_size=1, shuffle=True)

config = {
    "model": {
        "modes": [32], 
        "fc_dim": 256, 
        "act": "gelu", 
        "num_pad": 4
    },
    "train": {
        "epochs": 500,
        "milestones": [50, 100],
        "base_lr": 0.00005, 
        "scheduler_gamma": 0.5
    },
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_dim, out_dim = x_trainset.shape[2], y_trainset.shape[2]
print(in_dim, out_dim)

model = FNO1d(modes=config['model']['modes'],
              fc_dim=config['model']['fc_dim'],
              act=config['model']['act'],
              in_dim=in_dim, out_dim=out_dim).to(device)

print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['base_lr'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['train']['milestones'], gamma=config['train']['scheduler_gamma'])

rank = 0

model.train()
pbar_train = tqdm(range(config['train']['epochs']), dynamic_ncols=True, smoothing=0.1)

loss_of_train, loss_of_test = [], []
myloss = LpLoss(size_average=True)

for e in pbar_train:
    
    model.train()
    train_loss_l2 = 0.0

    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(rank), y_train.to(rank)

        out_train = model(x_train)
        train_data_loss = myloss(out_train, y_train)

        optimizer.zero_grad()
        train_data_loss.backward()
        optimizer.step()

        train_loss_l2 += train_data_loss.item()

    scheduler.step()

    train_loss_l2 /= len(train_loader)
    loss_of_train.append(train_loss_l2)

    # Evaluation on the test set
    model.eval()
    test_loss_l2 = 0.0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(rank), y_test.to(rank)

            out_test = model(x_test)
            test_data_loss = myloss(out_test, y_test)

            test_loss_l2 += test_data_loss.item()
            
    test_loss_l2 /= len(test_loader)
    loss_of_test.append(test_loss_l2)

    pbar_train.set_description((f'Epoch {e}, train loss: {train_loss_l2:.5f} ,  test loss: {test_loss_l2:.5f}'))

print('Completed!')


import numpy as np

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.ticker as ptick
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# loss plot

plt.figure(figsize=(6, 5))
plt.plot(loss_of_train, color="black")
plt.plot(loss_of_test, color="red")
plt.xlabel('iterations', fontsize = 16)
plt.ylabel('Total Loss', fontsize = 16)
plt.title('Total Loss', fontsize = 20)
plt.xlim(0, len(loss_of_train))
plt.grid(True)
plt.savefig("")
plt.savefig("./figures/loss.png", bbox_inches='tight', dpi=150)
plt.show()

# result plot

x_plot = np.linspace(0, 20, 500)
n = 610
validate_x = x_dataset[n, ::2, :].reshape(1, 500, 9).to(device)
validate_y = y_dataset[n, ::2, :].reshape(1, 500, 2).to(device)
validate_out = model(validate_x)

plt.figure(figsize=(15, 5))
plt.plot(x_plot, validate_y[0, :, 0].detach().cpu().numpy(), color = "black", label = "true")
plt.plot(x_plot, validate_out[0, :, 0].detach().cpu().numpy(), color = "red", label = "predict")
plt.grid(True, which='major', linestyle='-')
plt.grid(True, which='minor', linestyle='--')
plt.xlim(0, 20)
plt.legend()
plt.savefig("./figures/results.png", bbox_inches='tight', dpi=150)
plt.show()