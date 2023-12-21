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