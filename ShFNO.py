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
output_disp_file = "./datasets/1.07g/Output_1.07g1_disp.csv"

input_x_df = pd.read_csv(input_x_file).dropna(axis=1, how='all')
input_v_df = pd.read_csv(input_v_file).dropna(axis=1, how='all')
output_acc_df = pd.read_csv(output_acc_file).dropna(axis=1, how='all')
output_disp_df = pd.read_csv(output_disp_file).dropna(axis=1, how='all')