import torch
import os
import numpy as np
import pandas as pd
from torch import nn


class Shallow_cnn(nn.Module):
    def __init__(self):
