import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from utils import (
    pca_data,
    set_data_date,
    set_data_date_expanded,
    solar_columns,
    mean_columns,
    date_columns,
)
from scipy.optimize import curve_fit

# load lstm_model.pth
model = torch.load("lstm_model.pth")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load weather_solar_data.csv
weather_solar_data = pd.read_csv("wunderground/data/weather_solar_data.csv")
