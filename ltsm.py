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

start_date = "2022-06-01"
end_date = "2022-06-30"

weather_solar_data = pd.read_csv("wunderground/data/weather_solar_data.csv")

weather_solar_data = set_data_date_expanded(weather_solar_data, "obsTimeLocal")

# print(weather_solar_data.columns)
# weather_solar_data["obsTimeLocal"] = pd.to_datetime(weather_solar_data["obsTimeLocal"])

print(weather_solar_data.head())
weather_solar_data.fillna(0, inplace=True)

weather_solar_data = weather_solar_data[
    (weather_solar_data["obsTimeLocal"] >= start_date)
    & (weather_solar_data["obsTimeLocal"] < end_date)
]

### set weather_solar_data obsTimeLocal to float
# weather_solar_data["obsTimeLocal"] = weather_solar_data["obsTimeLocal"].astype(int)

# break the data into 1 day increments
weather_data = weather_solar_data[mean_columns + date_columns]
solar_data = weather_solar_data[solar_columns + date_columns]

# break data into groups of 24 hours
# loop through each day in a year
binned_data_weather = []
binned_data_solar = []

# for each day in between start and end date

start_date_dt = pd.to_datetime(start_date)
end_date_dt = pd.to_datetime(end_date)

while start_date_dt <= end_date_dt:
    start_date_timestamp = start_date_dt
    end_date_timestamp = start_date_dt + pd.Timedelta(days=1)
    # get the data for that day
    day_data_weather = weather_data[
        (weather_data["obsTimeLocal"] >= start_date_timestamp)
        & (weather_data["obsTimeLocal"] < end_date_timestamp)
    ].copy()
    day_data_weather.drop(columns=["obsTimeLocal"], inplace=True)
    if day_data_weather.shape[0] > 0:
        binned_data_weather.append(day_data_weather.values)

    day_data_solar = solar_data[
        (solar_data["obsTimeLocal"] >= start_date_timestamp)
        & (solar_data["obsTimeLocal"] < end_date_timestamp)
    ].copy()
    day_data_solar.drop(columns=["obsTimeLocal"], inplace=True)
    if day_data_solar.shape[0] > 0:
        binned_data_solar.append(day_data_solar.values)

    start_date_dt += pd.Timedelta(days=1)

# print size of binned data
print(len(binned_data_weather), len(binned_data_solar))


# implement lstm model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True
        )  # Ensure batch_first is set
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with batch size from input
        batch_size = x.size(0)
        h_0 = torch.zeros(1, batch_size, self.hidden_size)
        c_0 = torch.zeros(1, batch_size, self.hidden_size)

        # Pass data through the LSTM
        out, _ = self.lstm(x, (h_0, c_0))
        # Adjust the following line if your model expects a different output handling
        out = self.fc(out[:, -1, :])  # Assuming you need the last timestep output
        return out


# print the size of the binned data
for i in range(len(binned_data_weather)):
    print(binned_data_weather[i].shape, binned_data_solar[i].shape)

# Create sequences
input_seq = np.array(binned_data_weather)
target_seq = np.array(binned_data_solar)

input_seq = torch.tensor(input_seq, dtype=torch.float32)
target_seq = torch.tensor(target_seq, dtype=torch.float32)

model = LSTMModel(input_size=15, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    for inputs, labels in zip(input_seq, target_seq):
        # Reshape inputs and labels to include batch dimension
        inputs = inputs.view(1, -1, input_seq.shape[-1])  # Add batch dimension
        labels = labels.view(1, -1)

        outputs = model(inputs)
        print(outputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
