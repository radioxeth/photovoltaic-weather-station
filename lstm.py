import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from utils import (
    load_lstm_data,
    pca_data,
    set_data_date,
    set_data_date_expanded,
    solar_columns,
    mean_columns,
    date_columns,
)
from scipy.optimize import curve_fit


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
start_date = "2022-01-01"
end_date = "2023-01-01"

weather_solar_data = pd.read_csv("wunderground/data/weather_solar_data.csv")

binned_weather_solar = load_lstm_data(weather_solar_data, start_date, end_date)
weather = binned_weather_solar["weather"]
solar = binned_weather_solar["solar"]

# print size of binned data
print(len(weather), len(solar))


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


# Create sequences
input_seq = np.array(weather)
target_seq = np.array(solar)

input_seq = torch.tensor(input_seq, dtype=torch.float32)
target_seq = torch.tensor(target_seq, dtype=torch.float32)

input_size = input_seq.shape[-1]
output_size = target_seq.shape[-1]

model = LSTMModel(input_size=input_size, hidden_size=64, output_size=output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

print(input_seq.shape[-1], target_seq.shape[-1])

for epoch in range(num_epochs):
    for inputs, labels in zip(input_seq, target_seq):
        # Reshape inputs and labels to include batch dimension
        inputs = inputs.view(1, -1, input_seq.shape[-1])  # Add batch dimension
        # print("inputs", inputs[0][0])
        labels = labels.view(1, -1, target_seq.shape[-1])
        # print("labels", labels)
        outputs = model(inputs)
        # print(outputs.shape, labels.shape)
        # print("outputs", outputs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# # save the model
torch.save(model.state_dict(), "lstm_model.pth")
print("Model saved")

# # determine accuracy of model

# results = model(input_seq[0].view(1, -1, input_seq.shape[-1]))
# print("inputs", input_seq[0].view(1, -1, input_seq.shape[-1]))


# results_all = []
# for i in range(1, len(input_seq)):
#     print(input_seq[i])
#     results = model(input_seq[i].view(i, -1, input_seq.shape[-1]))
#     results_all.append(results)
# print(results_all)
