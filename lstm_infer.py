import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import pandas as pd
from lstm import LSTMModel
from utils import load_lstm_data, weather_columns, solar_columns
from torch.utils.data import TensorDataset, DataLoader


def infer_lstm_model(
    input_file_path="input_seq_infer.npy",
    target_file_path="target_seq_infer.npy",
    model_save_path="lstm_model.pth",
    results_save_path="forecast_results.npy",
    batch_size=1,
    activation="sigmoid",
    hidden_size=128,
    num_layers=4,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    input_seq = np.load(input_file_path)
    target_seq = np.load(target_file_path)

    input_seq = torch.tensor(input_seq, dtype=torch.float32).to(device)
    target_seq = torch.tensor(target_seq, dtype=torch.float32).to(device)

    print("input_seq:", input_seq.shape)
    print("target_seq:", target_seq.shape)

    input_size = input_seq.shape[-1]
    output_size = target_seq.shape[-1]

    print("input_size:", input_size)
    print("output_size:", output_size)

    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        activation=activation,
    ).to(device)
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    train_data = TensorDataset(input_seq, target_seq)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)

    day = 0
    # create a 2D array to store the results
    # results = np.zeros((input_seq.shape[0], input_seq.shape[1]))
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.view(1, -1, input_size)  # Add batch dimension
            labels = labels.view(-1, output_size)  # Add batch dimension
            outputs = model(inputs)
            outputs = outputs.view(-1, output_size)
            # print("outputs:", outputs.shape)
            # print("labels:", labels.shape)

            # Plot the first day of predictions
            plt.title(f"Day {day}")
            plt.plot(labels[:, 0].cpu(), label="Actual")
            plt.plot(outputs[:, 0].cpu(), label="Forecasted", linestyle="--")
            plt.xlabel("time (minutes)")
            plt.ylabel("solar power (W)")
            plt.legend()
            plt.savefig(f"forecast_plots/lstm_infer_{day}.png")
            plt.close()

            # sum the outputs and labels
            outputs_sum = outputs[:, 0].sum()
            labels_sum = labels[:, 0].sum()
            print(f"Day {day} Predicted: {outputs_sum:.2f} Actual: {labels_sum:.2f}")

            # find the mean squared error
            criterion = nn.MSELoss()

            # Assuming outputs and labels are of shape [batch_size, num_features]
            # and you only care about the first feature (index 0)
            focused_outputs = outputs[:, 0]
            focused_labels = labels[:, 0]

            # save focused_outputs and focused_labels to results

            # Now calculate the loss only on the selected feature
            loss = criterion(focused_outputs, focused_labels)
            print(f"Day {day} MSE: {loss.item():.4f}")

            # find the mse comparing outputs[:,0] and labels[:,0]
            # mse = ((outputs[:, 0] - labels[:, 0]) ** 2).mean()
            # print(f"Day {day} MSE 2: {mse:.4f}")

            # find the accuracy of the model
            # accuracy = outputs_sum / labels_sum
            # accuracy = 1 - loss.item()[0] / labels_sum
            # print(f"Day {day} Accuracy: {accuracy:.4f}")
            day += 1


infer_lstm_model()
