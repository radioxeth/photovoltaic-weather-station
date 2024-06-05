import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
import pandas as pd
from lstm import LSTMModel
from torch.utils.data import TensorDataset, DataLoader


def infer_lstm_model(
    input_file_path="input_seq_infer.npy",
    target_file_path="target_seq_infer.npy",
    model_save_path="lstm_model.pth",
    results_save_path="forecast_results.json",
    batch_size=1,
    activation="sigmoid",
    hidden_size=128,
    num_layers=4,
    forecast_plots_directory="forecast_results",
    start_date="2021-01-01",
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if not os.path.exists(forecast_plots_directory):
        os.makedirs(forecast_plots_directory)

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
    # create a list of dictionaries to store the results
    results = []

    start_date_dt = pd.to_datetime(start_date)
    with torch.no_grad():
        for inputs, labels in train_loader:
            date_str = start_date_dt.strftime("%Y-%m-%d")
            inputs = inputs.view(1, -1, input_size)  # Add batch dimension
            labels = labels.view(-1, output_size)  # Add batch dimension
            outputs = model(inputs)
            outputs = outputs.view(-1, output_size)
            # Plot the first day of predictions
            plt.title(f"Forecasted Power {date_str}")
            plt.plot(labels[:, 0].cpu(), label="Actual")
            plt.plot(outputs[:, 0].cpu(), label="Forecasted", linestyle="--")
            plt.xlabel("time (minutes)")
            plt.ylabel("solar power (W)")
            plt.legend()
            plt.savefig(f"{forecast_plots_directory}/lstm_infer_{date_str}.png")
            plt.close()

            # sum the outputs and labels
            outputs_sum = outputs[:, 0].sum()
            labels_sum = labels[:, 0].sum()
            # print(f"{date_str} Predicted: {outputs_sum:.2f} Actual: {labels_sum:.2f}")

            # find the mean squared error
            criterion = nn.MSELoss()

            # Assuming outputs and labels are of shape [batch_size, num_features]
            # and you only care about the first feature (index 0)
            focused_outputs = outputs[:, 0]
            focused_labels = labels[:, 0]

            # save focused_outputs and focused_labels to results

            # Now calculate the loss only on the selected feature
            loss = criterion(focused_outputs, focused_labels)
            # print(f"{date_str} MSE: {loss.item():.4f}")

            # add results to array
            results.append(
                {
                    "date": date_str,
                    "predicted": outputs_sum.item(),
                    "actual": labels_sum.item(),
                    "delta": outputs_sum.item() - labels_sum.item(),
                    "mse": loss.item(),
                }
            )
            start_date_dt += pd.Timedelta(days=1)
    # save results to json
    metadata = {
        "input_file_path": input_file_path,
        "target_file_path": target_file_path,
        "results": results,
    }
    pd.Series(metadata).to_json(results_save_path)


# main function
if __name__ == "__main__":
    infer_lstm_model()
