import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
import pandas as pd
from lstm import LSTMModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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
    criterion_type="MSE",
    output_type="linear",
    hours=24,
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
        output_type=output_type,
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
            print(date_str)
            inputs = inputs.view(1, -1, input_size)  # Add batch dimension
            labels = labels.view(-1, output_size)  # Add batch dimension
            outputs = model(inputs)
            outputs = outputs.view(-1, output_size)

            # set the x-axis to be the hours
            x = range(hours * 4)

            # Plot the first day of predictions
            plt.title(f"Forecasted Power {date_str}")
            plt.plot(x, labels[:, 0].cpu(), label="Actual")
            plt.plot(x, outputs[:, 0].cpu(), label="Forecasted", linestyle="--")
            plt.xlabel("time (hours)")
            plt.ylabel("solar power (W)")
            plt.legend()

            # calculate the errors
            r2 = r2_score(labels[:, 0].cpu().numpy(), outputs[:, 0].cpu().numpy())
            mae = mean_absolute_error(
                labels[:, 0].cpu().numpy(), outputs[:, 0].cpu().numpy()
            )
            mse = mean_squared_error(
                labels[:, 0].cpu().numpy(), outputs[:, 0].cpu().numpy()
            )
            # determine the midpoint of the y axis
            y_mid_labels = (
                labels[:, 0].cpu().numpy().max() + labels[:, 0].cpu().numpy().min()
            ) / 2

            y_mid_output = (
                outputs[:, 0].cpu().numpy().max() + outputs[:, 0].cpu().numpy().min()
            ) / 2

            y_mid = (y_mid_labels + y_mid_output) / 2

            # add r2, mae, mse to the plot
            plt.text(
                0, y_mid, f"R2: {r2:.2f}\nMAE: {mae:.2f}\nMSE: {mse:.2f}", fontsize=11
            )

            plt.savefig(f"{forecast_plots_directory}/lstm_infer_{date_str}.png")
            plt.close()

            # sum the outputs and labels
            outputs_sum = outputs[:, 0].sum()
            labels_sum = labels[:, 0].sum()
            # print(f"{date_str} Predicted: {outputs_sum:.2f} Actual: {labels_sum:.2f}")

            criterion = nn.MSELoss()
            if criterion_type == "MAE":
                criterion = nn.L1Loss()
            elif criterion_type == "MSE":
                criterion = nn.MSELoss()
            # find the mean squared error
            # criterion = nn.MSELoss()

            # # Assuming outputs and labels are of shape [batch_size, num_features]
            # # and you only care about the first feature (index 0)
            # focused_outputs = outputs[:, 0]
            # focused_labels = labels[:, 0]

            # save focused_outputs and focused_labels to results

            # Now calculate the loss only on the selected feature
            # loss = criterion(focused_outputs, focused_labels)
            # print(f"{date_str} MSE: {loss.item():.4f}")
            # calculate the r2 score

            # add results to array
            results.append(
                {
                    "date": date_str,
                    "predicted": outputs_sum.item(),
                    "actual": labels_sum.item(),
                    "delta": outputs_sum.item() - labels_sum.item(),
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                }
            )
            start_date_dt += pd.Timedelta(hours=hours)
    # save results to json
    metadata = {
        "input_file_path": input_file_path,
        "target_file_path": target_file_path,
        "results": results,
    }
    pd.Series(metadata).to_json(results_save_path)


# main function
if __name__ == "__main__":
    infer_lstm_model(
        input_file_path="lstm_data/input_seq_infer_low_correlation.npy",
        target_file_path="lstm_data/target_seq_infer_low_correlation.npy",
        model_save_path="forecast_results/20240607173904/lstm_model.pth",
        results_save_path="forecast_results/20240607173904/forecast_results_2.json",
        batch_size=1,
        activation="sigmoid",
        hidden_size=256,
        num_layers=8,
        forecast_plots_directory="forecast_results/20240607173904/infer_plots_2",
        start_date="2023-05-01",
        hours=24,
    )
