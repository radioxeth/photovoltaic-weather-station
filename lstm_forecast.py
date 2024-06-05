import os
from lstm_infer import infer_lstm_model
from datetime import datetime

hidden_size = 128
num_layers = 4
activation = "sigmoid"
num_epochs = 100

# timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
directory = f"forecast_results/{timestamp}"
forecast_plots_directory = f"{directory}/forecast_plots"

# check if directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists(forecast_plots_directory):
    os.makedirs(forecast_plots_directory)


print(f"Results will be saved in {directory}")

input_file_path_infer = "input_seq_forecast.npy"
target_file_path_infer = "target_seq_forecast.npy"
model_save_path = "forecast_results/20240527122939/lstm_model.pth"

infer_lstm_model(
    input_file_path=input_file_path_infer,
    target_file_path=target_file_path_infer,
    model_save_path=model_save_path,
    batch_size=1,
    activation=activation,
    hidden_size=hidden_size,
    num_layers=num_layers,
    forecast_plots_directory=forecast_plots_directory,
    start_date="2024-05-29",
)
