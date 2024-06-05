import os
from lstm_train import train_lstm_model
from lstm_infer import infer_lstm_model
from datetime import datetime

hidden_size = 16
num_layers = 2
activation = "sigmoid"
num_epochs = 500

# timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
directory = f"forecast_results/{timestamp}"
infer_plots_directory = f"{directory}/infer_plots"
forecast_plots_directory = f"{directory}/forecast_plots"
data_dir = "lstm_data"

description = "_without_time_since"

# check if directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists(infer_plots_directory):
    os.makedirs(infer_plots_directory)

if not os.path.exists(forecast_plots_directory):
    os.makedirs(forecast_plots_directory)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(f"Results will be saved in {directory}")

input_file_path_train = f"{data_dir}/input_seq_train{description}.npy"
target_file_path_train = f"{data_dir}/target_seq_train{description}.npy"
model_save_path = f"{directory}/lstm_model.pth"
metadata_save_path = f"{directory}/metadata.json"
results_save_path = f"{directory}/forecast_results.json"

train_lstm_model(
    input_file_path=input_file_path_train,
    target_file_path=target_file_path_train,
    model_save_path=model_save_path,
    metadata_save_path=metadata_save_path,
    batch_size=32,
    activation=activation,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_epochs=num_epochs,
)

input_file_path_infer = f"{data_dir}/input_seq_infer{description}.npy"
target_file_path_infer = f"{data_dir}/target_seq_infer{description}.npy"
model_save_path = f"{directory}/lstm_model.pth"

infer_lstm_model(
    input_file_path=input_file_path_infer,
    target_file_path=target_file_path_infer,
    model_save_path=model_save_path,
    results_save_path=results_save_path,
    batch_size=1,
    activation=activation,
    hidden_size=hidden_size,
    num_layers=num_layers,
    forecast_plots_directory=infer_plots_directory,
    start_date="2023-05-01",
)

input_file_path_forecast = f"{data_dir}/input_seq_forecast{description}.npy"
target_file_path_forecast = f"{data_dir}/target_seq_forecast{description}.npy"
model_save_path = f"{directory}/lstm_model.pth"

infer_lstm_model(
    input_file_path=input_file_path_forecast,
    target_file_path=target_file_path_forecast,
    model_save_path=model_save_path,
    batch_size=1,
    activation=activation,
    hidden_size=hidden_size,
    num_layers=num_layers,
    forecast_plots_directory=forecast_plots_directory,
    start_date="2024-05-29",
)
