import os
from lstm_infer import infer_lstm_model
from datetime import datetime
import json
from utils import (
    calculate_mean_mae,
    calculate_mean_mse,
    calculate_mean_rmse,
    calculate_mean_r2,
)

test_dir = 20240607173904
description = "_low_correlation_forecast_linear_interpolation"

input_file_path = f"lstm_data/input_seq_forecast{description}.npy"
target_file_path = f"lstm_data/target_seq_forecast{description}.npy"


metadata_file_path = f"forecast_results/{test_dir}/metadata.json"

forecast_save_directory = f"noaa_forecast_results/{test_dir}"
forecast_save_path = f"{forecast_save_directory}/noaa_forecast_results.json"

forecast_plots_directory = f"noaa_forecast_results/{test_dir}/forecast_plots"


with open(metadata_file_path, "r") as f:
    metadata = json.load(f)

start_date = "2024-05-29"
input_size = metadata["input_size"]
output_size = metadata["output_size"]
hidden_size = metadata["hidden_size"]
num_layers = metadata["num_layers"]
activation = metadata["activation"]
batch_size = 1

model_save_path = metadata["model_save_path"]
# if criterion is not specified, default to MSE
if "criterion" not in metadata:
    criterion_type = "MSE"
else:
    criterion_type = metadata["criterion"]

if "output_type" not in metadata:
    output_type = "linear"
else:
    output_type = metadata["output_type"]

print(input_file_path)
print(target_file_path)
# infer the lstm model
infer_lstm_model(
    input_file_path=input_file_path,
    target_file_path=target_file_path,
    model_save_path=model_save_path,
    results_save_path=forecast_save_path,
    batch_size=batch_size,
    activation=activation,
    hidden_size=hidden_size,
    num_layers=num_layers,
    forecast_plots_directory=forecast_plots_directory,
    start_date=start_date,
    criterion_type="MSE",
    output_type=output_type,
    hours=24,
)


# # print mse,mase,rmse,r2
print("MSE: ", calculate_mean_mse(forecast_save_path))
print("MAE: ", calculate_mean_mae(forecast_save_path))
print("RMSE: ", calculate_mean_rmse(forecast_save_path))
print("R2: ", calculate_mean_r2(forecast_save_path))


# forecast_results_directory = "forecast_results"
# noaa_forecast_results_directory = "noaa_forecast_results"
# for directory in os.listdir(forecast_results_directory):
#     print(directory)
#     if directory == "20240608122239":
#         hours = 72
#     elif directory == "20240607113540":
#         continue
#     else:
#         hours = 24

#     start_date = "2024-05-29"
#     # gather model information from metadata.json
#     with open(f"{forecast_results_directory}/{directory}/metadata.json", "r") as f:
#         metadata = json.load(f)
#     input_size = metadata["input_size"]
#     output_size = metadata["output_size"]
#     hidden_size = metadata["hidden_size"]
#     num_layers = metadata["num_layers"]
#     activation = metadata["activation"]
#     batch_size = 1
#     input_file_path = metadata["input_file_path"].replace(
#         "input_seq_train", "input_seq_forecast"
#     )
#     target_file_path = metadata["target_file_path"].replace(
#         "target_seq_train", "target_seq_forecast"
#     )
#     model_save_path = metadata["model_save_path"]
#     # if criterion is not specified, default to MSE
#     if "criterion" not in metadata:
#         criterion_type = "MSE"
#     else:
#         criterion_type = metadata["criterion"]

#     if "output_type" not in metadata:
#         output_type = "linear"
#     else:
#         output_type = metadata["output_type"]

#     print(input_file_path)
#     print(target_file_path)
#     # infer the lstm model
#     infer_lstm_model(
#         input_file_path=input_file_path,
#         target_file_path=target_file_path,
#         model_save_path=model_save_path,
#         results_save_path=f"{forecast_results_directory}/{directory}/noaa_forecast_results.json",
#         batch_size=batch_size,
#         activation=activation,
#         hidden_size=hidden_size,
#         num_layers=num_layers,
#         forecast_plots_directory=f"{forecast_results_directory}/{directory}/forecast_plots_2",
#         start_date=start_date,
#         criterion_type=criterion_type,
#         output_type=output_type,
#         hours=hours,
#     )
