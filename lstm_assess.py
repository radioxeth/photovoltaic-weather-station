import os
from lstm_train import train_lstm_model
from lstm_infer import infer_lstm_model
from datetime import datetime

hidden_size = 128
num_layers = 4
activation = "sigmoid"
num_epochs = 100

# timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
directory = f"forecast_results/{timestamp}"

# check if directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

print(f"Results will be saved in {directory}")

input_file_path_train = "input_seq_train.npy"
target_file_path_train = "target_seq_train.npy"
model_save_path = f"{directory}/lstm_model.pth"
metadata_save_path = f"{directory}/metadata.json"

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

input_file_path_infer = "input_seq_infer.npy"
target_file_path_infer = "target_seq_infer.npy"
model_save_path = f"{directory}/lstm_model.pth"

infer_lstm_model(
    input_file_path=input_file_path_infer,
    target_file_path=target_file_path_infer,
    model_save_path=model_save_path,
    batch_size=1,
    activation=activation,
    hidden_size=hidden_size,
    num_layers=num_layers,
)
