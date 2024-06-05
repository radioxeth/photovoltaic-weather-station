import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from lstm import LSTMModel
from torch.utils.data import TensorDataset, DataLoader


# main function to train the lstm model
def train_lstm_model(
    input_file_path="input_seq_train.npy",
    target_file_path="target_seq_train.npy",
    model_save_path="lstm_model.pth",
    metadata_save_path="metadata.json",
    num_epochs=100,
    batch_size=32,
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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size = batch_size  # You can adjust the batch size

    train_data = TensorDataset(input_seq, target_seq)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    # Training loop with batches
    model.train()
    epoch_loss = []
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        epoch_loss.append(loss.item())
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_save_path)
    print("Model saved")

    # save the metadata
    metadata = {
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "activation": activation,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "epoch_loss": epoch_loss,
        "model_save_path": model_save_path,
        "input_file_path": input_file_path,
        "target_file_path": target_file_path,
    }
    pd.Series(metadata).to_json(metadata_save_path)
    print("Metadata saved")


# train_lstm_model()

# main function to train the lstm model
if __name__ == "__main__":
    train_lstm_model()
