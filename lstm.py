import torch
import torch.nn as nn
import pandas as pd


# Implement LSTM model
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        activation="relu",
        output_type="linear",
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_type = output_type
        self.fc = nn.Linear(
            hidden_size, output_size
        )  # Maps each time step output to the desired output size
        self.activation_name = activation
        self.activation = self._get_activation(activation)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)  # Apply the linear layer to each time step output

        if self.output_type == "sinusoidal":
            out = torch.sin(out)  # Apply sinusoidal transformation
        return out

    def _get_activation(self, name):
        """Retrieve activation function by name."""
        if name == "relu":
            return nn.ReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "none":  # Use 'none' for linear activation
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {name}")


# def initiate_lstm_model(input_size, hidden_size, output_size, num_layers, activation, output_type="linear"):
#     model = LSTMModel(
#         input_size=input_size,
#         hidden_size=hidden_size,
#         output_size=output_size,
#         num_layers=num_layers,
#         activation=activation,
#         output_type=output_type
#     )
#     return model

# Example usage:
# model = initiate_lstm_model(10, 50, 1, 2, "relu", "sinusoidal")
