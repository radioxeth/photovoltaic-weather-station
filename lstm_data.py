import pandas as pd
import pandas as pd
import numpy as np
from utils import (
    load_lstm_data,
    weather_columns,
    solar_columns,
)

weather_solar_data = pd.read_csv("wunderground/data/weather_solar_data.csv")

# training data
start_date_train = "2020-03-12"
end_date_train = "2023-04-01"

binned_data_train = load_lstm_data(
    weather_solar_data,
    start_date_train,
    end_date_train,
    weather_columns,
    solar_columns,
    1,
)

input_seq_train = binned_data_train["input_seq"]
target_seq_train = binned_data_train["target_seq"]
np.save("input_seq_train.npy", input_seq_train)
np.save("target_seq_train.npy", target_seq_train)

# inference data
start_date_infer = "2023-04-01"
end_date_infer = "2024-04-01"

binned_data_infer = load_lstm_data(
    weather_solar_data,
    start_date_infer,
    end_date_infer,
    weather_columns,
    solar_columns,
    24,
)

input_seq_infer = binned_data_infer["input_seq"]
target_seq_infer = binned_data_infer["target_seq"]
np.save("input_seq_infer.npy", input_seq_infer)
np.save("target_seq_infer.npy", target_seq_infer)
