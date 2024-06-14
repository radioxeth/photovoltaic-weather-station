import pandas as pd
import pandas as pd
import numpy as np
from utils import (
    load_lstm_data,
    load_lstm_data_pca,
    weather_columns,
    solar_columns,
    numerical_columns,
    weather_columns_low_correlation,
)
import os

weather_solar_data = pd.read_csv("wunderground/data/weather_solar_data.csv")

train = False
infer = False
forecast = True
description = "_low_correlation_forecast_linear_interpolation"
n_components = 8
w_columns = weather_columns_low_correlation
pca = False
seasonal = False
train_hours = 1
infer_hours = 24
forecast_hours = 24
d_days = 1

data_dir = "lstm_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

input_seq_train_save_path = f"{data_dir}/input_seq_train{description}.npy"
target_seq_train_save_path = f"{data_dir}/target_seq_train{description}.npy"
input_seq_infer_save_path = f"{data_dir}/input_seq_infer{description}.npy"
target_seq_infer_save_path = f"{data_dir}/target_seq_infer{description}.npy"
input_seq_forecast_save_path = f"{data_dir}/input_seq_forecast{description}.npy"
target_seq_forecast_save_path = f"{data_dir}/target_seq_forecast{description}.npy"

# throw error if files exist
if os.path.exists(input_seq_train_save_path) and train:
    raise ValueError(f"{input_seq_train_save_path} already exists")
if os.path.exists(target_seq_train_save_path) and train:
    raise ValueError(f"{target_seq_train_save_path} already exists")
if os.path.exists(input_seq_infer_save_path) and infer:
    raise ValueError(f"{input_seq_infer_save_path} already exists")
if os.path.exists(target_seq_infer_save_path) and infer:
    raise ValueError(f"{target_seq_infer_save_path} already exists")
if os.path.exists(input_seq_forecast_save_path) and forecast:
    raise ValueError(f"{input_seq_forecast_save_path} already exists")
if os.path.exists(target_seq_forecast_save_path) and forecast:
    raise ValueError(f"{target_seq_forecast_save_path} already exists")

# training data
if train:
    start_date_train = "2020-03-12"
    end_date_train = "2023-05-01"

    if pca:
        binned_data_train = load_lstm_data_pca(
            weather_solar_data,
            start_date_train,
            end_date_train,
            w_columns=w_columns,
            s_columns=solar_columns,
            d_hours=train_hours,
            n_components=n_components,
        )
    else:
        binned_data_train = load_lstm_data(
            weather_solar_data,
            start_date_train,
            end_date_train,
            w_columns=w_columns,
            s_columns=solar_columns,
            d_hours=train_hours,
            d_days=d_days,
            seasonal=seasonal,
        )

    input_seq_train = binned_data_train["input_seq"]
    target_seq_train = binned_data_train["target_seq"]
    np.save(input_seq_train_save_path, input_seq_train)
    np.save(target_seq_train_save_path, target_seq_train)


# inference data
if infer:
    start_date_infer = "2023-05-01"
    end_date_infer = "2024-05-29"

    if pca:
        binned_data_infer = load_lstm_data_pca(
            weather_solar_data,
            start_date_infer,
            end_date_infer,
            w_columns=w_columns,
            s_columns=solar_columns,
            d_hours=infer_hours,
            n_components=n_components,
        )
    else:
        binned_data_infer = load_lstm_data(
            weather_solar_data,
            start_date_infer,
            end_date_infer,
            w_columns=w_columns,
            s_columns=solar_columns,
            d_hours=infer_hours,
            d_days=d_days,
            seasonal=seasonal,
        )

    input_seq_infer = binned_data_infer["input_seq"]
    target_seq_infer = binned_data_infer["target_seq"]
    np.save(input_seq_infer_save_path, input_seq_infer)
    np.save(target_seq_infer_save_path, target_seq_infer)

# forecast data
if forecast:
    forecast_directory = "noaa_forecasts"
    forecast_weather_solar_data = pd.read_csv(
        f"{forecast_directory}/forecasted_weather_solar_data.csv"
    )

    start_date_forecast = "2024-05-29"
    end_date_forecast = "2024-06-08"

    if pca:
        binned_data_forecast = load_lstm_data_pca(
            forecast_weather_solar_data,
            start_date_forecast,
            end_date_forecast,
            w_columns=w_columns,
            s_columns=solar_columns,
            d_hours=forecast_hours,
            n_components=n_components,
        )
    else:
        binned_data_forecast = load_lstm_data(
            forecast_weather_solar_data,
            start_date_forecast,
            end_date_forecast,
            w_columns=w_columns,
            s_columns=solar_columns,
            d_hours=forecast_hours,
            d_days=d_days,
            seasonal=seasonal,
        )
    input_seq_forecast = binned_data_forecast["input_seq"]
    target_seq_forecast = binned_data_forecast["target_seq"]
    np.save(input_seq_forecast_save_path, input_seq_forecast)
    np.save(target_seq_forecast_save_path, target_seq_forecast)
