import json
from astral import LocationInfo
from astral import Observer
from astral.sun import sun
import datetime
from astral import zoneinfo
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

solar_columns = [
    "obsTimeLocal",
    "solarPower",
    # "solarEnergy",
]
numerical_columns = [
    "obsTimeLocal",
    "winddirAvg",
    "humidityHigh",
    "humidityLow",
    "humidityAvg",
    "tempHigh",
    "tempLow",
    "tempAvg",
    "windspeedHigh",
    "windspeedLow",
    "windspeedAvg",
    "windgustHigh",
    "windgustLow",
    "windgustAvg",
    "dewptHigh",
    "dewptLow",
    "dewptAvg",
    "windchillHigh",
    "windchillLow",
    "windchillAvg",
    "heatindexHigh",
    "heatindexLow",
    "heatindexAvg",
    "pressureMax",
    "pressureMin",
    "pressureTrend",
    "precipRate",
    "precipTotal",
]

mean_columns = [
    "obsTimeLocal",
    "winddirAvg",
    "humidityAvg",
    "tempAvg",
    "windspeedAvg",
    "windgustAvg",
    "dewptAvg",
    "windchillAvg",
    "heatindexAvg",
    "pressureTrend",
]

max_columns = [
    "obsTimeLocal",
    "humidityHigh",
    "tempHigh",
    "windspeedHigh",
    "windgustHigh",
    "dewptHigh",
    "windchillHigh",
    "heatindexHigh",
    "pressureMax",
    "precipRate",
]

min_columns = [
    "obsTimeLocal",
    "humidityLow",
    "tempLow",
    "windspeedLow",
    "windgustLow",
    "dewptLow",
    "windchillLow",
    "heatindexLow",
    "pressureMin",
]

sum_columns = [
    "obsTimeLocal",
    "precipTotal",
]

weather_columns = [
    "obsTimeLocal",
    "tempAvg",
    "humidityAvg",
    "pressureTrend",
    "winddirAvg",
    "windspeedAvg",
    "precipRate",
    "dewptAvg",
    "windchillAvg",
    "heatindexAvg",
]

weather_columns_low_correlation = [
    "obsTimeLocal",
    "tempAvg",
    "humidityAvg",
    "pressureTrend",
    "winddirAvg",
    "windspeedAvg",
    "precipRate",
    "dewptAvg",
]

correlation_plot_columns = [
    "tempAvg",
    "humidityAvg",
    "pressureTrend",
    "winddirAvg",
    "windspeedAvg",
    "precipRate",
    "dewptAvg",
    "windchillAvg",
    "heatindexAvg",
    "solarPower",
    "solarEnergy",
]
time_since_columns = ["timeSinceFreezing", "timeSincePrecip"]


date_columns = ["year", "month", "day", "hour", "minute"]


# use the actual and predicted values to calculate the mean squared error
def calculate_mse(file):
    with open(file) as f:
        data = json.load(f)
    actual = []
    predicted = []
    for i in range(len(data["results"])):
        actual.append(data["results"][i]["actual"])
        predicted.append(data["results"][i]["predicted"])
    mse = mean_squared_error(actual, predicted)

    file_name = data["input_file_path"].split("/")[-1]
    return mse, file_name


def calculate_rmse(file):
    with open(file) as f:
        data = json.load(f)
    actual = []
    predicted = []
    for i in range(len(data["results"])):
        actual.append(data["results"][i]["actual"])
        predicted.append(data["results"][i]["predicted"])
    rmse = mean_squared_error(actual, predicted) ** 0.5

    file_name = data["input_file_path"].split("/")[-1]
    return rmse, file_name


def calculate_mae(file):
    with open(file) as f:
        data = json.load(f)
    actual = []
    predicted = []
    for i in range(len(data["results"])):
        actual.append(data["results"][i]["actual"])
        predicted.append(data["results"][i]["predicted"])
    mae = mean_absolute_error(actual, predicted)

    file_name = data["input_file_path"].split("/")[-1]
    return mae, file_name


def calculate_r2(file):
    with open(file) as f:
        data = json.load(f)
    actual = []
    predicted = []
    for i in range(len(data["results"])):
        actual.append(data["results"][i]["actual"])
        predicted.append(data["results"][i]["predicted"])
    r2 = r2_score(actual, predicted)

    file_name = data["input_file_path"].split("/")[-1]
    return r2, file_name


def get_sunrise_sunset(latitude, longitude, elevation, date_input):
    # Create a location object for the specified city
    city = LocationInfo(
        latitude=latitude,
        longitude=longitude,
        name="Denver",
        region="Colorado",
        timezone="US/Mountain",
    )
    observer = Observer(latitude=latitude, longitude=longitude, elevation=elevation)
    s = sun(observer=observer, date=date_input, tzinfo=city.timezone)

    sunrise = s["sunrise"].strftime("%H:%M:%S")
    sunset = s["sunset"].strftime("%H:%M:%S")
    print(s)

    return sunrise, sunset


def set_data_date_expanded(data, date_col):
    data[date_col] = pd.to_datetime(data[date_col])
    data["year"] = data["obsTimeLocal"].dt.year
    data["month"] = data["obsTimeLocal"].dt.month
    data["day"] = data["obsTimeLocal"].dt.day
    data["hour"] = data["obsTimeLocal"].dt.hour
    data["minute"] = data["obsTimeLocal"].dt.minute
    return data


def set_data_date_expanded_with_seasonality(data, date_col):
    data[date_col] = pd.to_datetime(data[date_col])
    data["year"] = data["obsTimeLocal"].dt.year
    data["month"] = np.sin(2 * np.pi * data["obsTimeLocal"].dt.month / 12)
    data["day"] = np.sin(2 * np.pi * data["obsTimeLocal"].dt.day / 30.4)
    data["hour"] = np.sin(2 * np.pi * data["obsTimeLocal"].dt.hour / 24)
    data["minute"] = np.sin(2 * np.pi * data["obsTimeLocal"].dt.minute / 60)
    return data


def linearize_value(x1, x2, y1, y2, x_prime):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x_prime + b


def set_data_date(data, date_col):
    data[date_col] = pd.to_datetime(data[date_col])
    return data


def pca_data(data, n_components):
    # Initialize and apply imputation
    imputer = SimpleImputer(strategy="mean")
    data_imputed = imputer.fit_transform(data)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(data_imputed)

    # X_pca to df
    X_pca = pd.DataFrame(X_pca, index=data.index)  # Preserve the original index
    return X_pca


def binary_column(data, column, threshold=0):
    data[column] = data[column].apply(lambda x: 1 if x > threshold else 0)
    return data


def normalize_lstm_data_mean(data, columns):
    data = data.copy()
    # Fill NaNs with the mean for each specified column
    for col in columns:
        if data[col].isna().any():  # Check if there are any NaNs in the column
            data[col].fillna(data[col].mean(), inplace=True)

    scaler = StandardScaler()
    for col in columns:
        # Reshape data[col] to (-1, 1) if it's a 1D array, as fit_transform expects 2D input
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    return data


def normalize_lstm_data_zeros(data, columns):
    data = data.copy()
    # Fill NaNs with 0 for each specified column
    data.fillna(0, inplace=True)

    scaler = StandardScaler()
    for col in columns:
        # Reshape data[col] to (-1, 1) if it's a 1D array, as fit_transform expects 2D input
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    return data


def lstm_data_fill_zeros(data):
    data = data.copy()
    # Fill NaNs with 0 for each specified column
    data.fillna(0, inplace=True)
    return data


def load_lstm_data(
    weather_solar_data,
    start_date,
    end_date,
    w_columns=weather_columns,
    s_columns=solar_columns,
    d_hours=1,
    d_days=1,
    seasonal=False,
):
    if seasonal:
        weather_solar_data = set_data_date_expanded_with_seasonality(
            weather_solar_data, "obsTimeLocal"
        )
    else:
        weather_solar_data = set_data_date_expanded(weather_solar_data, "obsTimeLocal")

    weather_solar_data = weather_solar_data[
        (weather_solar_data["obsTimeLocal"] >= start_date)
        & (weather_solar_data["obsTimeLocal"] < end_date)
    ]

    print(weather_solar_data)
    # break the data into 1 day increments
    weather_data = weather_solar_data[w_columns + ["month", "day", "hour", "minute"]]
    solar_data = weather_solar_data[s_columns + ["month", "day", "hour", "minute"]]

    # normalize every column except for obsTimeLocal
    weather_data = normalize_lstm_data_mean(weather_data, w_columns[1:])
    solar_data = lstm_data_fill_zeros(solar_data)

    # break data into groups of 24 hours
    # loop through each day in a year
    binned_data_weather = []
    binned_data_solar = []

    # for each day in between start and end date
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date) - pd.Timedelta(days=d_days)

    print("weather data size:", weather_data.size)
    print("solar data size:", solar_data.size)

    while start_date_dt < end_date_dt:
        start_date_timestamp = start_date_dt
        end_date_timestamp = start_date_dt + pd.Timedelta(days=d_days)
        # get the data for that day
        weather_data_day = weather_data[
            (weather_data["obsTimeLocal"] >= start_date_timestamp)
            & (weather_data["obsTimeLocal"] < end_date_timestamp)
        ].copy()
        solar_data_day = solar_data[
            (solar_data["obsTimeLocal"] >= start_date_timestamp)
            & (solar_data["obsTimeLocal"] < end_date_timestamp)
        ].copy()

        # set obsTimeLocal to float
        if weather_data_day.shape[0] == (96 * d_days):
            weather_data_day.drop(columns=["obsTimeLocal"], inplace=True)
            binned_data_weather.append(weather_data_day.values)

        # set obsTimeLocal to float
        if solar_data_day.shape[0] == (96 * d_days):
            solar_data_day.drop(columns=["obsTimeLocal"], inplace=True)
            binned_data_solar.append(solar_data_day.values)
        # print(start_date_dt)
        start_date_dt += pd.Timedelta(hours=d_hours)

    print("weather columns", weather_data.columns)
    print("solar columns", solar_data.columns)
    print("done binning")
    input_seq = np.array(binned_data_weather)
    target_seq = np.array(binned_data_solar)

    input_seq = torch.tensor(input_seq, dtype=torch.float32)
    target_seq = torch.tensor(target_seq, dtype=torch.float32)

    return {"input_seq": input_seq, "target_seq": target_seq}


def load_lstm_data_pca(
    weather_solar_data,
    start_date,
    end_date,
    w_columns=numerical_columns,
    s_columns=solar_columns,
    d_hours=1,
    n_components=5,
):
    weather_solar_data = set_data_date_expanded(weather_solar_data, "obsTimeLocal")

    weather_solar_data = weather_solar_data[
        (weather_solar_data["obsTimeLocal"] >= start_date)
        & (weather_solar_data["obsTimeLocal"] < end_date)
    ]

    print(weather_solar_data)
    # break the data into 1 day increments
    weather_data = weather_solar_data[
        w_columns + ["year", "month", "day", "hour", "minute"]
    ]
    solar_data = weather_solar_data[s_columns]

    # normalize every column except for obsTimeLocal
    weather_data = normalize_lstm_data_mean(weather_data, w_columns[1:])
    solar_data = lstm_data_fill_zeros(solar_data)

    # # drop obsTimeLocal
    # weather_data.drop(columns=["obsTimeLocal"], inplace=True)

    print("weather data size before pca:", weather_data.size)
    print("solar data size before pca:", solar_data.size)

    # apply pca to weather data, excluding the first column
    weather_data_pca = pca_data(weather_data.drop("obsTimeLocal", axis=1), n_components)
    weather_data_pca["obsTimeLocal"] = weather_solar_data["obsTimeLocal"]
    weather_solar_data_final = weather_data_pca.merge(
        solar_data, on="obsTimeLocal", how="inner"
    )
    print(weather_solar_data_final)
    pca_columns = [i for i in range(n_components)]
    print(pca_columns)
    weather_data = weather_solar_data_final[pca_columns + ["obsTimeLocal"]]
    solar_data = weather_solar_data_final[s_columns]

    # break data into groups of 24 hours
    # loop through each day in a year
    binned_data_weather = []
    binned_data_solar = []

    # for each day in between start and end date
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date) - pd.Timedelta(days=1)

    while start_date_dt < end_date_dt:
        start_date_timestamp = start_date_dt
        end_date_timestamp = start_date_dt + pd.Timedelta(days=1)
        # get the data for that day
        weather_data_day = weather_data[
            (weather_data["obsTimeLocal"] >= start_date_timestamp)
            & (weather_data["obsTimeLocal"] < end_date_timestamp)
        ].copy()
        solar_data_day = solar_data[
            (solar_data["obsTimeLocal"] >= start_date_timestamp)
            & (solar_data["obsTimeLocal"] < end_date_timestamp)
        ].copy()

        # set obsTimeLocal to float
        if weather_data_day.shape[0] == 96:
            weather_data_day.drop(columns=["obsTimeLocal"], inplace=True)
            binned_data_weather.append(weather_data_day.values)

        # set obsTimeLocal to float
        if solar_data_day.shape[0] == 96:
            solar_data_day.drop(columns=["obsTimeLocal"], inplace=True)
            binned_data_solar.append(solar_data_day.values)
        # print(start_date_dt)
        start_date_dt += pd.Timedelta(hours=d_hours)

    print("weather columns", weather_data.columns, len(binned_data_weather))
    print("solar columns", solar_data.columns, len(binned_data_solar))

    print("done binning")
    input_seq = np.array(binned_data_weather)
    target_seq = np.array(binned_data_solar)

    input_seq = torch.tensor(input_seq, dtype=torch.float32)
    target_seq = torch.tensor(target_seq, dtype=torch.float32)

    return {"input_seq": input_seq, "target_seq": target_seq}
