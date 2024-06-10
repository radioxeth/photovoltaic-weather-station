import pandas as pd
from datetime import datetime
import os

# Wind direction mappings
wind_dirs = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]
wind_dirs_degrees = [
    0,
    22.5,
    45,
    67.5,
    90,
    112.5,
    135,
    157.5,
    180,
    202.5,
    225,
    247.5,
    270,
    292.5,
    315,
    337.5,
]
wind_dir_dict = dict(zip(wind_dirs, wind_dirs_degrees))


def binary_column(data, column, threshold=0):
    data[column] = data[column].apply(lambda x: 1 if x > threshold else 0)
    return data


# Load data
df = pd.read_csv("weather_periods.csv")

# Convert startTime to datetime and extract components
df["obsTimeLocal"] = pd.to_datetime(df["startTime"])
# remove times zone
df["obsTimeLocal"] = df["obsTimeLocal"].dt.tz_localize(None)
df[["month", "day", "hour", "minute"]] = df["obsTimeLocal"].apply(
    lambda dt: pd.Series([dt.month, dt.day, dt.hour, dt.minute])
)

# Reformat the data
df["tempAvg"] = df["temperature"]
df["humidityAvg"] = df["relativeHumidity"]
df["pressureTrend"] = 0
df["winddirAvg"] = df["windDirection"].map(wind_dir_dict)
df["windspeedAvg"] = df["windSpeed"].str.extract("(\d+)")[0].astype(float)
df["precipRate"] = df["precipProbability"].apply(lambda x: 1 if x >= 50 else 0)
# convert celcius to farenheit
df["dewptAvg"] = df["dewpoint"]
# calculate windchill
df["windchillAvg"] = (
    35.74
    + 0.6215 * df["tempAvg"]
    - 35.75 * df["windspeedAvg"] ** 0.16
    + 0.4275 * df["tempAvg"] * df["windspeedAvg"] ** 0.16
)
# calculate heat index
df["heatindexAvg"] = (
    -42.379
    + 2.04901523 * df["tempAvg"]
    + 10.14333127 * df["humidityAvg"]
    - 0.22475541 * df["tempAvg"] * df["humidityAvg"]
    - 6.83783e-3 * df["tempAvg"] ** 2
    - 5.481717e-2 * df["humidityAvg"] ** 2
    + 1.22874e-3 * df["tempAvg"] ** 2 * df["humidityAvg"]
    + 8.5282e-4 * df["tempAvg"] * df["humidityAvg"] ** 2
    - 1.99e-6 * df["tempAvg"] ** 2 * df["humidityAvg"] ** 2
)


# Select required columns
df_final = df[
    [
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
        "month",
        "day",
        "hour",
        "minute",
    ]
]

# Set index for resampling
df_final.set_index("obsTimeLocal", inplace=True)
df_resampled = df_final.resample("15T").ffill()
df_resampled["minute"] = df_resampled.index.minute

# Save to new CSV
# todays date as iso. not time
today = datetime.now().isoformat().split("T")[0]
if not os.path.exists(today):
    os.makedirs(today)
df_resampled.reset_index().to_csv(f"{today}/formatted_weather_data.csv", index=False)
print("Data has been reformatted and saved to 'formatted_weather_data.csv'")
