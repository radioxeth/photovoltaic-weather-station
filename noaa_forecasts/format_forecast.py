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
df["dewptAvg"] = df["dewpoint"]
df["windchillAvg"] = df["temperature"]  # Placeholder
df["heatindexAvg"] = df["temperature"]  # Placeholder

# # add a column to determine how much time has passed since precipRate > 0 and temperature < 32
# df["timeSincePrecip"] = 0
# df["timeSinceFreezing"] = 0
# time_since_precip = 0
# time_since_freezing = 0
# for date in df.index:
#     if df.loc[date, "precipRate"] > 0:
#         time_since_precip = 0
#     else:
#         time_since_precip += 1
#     if df.loc[date, "tempAvg"] < 32:
#         time_since_freezing = 0
#     else:
#         time_since_freezing += 1
#     df.loc[date, "timeSincePrecip"] = time_since_precip
#     df.loc[date, "timeSinceFreezing"] = time_since_freezing


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
