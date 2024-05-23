import pandas as pd
from datetime import datetime

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

# Load data
df = pd.read_csv("weather_periods.csv")


# Function to extract date components
def extract_date_components(date_str):
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    return dt.month, dt.day, dt.hour, dt.minute


# Convert startTime to datetime and extract components
df["obsTimeLocal"] = pd.to_datetime(df["startTime"])
df[["month", "day", "hour", "minute"]] = df["obsTimeLocal"].apply(
    lambda dt: pd.Series([dt.month, dt.day, dt.hour, dt.minute])
)

# Reformat the data
df["tempAvg"] = df["temperature"]
df["humidityAvg"] = df["relativeHumidity"]
df["pressureTrend"] = df["temperatureTrend"]
df["winddirAvg"] = df["windDirection"].map(wind_dir_dict)
df["windspeedAvg"] = df["windSpeed"].str.extract("(\d+)")[0].astype(float)
df["precipRate"] = df["precipProbability"]
df["dewptAvg"] = df["dewpoint"]
df["windchillAvg"] = df["temperature"]  # Placeholder
df["heatindexAvg"] = df["temperature"]  # Placeholder

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

# Save to new CSV
df_resampled.reset_index().to_csv("formatted_weather_data.csv", index=False)
print("Data has been reformatted and saved to 'formatted_weather_data.csv'")
