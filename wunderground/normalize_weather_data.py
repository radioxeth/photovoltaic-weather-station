import os
import pandas as pd

# Load all the data files
# data_files = os.listdir("data")
# dfs = []
# for file in data_files:
#     if file.startswith("wu-"):
#         df = pd.read_csv(f"data/{file}")
#         dfs.append(df)

# # Combine all the data files into a single DataFrame
# combined_df = pd.concat(dfs)
# combined_df.to_csv("data/weather_data.csv", index=False)

agg_columns = {
    "winddirAvg": "mean",
    "humidityAvg": "mean",
    "tempAvg": "mean",
    "windspeedAvg": "mean",
    "windgustAvg": "mean",
    "dewptAvg": "mean",
    "windchillAvg": "mean",
    "heatindexAvg": "mean",
    "pressureTrend": "mean",
    "solarRadiationHigh": "max",
    "uvHigh": "max",
    "humidityHigh": "max",
    "tempHigh": "max",
    "windspeedHigh": "max",
    "windgustHigh": "max",
    "dewptHigh": "max",
    "windchillHigh": "max",
    "heatindexHigh": "max",
    "pressureMax": "max",
    "precipRate": "max",
    "humidityLow": "min",
    "tempLow": "min",
    "windspeedLow": "min",
    "windgustLow": "min",
    "dewptLow": "min",
    "windchillLow": "min",
    "heatindexLow": "min",
    "pressureMin": "min",
    "precipTotal": "sum",
}

# Load the weather_data
weather_data = pd.read_csv("data/weather_data.csv")
print(weather_data.head())
print(weather_data.columns)

## group the data by obsTimeLocal into 15 minute intervals taking the average of numerical columns
# filter out the columns that are not numerical

# weather_data = weather_data[avg_columns]
# Resample the data into 15-minute intervals starting on the hour
weather_data["obsTimeLocal"] = pd.to_datetime(weather_data["obsTimeLocal"])
weather_data.set_index("obsTimeLocal", inplace=True)

resampled_data = weather_data.resample("15T").agg(agg_columns)

# set precipRate to binary column based on threshold of 0
resampled_data["precipRate"] = resampled_data["precipRate"].apply(
    lambda x: 1 if x > 0 else 0
)


print(resampled_data.head())
# write to a new csv file
resampled_data.to_csv("data/weather_data_resampled.csv")

# combine solar and power data into weather data
# Load the solar and power data
solar_power_data = pd.read_csv("../solar_power/data/solar_power_data.csv")
solar_power_data["date"] = pd.to_datetime(solar_power_data["date"])
solar_power_data.set_index("date", inplace=True)

solar_energy_data = pd.read_csv("../solar_energy/data/solar_energy_data.csv")
solar_energy_data["date"] = pd.to_datetime(solar_energy_data["date"])
solar_energy_data.set_index("date", inplace=True)

print(solar_power_data.head())

# join the solar and power data
for date in resampled_data.index:
    solar_power = solar_power_data.loc[solar_power_data.index == date]
    solar_energy = solar_energy_data.loc[solar_energy_data.index == date]
    if not solar_power.empty:
        resampled_data.loc[date, "solarPower"] = solar_power["value"].values[0]
    if not solar_energy.empty:
        resampled_data.loc[date, "solarEnergy"] = solar_energy["value"].values[0]
    print(f"Processed {date}")

resampled_data.drop(columns=["solarRadiationHigh", "uvHigh"], inplace=True)

# fill missing values with 0
# resampled_data.fillna(0, inplace=True)
resampled_data.to_csv("data/weather_solar_data.csv")
