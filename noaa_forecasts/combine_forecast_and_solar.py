import os
import pandas as pd

# for each directory in the noaa_forecasts directory open the csv file and extract the next day of data

start_date = "2024-05-26"
end_date = "2024-06-02"

# as datetime
start_date_dt = pd.to_datetime(start_date)
end_date_dt = pd.to_datetime(end_date)
current_date = start_date_dt

solar_power_data = pd.read_csv("../solar_power/data/solar_power_data.csv")
solar_power_data["date"] = pd.to_datetime(solar_power_data["date"])
# solar_power_data.set_index("date", inplace=True)

final_df = pd.DataFrame()

# new data frame with the columns from the csv file
while current_date < end_date_dt:
    print(current_date.strftime("%Y-%m-%d"))
    next_date = current_date + pd.Timedelta(days=1)
    next_date_end = next_date + pd.Timedelta(days=1)
    # check that the directory exists
    current_forecast_directory = current_date.strftime("%Y-%m-%d")
    current_power_directory = "solar_power/data"
    if os.path.exists(current_forecast_directory):
        # load the csv file
        df = pd.read_csv(f"{current_forecast_directory}/formatted_weather_data.csv")
        df["obsTimeLocal"] = pd.to_datetime(df["obsTimeLocal"])
        # filter to only the data for the next day starting at midnight
        df = df[
            (df["obsTimeLocal"] >= next_date) & (df["obsTimeLocal"] < next_date_end)
        ]
        # print(df.head())

        # load the solar power data for the next day
        solar_power = solar_power_data[
            (solar_power_data["date"] >= next_date)
            & (solar_power_data["date"] < next_date_end)
        ].copy()
        solar_power.fillna(0, inplace=True)

        # append the solar_power value column to the weather data for the date
        for date in df["obsTimeLocal"]:
            solar_power_date = solar_power[solar_power["date"] == date]
            if not solar_power_date.empty:
                df.loc[df["obsTimeLocal"] == date, "solarPower"] = solar_power_date[
                    "value"
                ].values[0]

        # print(solar_power.head())
        # print(df.head())
        final_df = pd.concat([final_df, df])

    else:
        print(f"No data for {current_date}")

    current_date += pd.Timedelta(days=1)

print(final_df.head())
final_df.to_csv("combined_weather_solar_data.csv", index=False)
