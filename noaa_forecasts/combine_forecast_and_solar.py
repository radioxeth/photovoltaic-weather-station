import os
import pandas as pd

# for each directory in the noaa_forecasts directory open the csv file and extract the next day of data


def linearize_value(x1, x2, y1, y2, x_prime):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x_prime + b


def windchill(temp, windspeed):
    if temp < 50 and windspeed > 3:
        return (
            35.74
            + 0.6215 * temp
            - 35.75 * windspeed**0.16
            + 0.4275 * temp * windspeed**0.16
        )
    else:
        return temp


def heatindex(temp, humidity):
    if temp >= 80:
        return (
            -42.379
            + 2.04901523 * temp
            + 10.14333127 * humidity
            - 0.22475541 * temp * humidity
            - 6.83783e-3 * temp**2
            - 5.481717e-2 * humidity**2
            + 1.22874e-3 * temp**2 * humidity
            + 8.5282e-4 * temp * humidity**2
            - 1.99e-6 * temp**2 * humidity**2
        )
    else:
        return temp


start_date = "2024-05-27"
end_date = "2024-06-10"

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
        df["dewptAvg"] = df["dewptAvg"] * 1.8 + 32
        # convert temperature in F and windspeed in mph to windchill and heatindex
        df["windchillAvg"] = df.apply(
            lambda x: windchill(x["tempAvg"], x["windspeedAvg"]), axis=1
        )
        df["heatindexAvg"] = df.apply(
            lambda x: heatindex(x["tempAvg"], x["humidityAvg"]), axis=1
        )
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

        final_df = pd.concat([final_df, df])

    else:
        print(f"No data for {current_date}")

    current_date += pd.Timedelta(days=1)


# fill in each 15 minute interval with a linear interpolation of the values for final_df
# not not resample solarPower
final_df["obsTimeLocal"] = pd.to_datetime(final_df["obsTimeLocal"])
final_df.set_index("obsTimeLocal", inplace=True)
# for each weather column interpolate the values
for column in final_df.columns:
    # don't do solarPower, month, hour, minute, day
    if column not in ["solarPower", "month", "hour", "minute", "day"]:
        # for each hour in the day interpolate the values
        start_time = final_df.index.min()
        end_time = final_df.index.max()
        current_time = start_time
        while current_time < end_time:

            if current_time + pd.Timedelta(minutes=60) in final_df.index:
                # interpolate the value
                next_0 = current_time
                next_15 = current_time + pd.Timedelta(minutes=15)
                next_30 = current_time + pd.Timedelta(minutes=30)
                next_45 = current_time + pd.Timedelta(minutes=45)
                next_60 = current_time + pd.Timedelta(minutes=60)

                x1 = 0
                x2 = 60
                y1 = final_df.loc[next_0, column]
                y2 = final_df.loc[next_60, column]
                print(f"Interpolating {column} at {current_time}")
                print(y1, y2)
                next_15_value = linearize_value(x1, x2, y1, y2, 15)
                next_30_value = linearize_value(x1, x2, y1, y2, 30)
                next_45_value = linearize_value(x1, x2, y1, y2, 45)

                final_df.loc[next_15, column] = next_15_value
                final_df.loc[next_30, column] = next_30_value
                final_df.loc[next_45, column] = next_45_value

            current_time += pd.Timedelta(hours=1)

final_df.reset_index(inplace=True)
print(final_df.head())
final_df.to_csv("combined_weather_solar_data.csv", index=False)
