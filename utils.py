from astral import LocationInfo
from astral import Observer
from astral.sun import sun
import datetime
from astral import zoneinfo
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

solar_columns = [
    "obsTimeLocal",
    "solarPower",
    # "solarEnergy",
]
numerical_columns = [
    "obsTimeLocal",
    "solarRadiationHigh",
    "uvHigh",
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

date_columns = ["year", "month", "day", "hour", "minute", "second"]


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
    data["year"] = data["obsTimeLocal"].dt.day
    data["month"] = data["obsTimeLocal"].dt.month
    data["day"] = data["obsTimeLocal"].dt.day
    data["hour"] = data["obsTimeLocal"].dt.hour
    data["minute"] = data["obsTimeLocal"].dt.minute
    data["second"] = data["obsTimeLocal"].dt.second
    return data


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
    X_pca = pd.DataFrame(X_pca)
    return X_pca


def normalize_lstm_data(data, columns):
    data.fillna(0, inplace=True)
    scaler = MinMaxScaler()
    for col in columns:
        data[col] = scaler.fit_transform(data[[col]])
    return data


def load_lstm_data(weather_solar_data, start_date, end_date):
    weather_solar_data = set_data_date_expanded(weather_solar_data, "obsTimeLocal")

    weather_solar_data = weather_solar_data[
        (weather_solar_data["obsTimeLocal"] >= start_date)
        & (weather_solar_data["obsTimeLocal"] < end_date)
    ]
    # break the data into 1 day increments
    weather_data = weather_solar_data[mean_columns]
    solar_data = weather_solar_data[solar_columns]

    # normalize every column except for obsTimeLocal
    weather_data = normalize_lstm_data(weather_data, mean_columns[1:])
    solar_data = normalize_lstm_data(solar_data, solar_columns[1:])

    # break data into groups of 24 hours
    # loop through each day in a year
    binned_data_weather = []
    binned_data_solar = []

    # for each day in between start and end date

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    while start_date_dt < end_date_dt:
        start_date_timestamp = start_date_dt
        end_date_timestamp = start_date_dt + pd.Timedelta(days=1)
        # get the data for that day
        day_data_weather = weather_data[
            (weather_data["obsTimeLocal"] >= start_date_timestamp)
            & (weather_data["obsTimeLocal"] < end_date_timestamp)
        ].copy()
        day_data_solar = solar_data[
            (solar_data["obsTimeLocal"] >= start_date_timestamp)
            & (solar_data["obsTimeLocal"] < end_date_timestamp)
        ].copy()

        # set obsTimeLocal to float
        day_data_weather["obsTimeLocal"] = day_data_weather["obsTimeLocal"].astype(int)
        if day_data_weather.shape[0] > 0:
            binned_data_weather.append(day_data_weather.values)

        # set obsTimeLocal to float
        day_data_solar["obsTimeLocal"] = day_data_solar["obsTimeLocal"].astype(int)
        if day_data_solar.shape[0] > 0:
            day_data_solar.drop(columns=["obsTimeLocal"], inplace=True)
            binned_data_solar.append(day_data_solar.values)

        start_date_dt += pd.Timedelta(days=1)
    print(len(binned_data_weather))
    print(len(binned_data_solar))
    return {"weather": binned_data_weather, "solar": binned_data_solar}
