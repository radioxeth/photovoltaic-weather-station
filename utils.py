from astral import LocationInfo
from astral import Observer
from astral.sun import sun
import datetime
from astral import zoneinfo
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

solar_columns = [
    "obsTimeLocal",
    "solarPower",
    "solarEnergy",
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
