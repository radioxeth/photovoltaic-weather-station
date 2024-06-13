from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from utils import pca_data, set_data_date_index, solar_columns
from scipy.optimize import curve_fit

weather_solar_data = pd.read_csv("wunderground/data/weather_solar_data.csv")


# print column names
print(weather_solar_data.columns)
print(weather_solar_data.head())

weather_solar_data = set_data_date_index(weather_solar_data, "obsTimeLocal")

# get one day of weather/solar data
weather_solar_data = weather_solar_data[
    (weather_solar_data.index >= "2022-06-01")
    & (weather_solar_data.index < "2022-06-08")
]

solar_data = weather_solar_data[solar_columns]
## drop solar_columns from weather_solar_data
# weather_solar_data.dropna(inplace=True)
solar_data = weather_solar_data[solar_columns]
solar_data.fillna(0, inplace=True)
weather_solar_data.drop(columns=solar_columns, inplace=True)
weather_solar_data.drop(columns=["month", "day_of_month", "day_of_year"], inplace=True)
# drop index
weather_solar_data.reset_index(drop=True, inplace=True)
# weather_solar_data.drop(columns=["obsTimeLocal"], inplace=True)

print(weather_solar_data.head())
n_components = 17
# PCA the data
X_pca = pca_data(weather_solar_data, n_components=n_components)
print(X_pca.head())

# Assuming X_pca is a DataFrame and its index is the time
time = X_pca.index


def sin_func(x, A, B, C, D):
    return A * np.sin(7 * B * x + C) + D


X_0 = X_pca.iloc[:, 1]

# Estimate initial parameters
p0 = [
    max(X_0) - min(X_0),
    2 * np.pi / (time[-1] - time[0]),
    0,
    np.mean(X_0),
]

# Fit the sine function to the data
params, covariance = curve_fit(sin_func, time, X_0, p0=p0)
params_solar, _ = curve_fit(sin_func, time, solar_data["solarEnergy"], p0=p0)

print("fitted params", params)
plt.scatter(time, X_0, alpha=0.8)
plt.scatter(time, solar_data["solarEnergy"], alpha=0.8, color="green")
# Plot the fitted sine wave
plt.plot(time, sin_func(time, *params), color="red")
plt.plot(time, sin_func(time, *params_solar), color="yellow")
plt.xlabel("time")
plt.ylabel("PCA 0")
plt.title("time vs PCA 0")
plt.savefig("pca_plot/pca_0_vs_time_regression.png")
plt.close()
