## a script to calculate the variance of the principal components of the data
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import matplotlib.pyplot as plt

from utils import (
    lstm_data_fill_zeros,
    normalize_lstm_data_mean,
    numerical_columns,
    set_data_date_expanded,
)

data_dir = "pca_plot"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

weather_solar_data = pd.read_csv("wunderground/data/weather_solar_data.csv")
# weather_solar_data = set_data_date_expanded(weather_solar_data, "obsTimeLocal")

weather_data = weather_solar_data[numerical_columns]
weather_data = normalize_lstm_data_mean(weather_data, numerical_columns[1:])

pca = PCA()
pca.fit(weather_data.drop(columns=["obsTimeLocal"], axis=1))

variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance)

print("Variance of the principal components:")
print(variance)
print("Cumulative variance of the principal components:")
print(cumulative_variance)

# plot cumulative variance vs number of components

max_components = 11
plt.plot(cumulative_variance[:max_components])
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.title("Cumulative Variance vs Number of Components")
# x-axis ticks every 1
plt.xticks(np.arange(0, max_components, step=1))
plt.grid()
plt.savefig(f"{data_dir}/cumulative_variance.png")
