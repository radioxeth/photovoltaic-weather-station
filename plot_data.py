import pandas as pd
from utils_plt import (
    plot_3d_scatter_no_index,
    plot_corr_matrix,
    plot_scatter,
    plot_line,
    plot_3d_scatter,
)
from utils import mean_columns, pca_data, solar_columns, correlation_plot_columns


# Load the solar_energy data
weather_solar_data = pd.read_csv("wunderground/data/weather_solar_data.csv")
weather_solar_data.set_index("obsTimeLocal", inplace=True)
print(weather_solar_data.head())

# combine solar and mean columns
corr_data = weather_solar_data[correlation_plot_columns]

# weather_solar_data = set_data_date_index(weather_solar_data, "obsTimeLocal")
## remove NA columns from weather solar data

# make a correlation matrix
plot_corr_matrix(corr_data, "corr_matrix.png")


DIR_SCATTER_PLOT = "scatter_plots"
# plot_scatter(
#     weather_solar_data,
#     "tempAvg",
#     "solarEnergy",
#     "Temp Avg (F)",
#     "Energy (Wh)",
#     "Solar Energy vs Temp",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_temp.png",
# )

# plot_3d_scatter(
#     weather_solar_data,
#     "month",
#     "tempAvg",
#     "solarEnergy",
#     "Month",
#     "Temp Avg (F)",
#     "Energy (Wh)",
#     "Solar Energy vs Temp vs Month",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_temp_month.png",
# )

# plot_3d_scatter(
#     weather_solar_data,
#     "month",
#     "tempAvg",
#     "solarEnergy",
#     "Month",
#     "Temp Avg (F)",
#     "Energy (Wh)",
#     "Solar Energy vs Temp vs Month",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_temp_month_c_wind_dir.png",
#     "winddirAvg",
#     flip_y_axes=True,
# )

# plot_scatter(
#     weather_solar_data,
#     "winddirAvg",
#     "solarEnergy",
#     "Wind Dir Avg (deg)",
#     "Energy (Wh)",
#     "Solar Energy vs Wind Dir",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_winddir.png",
# )

# plot_3d_scatter(
#     weather_solar_data,
#     "month",
#     "winddirAvg",
#     "solarEnergy",
#     "Month",
#     "Wind Dir Avg (deg)",
#     "Energy (Wh)",
#     "Solar Energy vs Wind Dir vs Month",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_winddir_month.png",
# )

# plot_scatter(
#     weather_solar_data,
#     "humidityAvg",
#     "solarEnergy",
#     "Humidity Avg (%)",
#     "Energy (Wh)",
#     "Solar Energy vs Humidity",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_humidity.png",
# )

# plot_3d_scatter(
#     weather_solar_data,
#     "month",
#     "humidityAvg",
#     "solarEnergy",
#     "Month",
#     "Humidity Avg (%)",
#     "Energy (Wh)",
#     "Solar Energy vs Humidity vs Month",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_humidity_month.png",
# )

# plot_scatter(
#     weather_solar_data,
#     "windspeedAvg",
#     "solarEnergy",
#     "Wind Speed Avg (mph)",
#     "Energy (Wh)",
#     "Solar Energy vs Wind Speed",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_windspeed.png",
# )

# plot_3d_scatter(
#     weather_solar_data,
#     "day_of_year",
#     "windspeedAvg",
#     "solarEnergy",
#     "Day of Year",
#     "Wind Speed Avg (mph)",
#     "Energy (Wh)",
#     "Solar Energy vs Wind Speed vs Day of Year",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_windspeed_day_of_year.png",
#     flip_x_axes=True,
#     flip_y_axes=True,
# )

# plot_scatter(
#     weather_solar_data,
#     "precipTotal",
#     "solarEnergy",
#     "Precip Total (in)",
#     "Energy (Wh)",
#     "Solar Energy vs Precip",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_precip.png",
# )

# plot_3d_scatter(
#     weather_solar_data,
#     "day_of_year",
#     "precipTotal",
#     "solarEnergy",
#     "Day of Year",
#     "Precip Total (in)",
#     "Energy (Wh)",
#     "Solar Energy vs Precip vs Day of Year",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_precip_day_of_year.png",
#     flip_x_axes=True,
#     flip_y_axes=True,
# )

# plot_3d_scatter(
#     weather_solar_data,
#     "day_of_year",
#     "dewptAvg",
#     "solarEnergy",
#     "Day of Year",
#     "Dew Point Avg (F)",
#     "Energy (Wh)",
#     "Solar Energy vs Dew Point vs Day of Year",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_dewpt_day_of_year.png",
#     color_by="day_of_year",
#     flip_x_axes=True,
#     flip_y_axes=False,
# )

# plot_scatter(
#     weather_solar_data,
#     "precipRate",
#     "solarEnergy",
#     "Precip Rate (in/hr)",
#     "Energy (Wh)",
#     "Solar Energy vs Precip Rate",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_precip_rate.png",
# )

# plot_3d_scatter(
#     weather_solar_data,
#     "month",
#     "precipRate",
#     "solarEnergy",
#     "Month",
#     "Precip Rate (in/hr)",
#     "Energy (Wh)",
#     "Solar Energy vs Precip Rate vs Month",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_precip_rate_month.png",
# )

# plot_scatter(
#     weather_solar_data,
#     "pressureTrend",
#     "solarEnergy",
#     "Pressure Trend",
#     "Energy (Wh)",
#     "Solar Energy vs Pressure Trend",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_pressure.png",
# )

# plot_3d_scatter(
#     weather_solar_data,
#     "month",
#     "pressureTrend",
#     "solarEnergy",
#     "Month",
#     "Pressure Trend",
#     "Energy (Wh)",
#     "Solar Energy vs Pressure Trend vs Month",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_pressure_month.png",
# )


# # get only june data
# start_date = "2023-06-01"
# end_date = "2023-06-01"
# weather_solar_data = weather_solar_data.loc[start_date:end_date]
# weather_solar_data.reset_index(inplace=True)

# # plot temAvg and solar_energy vs obsTimeLocal
# plt.plot(
#     weather_solar_data["obsTimeLocal"],
#     weather_solar_data["solar_energy"] / 10,
#     label="solar_energy (dWh)",
# )

# plt.plot(
#     weather_solar_data["obsTimeLocal"],
#     weather_solar_data["humidityAvg"],
#     label="humidityAvg",
# )

# plt.plot(
#     weather_solar_data["obsTimeLocal"],
#     weather_solar_data["tempAvg"],
#     label="tempAvg",
# )
# # rotate x-axis labels
# plt.xticks(rotation=45)
# plt.legend()
# plt.xlabel("Date")
# plt.ylabel("Solar Energy/Temp Avg")
# plt.title("Solar Energy Time Series")
# plt.savefig("plot.png")
# plt.close()

# print(weather_solar_data.shape)


# # calculate the PCA
# X_pca = pca_data(weather_solar_data, 17)
# print(X_pca.shape)
