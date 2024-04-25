import pandas as pd
import matplotlib.pyplot as plt
from utils_plt import plot_corr_matrix, plot_scatter, plot_line, plot_3d_scatter

# Load the solar_energy data
weather_solar_data = pd.read_csv("wunderground/data/weather_solar_data.csv")
print(weather_solar_data.head())
weather_solar_data["obsTimeLocal"] = pd.to_datetime(weather_solar_data["obsTimeLocal"])


weather_solar_data["month"] = weather_solar_data["obsTimeLocal"].dt.month
weather_solar_data["day_of_year"] = weather_solar_data["obsTimeLocal"].dt.day_of_year
weather_solar_data.set_index("obsTimeLocal", inplace=True)

# make a correlation matrix
# plot_corr_matrix(weather_solar_data, "corr_matrix.png")

# resample the data into 24-hour intervals, sum the solar energy and power, mean for the rest
# resampled_data = weather_solar_data.resample("D").mean()
# resampled_data["solarEnergy"] = weather_solar_data["solarEnergy"].resample("24H").sum()
# resampled_data["solarPower"] = weather_solar_data["solarPower"].resample("24H").sum()


# plot a scatter chart of the solar power vs tempAvg

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

plot_3d_scatter(
    weather_solar_data,
    "day_of_year",
    "windspeedAvg",
    "solarEnergy",
    "Day of Year",
    "Wind Speed Avg (mph)",
    "Energy (Wh)",
    "Solar Energy vs Wind Speed vs Day of Year",
    f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_windspeed_day_of_year.png",
    flip_x_axes=True,
    flip_y_axes=True,
)

# plot_scatter(
#     weather_solar_data,
#     "precipTotal",
#     "solarEnergy",
#     "Precip Total (in)",
#     "Energy (Wh)",
#     "Solar Energy vs Precip",
#     f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_precip.png",
# )

plot_3d_scatter(
    weather_solar_data,
    "day_of_year",
    "precipTotal",
    "solarEnergy",
    "Day of Year",
    "Precip Total (in)",
    "Energy (Wh)",
    "Solar Energy vs Precip vs Day of Year",
    f"{DIR_SCATTER_PLOT}/scatter_Energy_vs_precip_day_of_year.png",
    flip_x_axes=True,
    flip_y_axes=True,
)

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
