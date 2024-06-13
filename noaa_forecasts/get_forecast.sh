#!/bin/bash

# URL to fetch data from
url="https://api.weather.gov/gridpoints/BOU/60,60/forecast/hourly"

# Use curl to fetch data from API and jq to extract the forecast URL
# forecast_url=$(curl -s $url | jq -r '.properties.forecast')

# Use curl to fetch forecast details and jq to parse JSON and extract periods
forecast_response=$(curl -s $url)

# Define CSV file name
csv_file="weather_periods.csv"

# Write header to CSV
echo "number,name,startTime,endTime,isDaytime,temperature,temperatureUnit,temperatureTrend,precipProbability,dewpoint,relativeHumidity,windSpeed,windDirection,icon,shortForecast,detailedForecast" > $csv_file

# Parse JSON and write data to CSV
echo "$forecast_response" | jq -r '.properties.periods[] | [ .number, .name, .startTime, .endTime, .isDaytime, .temperature, .temperatureUnit, .temperatureTrend, .probabilityOfPrecipitation.value, .dewpoint.value, .relativeHumidity.value, .windSpeed, .windDirection, .icon, .shortForecast, .detailedForecast ] | @csv' >> $csv_file

python3 format_forecast.py