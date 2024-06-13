#!/bin/bash

# Ensure SOLAR_SITE_ID and SOLAR_API_KEY are set
if [ -z "$SOLAR_SITE_ID" ] || [ -z "$SOLAR_API_KEY" ]; then
    echo "Please set the SOLAR_SITE_ID and SOLAR_API_KEY environment variables."
    exit 1
fi

# Start date as YYYY-MM-DD
start_date="2020-01-01"
# End date as YYYY-MM-DD (exclusive, so set to one day after the last day needed)
end_date="2024-07-01"

# Function to increment month
increment_month() {
    date -I -d "$1 +1 month"
}

# Function to get the last day of the current month
last_day_of_month() {
    date -I -d "$(date -d "$1 +1 month") -1 day"
}

# Loop from start_date to the month before end_date
current_date="$start_date"
while [[ "$current_date" < "$end_date" ]]; do
    # Get the year and month
    year_month=$(date -d "$current_date" "+%Y-%m")
    # Calculate the last day of the current month
    last_day=$(last_day_of_month "$current_date")
    # Formatted filename
    filename="data/power-${year_month}.csv"

    # Construct the URL
    url="https://monitoringapi.solaredge.com/site/$SOLAR_SITE_ID/power?timeUnit=QUARTER_OF_AN_HOUR&startTime=${current_date}%2000:00:00&endTime=${last_day}%2023:59:59&api_key=$SOLAR_API_KEY&format=csv"


    # Use curl to download the data
    curl -s "$url" -o "$filename"

    echo "Data for $current_date has been saved to $filename"

    # Update current_date to the first day of the next month
    current_date=$(increment_month "$current_date")
done

# combine all the csv files into one named solar_power_data.csv and remove the header row
echo date,value > data/solar_power_data.csv
for file in data/power-*.csv; do
    tail -n +2 $file >> data/solar_power_data.csv
done

echo "All data fetched."
