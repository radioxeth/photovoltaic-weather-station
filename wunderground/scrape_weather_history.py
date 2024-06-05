import os
import requests
import datetime
import pandas as pd

# get api key from bash profile
api_key = os.environ["WU_API_KEY"]
site_id = os.environ["WU_SITE_ID"]

# check if data directory exists, if not create it
if not os.path.exists("data"):
    os.makedirs("data")


# Function to fetch weather data
def fetch_weather_data(station_id, api_key, date):
    base_url = (
        "https://api.weather.com/v2/pws/history/all"  # This URL might be different
    )
    params = {
        "stationId": station_id,
        "format": "json",
        "units": "e",  # for imperial units; use "m" for metric
        "apiKey": api_key,
        "date": date,
    }
    response = requests.get(base_url, params=params)
    # print(response.url)
    if response.status_code == 200:
        return response.json()
    else:
        print(response)
        print(f"Failed to fetch data for {date}")
        return None


# Fetch the weather data for every day from 2020-01-01 to 2023-12-31

# use a date library
start_date = datetime.datetime(2024, 5, 1)
end_date = datetime.datetime(2024, 6, 2)
current_date = start_date
while current_date <= end_date:

    date = current_date.strftime("%Y%m%d")
    weather_data = fetch_weather_data(site_id, api_key, date)
    if weather_data and "observations" in weather_data:
        # write weather data to csv file in in the current month and year as data/wu-{year}-{month}.csv
        # save the data to a csv file, do not overwrite the file if it already exists
        # Flatten the JSON by merging 'imperial' with the rest of the data
        observations = weather_data["observations"]
        ## flatten the data for each row in observations
        flat_data = []
        for obs in observations:
            row = {**obs, **obs["imperial"]}
            del row["imperial"]
            flat_data.append(row)
        df = pd.DataFrame(flat_data)

        # check if csv exists
        filename = f"data/wu-{current_date.year}-{str(current_date.month).zfill(2)}.csv"
        if os.path.exists(filename):
            # append to existing csv
            df.to_csv(
                filename,
                mode="a",
                header=False,
            )
        else:
            df.to_csv(filename)

        print(f"Saved data for {current_date}")

    current_date += datetime.timedelta(days=1)


### Combine all the data into a single CSV file
# Load all the data files
# Combine all the data files into a single DataFrame

# Load all the data files
data_files = os.listdir("data")
dfs = []
for file in data_files:
    if file.startswith("wu-"):
        df = pd.read_csv(f"data/{file}")
        dfs.append(df)

# Combine all the data files into a single DataFrame
combined_df = pd.concat(dfs)
combined_df.to_csv("data/weather_data.csv", index=False)
