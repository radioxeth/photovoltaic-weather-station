# At Home Solar Forecasting

A repository for investigating the relationship between local, Personal Weather Station (PWS) and Residential Solar Panel (RSP) power production. We then train a Long-Short Term model using the available PWS and RSP data to forecast next-day power production given a National Weather Service (NWS) forecast

Caution! This is not production ready - it is a class project for CIS-600 Fundamentals of Data and Knowledge mining. The data scraping and normalization process is not automated.

Link to paper: [At Home Solar Forecasting](/daniel_shannon_final_paper.pdf)

**Contents**
- [Data](#data)
- [Overview of Project](#overview-of-project)
- [Data Collection](#data-collection)
- [Model Training and Testing](#model-training-and-testing)
- [Investigation and Evaluation](#investigation-and-evaluation)


## Data
([top](#at-home-solar-forecasting))

[data.zip](data.zip) contains the following files:
- `weather_solar_data.csv`
  - scraped PWS weather and RSP solar data combined
  - used in training and inference
- `forecasted_weather_solar_data.csv`
  - scraped NWS weather forecasts and solar data combined
  - used in forecast inference
- `inference_results.csv`
  - inference results and summaries

## Overview of Project
([top](#at-home-solar-forecasting))

![Satellite View](/images/satellite_view.png)

Single family home in Denver, CO with 24 solar panels and a 5-in-1 PWS
<hr>

![Schematic View](/images/pv_schematic.png)

Schematic view of the panels.
<hr>

![Data Collection](/images/data_collections.png)

The resampling and feature extraction used for RSP, PWS, and NWS data.
<hr>

![Project Design](/images/project_design.png)

General overview of the project design.
<hr>

![Binning](/images/binning.png)

How we bin the training data using a sliding window.
<hr>

![LSTM Model](/images/lstm.png)

An overview of the LSTM unit to address the vanishing gradient problem.


## Data Collection
([top](#at-home-solar-forecasting))

1. Scrape the solar data (SolarEdge API key needed)
    - [/solar_energy/scrape_energy.sh](/solar_energy/scrape_energy.sh)
    - [/solar_power/scrape_power.sh](/solar_power/scrape_power.sh)

2. Scrape the PWS data (wunderground API key needed)
    - [/wunderground/scrape_weather_history.py](/wunderground/scrape_weather_history.py)

3. Combine the weather and solar data
    - [/wunderground/combine_weather_solar_data.py](/wunderground/combine_weather_solar_data.py)
    - This combines the weather and solar data into `wunderground/data/weather_solar_data.csv`

4. Scrape the current NWS (NOAA) forecast
    - [/noaa_forecasts/get_forecast.sh](/noaa_forecasts/get_forecast.sh)

5. Combine the historical weather forecast and solar data
    - [/noaa_forecasts/combine_forecast_and_solar.py](/noaa_forecasts/combine_forecast_and_solar.py)

## Model Training and Testing
([top](#at-home-solar-forecasting))

The LSTM model itself is defined in [lstm.py](/lstm.py)

1. [lstm_assess.py](/lstm_assess.py)
   - combines steps 2-4 to build the data, train the model, and infer the model (including forecasts)
2. [lstm_data.py](/lstm_data.py)
   - Here we bin, normalize, and apply different features to the data in `wunderground/data/weather_solar_data.csv`. 
   - The data is saved to `lstm_data` as .npy files with a unique _description_.
   - Builds an input (weather) and output (solar) dataset for training, inference, and forecasts
     - **input_seq_train_save_path** = f"{data_dir}/input_seq_train{description}.npy"
     - **target_seq_train_save_path** = f"{data_dir}/target_seq_train{description}.npy"
     - **input_seq_infer_save_path** = f"{data_dir}/input_seq_infer{description}.npy"
     - **target_seq_infer_save_path** = f"{data_dir}/target_seq_infer{description}.npy"
     - **input_seq_forecast_save_path** = f"{data_dir}/input_seq_forecast{description}.npy"
     - **target_seq_forecast_save_path** = f"{data_dir}/target_seq_forecast{description}.npy"
3. [lstm_train.py](/lstm_train.py)
   - trains the model
   - use the same parameters as `lstm_data.py`
   - saves the following in `forecast_results/{timestamp}`
     - `lstm_model.pth`
     - `metadata.json`

4. [lstm_infer.py](/lstm_infer.py)
   - infers the model
   - use the same parameters as `lsmt_infer.py`
   - saves the following in `forecast_results/{timestamp}`
     - `forecast_results.json`
     - forecasted vs measured solar plots
5. [lstm_forecast.py](/lstm_forecast.py)
   - infers the model using NWS forecasts
   - saves the following in `noaa_forecast_results/{timestamp}`
     - `noaa_forecast_results.json`
     - forecasted vs measured solar plots

## Investigation and Evaluation
([top](#at-home-solar-forecasting))

- utility functions
  - [utils.py](/utils.py)
  - [utils_plt.py](/utils_plt.py)
- [pca_variance.py](/pca_variance.py)
  - PCA variance of raw data
- [evaluate.py](/evaluate.py)
  - Evaluates the performance of the models using the saved metadata and results files.
  - plots training epochs and saves results to `evaluate.csv`

### Correlation Plot

![Correlation Matrix](/corr_matrix.png)

### PCA Variance

![PCA Variance](/pca_plot/cumulative_variance.png)

### Epoch Training

![Training Loss](/forecast_epoch_loss.png)