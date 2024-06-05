import json

file_a = "forecast_results/20240604094740/forecast_results.json"
file_b = "forecast_results/20240604113546/forecast_results.json"
file_c = "forecast_results/20240604145215/forecast_results.json"


# use the actual and predicted values to calculate the mean squared error
def calculate_mse(file):
    with open(file) as f:
        data = json.load(f)
    mse = 0
    for i in range(len(data["results"])):
        mse += (data["results"][i]["actual"] - data["results"][i]["predicted"]) ** 2
    mse /= len(data["results"])
    return mse


print("MSE without time since:", calculate_mse(file_a))
print("MSE with time since:", calculate_mse(file_b))
print("MSE without time since, 8 layers:", calculate_mse(file_c))
