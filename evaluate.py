import os
import json
import pandas as pd
from utils import calculate_mse, calculate_mae, calculate_rmse, calculate_r2


def get_metadata(file):
    with open(file) as f:
        data = json.load(f)
    return {
        data["input_size"],
        data["output_size"],
        data["hidden_size"],
        data["num_layers"],
        data["batch_size"],
        data["num_epochs"],
        data["activation"],
    }


# for each directory in forecast_results, calculate the mean squared error
dirs = os.listdir("forecast_results")


results = []
for dir in dirs:
    file = f"forecast_results/{dir}/forecast_results.json"
    file_metadata = f"forecast_results/{dir}/metadata.json"
    if not os.path.exists(file):
        continue
    if not os.path.exists(file_metadata):
        continue
    with open(file) as f:
        data = json.load(f)
    with open(file_metadata) as f:
        metadata = json.load(f)

    mse = calculate_mse(file)
    mae = calculate_mae(file)
    rmse = calculate_rmse(file)
    r2 = calculate_r2(file)
    if "activation" not in metadata:
        metadata["activation"] = "sigmoid"  # default value
    if "criterion" not in metadata:
        metadata["criterion"] = "MSE"
    results.append(
        {
            "input_size": metadata["input_size"],
            "output_size": metadata["output_size"],
            "hidden_size": metadata["hidden_size"],
            "num_layers": metadata["num_layers"],
            "batch_size": metadata["batch_size"],
            "num_epochs": metadata["num_epochs"],
            "activation": metadata["activation"],
            "criterion": metadata["criterion"],
            "mse": mse[0],
            "mae": mae[0],
            "rmse": rmse[0],
            "r2": r2[0],
            "file": mse[1],
            "dir": dir,
        }
    )

# find the min mse and mae from results


min_mae = 100000000000
min_mse = 100000000000
min_mae_index = -1
min_mse_index = -1
# sort by mse
results = sorted(results, key=lambda x: x["mse"])
# add index to results
for i, result in enumerate(results):
    result["index"] = i

for result in results:
    if result["mae"] < min_mae:
        min_mae = result["mae"]
        min_mae_index = result["index"]
    if result["mse"] < min_mse:
        min_mse = result["mse"]
        min_mse_index = result["index"]


print(f"min mae: {min_mae}, index: {min_mae_index}")
print(f"min mse: {min_mse}, index: {min_mse_index}")


df = pd.DataFrame(results)

df.to_csv("inference_results.csv", index=False)


# plot the epoch loss as a sub plot
import matplotlib.pyplot as plt

print(len(dirs))

# Assuming 'dirs' is defined somewhere in your script.
fig, ax = plt.subplots(
    len(dirs), 1, figsize=(10, len(dirs) * 3)
)  # Set the figure size dynamically based on the number of directories.
if (
    len(dirs) == 1
):  # If there is only one subplot, ax will not be an array, so we make it into a list for consistency.
    ax = [ax]

index = 0
for dir in dirs:
    file = f"forecast_results/{dir}/metadata.json"
    if not os.path.exists(file):
        continue
    with open(file) as f:
        metadata = json.load(f)
    # Plot the epoch loss in the subplot
    ax[index].plot(metadata["epoch_loss"])
    ax[index].set_xlabel("Epoch")  # Correct method to set xlabel
    ax[index].set_ylabel("Loss")  # Correct method to set ylabel
    ax[index].set_title(f"Epoch Loss for {dir}")  # Correct method to set title

    # determine where metadata["epoch_loss"] achieves less than 1% change over a 10 epoch rolling window
    rolling_window_size = 10
    threshold = 0.01
    rolling_avg_loss = []
    for i in range(rolling_window_size, len(metadata["epoch_loss"])):
        rolling_avg_loss.append(
            sum(metadata["epoch_loss"][i - rolling_window_size : i])
            / rolling_window_size
        )
    change_percent = []
    for i in range(1, len(rolling_avg_loss)):
        change_percent.append(
            (rolling_avg_loss[i - 1] - rolling_avg_loss[i]) / rolling_avg_loss[i - 1]
        )

    if "activation" not in metadata:
        metadata["activation"] = "sigmoid"  # default value
    if "criterion" not in metadata:
        metadata["criterion"] = "MSE"
    # display the metadata in each subplot
    ax[index].text(
        0.5,
        0.5,
        f"Input Size: {metadata['input_size']}\nOutput Size: {metadata['output_size']}\nHidden Size: {metadata['hidden_size']}\nNum Layers: {metadata['num_layers']}\nBatch Size: {metadata['batch_size']}\nNum Epochs: {metadata['num_epochs']}\nActivation: {metadata['activation']}\nCriterion: {metadata['criterion']}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[index].transAxes,
    )

    index += 1

plt.tight_layout()
plt.savefig("forecast_epoch_loss.png")
