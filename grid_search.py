from os.path import abspath, expanduser
from torch import device, cuda

from ray import tune
from functools import partial
from training import train_model, read_config

config = read_config()
# use absolute path, b/c ray can't handle relative ones
config["Data"]["corpus"] = abspath(expanduser(config["Data"]["corpus"]))

def tune_report(scores, epoch_or_iteration):
    tune.report(**scores)

torch_device = device("cpu")
ray_devices = { "cpu": 1, "gpu": 1 } if torch_device.type == "cuda" else { "cpu": 1 }
print(f"running on device {torch_device}")

grid_axes = set(config["Gridsearch"]["grid_axes"].split())
choice_axes = set(config["Gridsearch"]["choice_axes"].split())
grid_samples = int(config["Gridsearch"]["samples"])
tune_parameters = {}
for key in config["Gridsearch"]:
    if key in grid_axes:
        tune_parameters[key] = tune.grid_search([float(v) for v in config["Gridsearch"][key].split()])
    elif key in choice_axes:
        tune_parameters[key] = tune.choice([float(v) for v in config["Gridsearch"][key].split()])
    else:
        tune_parameters[key] = config["Gridsearch"][key]

analysis = tune.run(
    partial(train_model, data_conf=config, torch_device=torch_device, report_loss=tune_report),
    name="Supertag-gridsearch",
    config=tune_parameters,
    num_samples=grid_samples,
    resources_per_trial=ray_devices,
    progress_reporter=tune.CLIReporter(max_progress_rows=5, max_error_rows=5, max_report_frequency=1800),
    verbose=1
)
print(analysis.get_best_config(metric="loss/test/combined"))
