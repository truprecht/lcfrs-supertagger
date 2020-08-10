from sys import argv
from configparser import ConfigParser
from torch import device, cuda

from ray import tune
from functools import partial
from training import train_model


if len(argv) < 2:
    msg = f"""use {argv[0]} <config> [corpus]
              or {argv[0]} default corpus"""
    print(msg)
    exit(1)

config = ConfigParser()
config.read("training.conf" if argv[1] == "default" else argv[1])
print(config)

def tune_report(scores, epoch_or_iteration):
    tune.report(**scores)

torch_device = device("cpu")
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
    resources_per_trial={torch_device.type.replace("cuda", "gpu"): 1},
    progress_reporter=tune.CLIReporter(max_progress_rows=5, max_error_rows=5, max_report_frequency=1800),
    verbose=1
)
print(analysis.get_best_config(metric="loss/test/combined"))
