from os.path import abspath, expanduser
from torch import device, cuda

from ray import tune, init
from functools import partial
from training import *

config = read_config()
# use absolute path, b/c ray can't handle relative ones
config["Data"]["corpusfilename"] = abspath(expanduser(config["Data"]["corpusfilename"]))
config["Test"]["evalfilename"] = abspath(expanduser(config["Test"]["evalfilename"]))

def tune_report(scores, epoch_or_iteration):
    tune.report(**scores)

torch_device = device("cpu")
ray_devices = { "cpu": 1, "gpu": 1 } if torch_device.type == "cuda" else { "cpu": 1 }
print(f"running on device {torch_device}")

def setup_and_train_model(data_conf, test_conf, vector_cache, training_conf):
    (train, test, _), data = load_data(loadconfig(**data_conf), vector_cache)

    tc = trainparam(**training_conf, **test_conf)
    model = tagger(data.dims, tc).double()
    model.to(torch_device)
    train_model(model, tc, train, test, data, \
        torch_device=torch_device, report_loss=tune_report)

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
    partial(setup_and_train_model, config["Data"], config["Test"], abspath("./.vector_cache")),
    name="Supertag-gridsearch",
    config=tune_parameters,
    num_samples=grid_samples,
    resources_per_trial=ray_devices,
    progress_reporter=tune.CLIReporter(max_progress_rows=5, max_error_rows=5, max_report_frequency=1800),
    verbose=1,
    fail_fast=True
)
print(analysis.get_best_config(metric="lf/test/parse"))
