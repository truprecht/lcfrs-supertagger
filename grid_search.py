from os.path import abspath, expanduser
from torch import device, cuda

from ray import tune
from functools import partial
from training import load_data, train_model, read_config, tagger

config = read_config()
# use absolute path, b/c ray can't handle relative ones
config["Data"]["corpus"] = abspath(expanduser(config["Data"]["corpus"]))

def tune_report(scores, epoch_or_iteration):
    tune.report(**scores)

torch_device = device("cpu")
ray_devices = { "cpu": 1, "gpu": 1 } if torch_device.type == "cuda" else { "cpu": 1 }
print(f"running on device {torch_device}")

def setup_and_train_model(training_conf, data_conf=None):
    (train, test, _), data = load_data(data_conf["Data"], \
        tag_distance=int(training_conf["tag_distance"]))

    model = tagger(data.dims, tagger.Hyperparameters.from_dict(training_conf)).double()
    model.to(torch_device)
    train_model(model, training_conf, train, test, \
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
    partial(setup_and_train_model, data_conf=config),
    name="Supertag-gridsearch",
    config=tune_parameters,
    num_samples=grid_samples,
    resources_per_trial=ray_devices,
    progress_reporter=tune.CLIReporter(max_progress_rows=5, max_error_rows=5, max_report_frequency=1800),
    verbose=1
)
print(analysis.get_best_config(metric="loss/test/combined"))
