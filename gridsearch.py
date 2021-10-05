from itertools import product
from multiprocessing import Process, Queue, set_start_method
from os import makedirs
from typing import Dict, Iterable, Tuple

import torch

from training import main, corpusparam, TrainingParameters

class Grid:
    @classmethod
    def parse_grid_option(cls, s):
        k, vs = s.split("=")
        vs = vs.split(",")
        return k, vs

    def __init__(self, gridkvs: Iterable[str]):
        gridkvs = (self.__class__.parse_grid_option(s) for s in gridkvs)
        self.keys, self.vals = zip(*gridkvs)

    def __iter__(self):
        gridpoints = product(*self.vals)
        return (dict(zip(self.keys, gp)) for gp in gridpoints)


def grid_training(basename, base_config, grid_point: Dict[str, str], gpu_queue: Queue, seed: int):
    import flair
    from copy import deepcopy

    device = gpu_queue.get()
    flair.device = device
    
    conf = deepcopy(base_config)
    for key, val in grid_point.items():
        try:
            section = next(s for s in conf if key in conf[s])
        except StopIteration:
            section = "Training"
        conf[section][key] = val

    gpstr = "-".join(f"{key}={val}" for key, val in grid_point.items())
    main(conf, f"{basename}/{gpstr}", seed, param_selection_mode=True)

    gpu_queue.put(device)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from configparser import ConfigParser
    from os.path import basename
    from datetime import datetime
    import flair

    set_start_method("spawn")

    args = ArgumentParser()
    args.add_argument("configs", nargs="+", help="configuration files for corpus and model, e.g. `example.conf example-model.conf`")
    args.add_argument("--seed", type=int, help="sets an integer as seed for initialization of the neural network, default is `0`", default=0)
    args.add_argument("--devices", nargs="+", type=torch.device, help="a list of torch devices")
    args.add_argument("--grid", nargs="+", help="a list of model configuration keys and values")
    args = args.parse_args()

    conf = ConfigParser()
    for config in args.configs:
        conf.read(config)
    grid = Grid(args.grid)

    conffilenames = (basename(f).replace('.conf', '') for f in args.configs)
    filename = ("gridsearch-"
                f"{'-'.join(conffilenames)}-"
                f"{args.seed}-"
                f"{datetime.now():%d-%m-%y-%H:%M}")
    makedirs(filename, exist_ok=True)

    gpu_queue = Queue()
    for device in args.devices or [flair.device]:
        gpu_queue.put(device)

    ps = []
    for grid_point in grid:
        trainargs = (filename, conf, grid_point, gpu_queue, args.seed)
        ps.append(
            Process(target=grid_training, args=trainargs))
        ps[-1].start()
    for p in ps:
        p.join()
