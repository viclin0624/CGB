import subprocess
from enum import Enum

import mlflow
import typer

from benchmarks.designed_model import CGB
from benchmarks.trained_model import CGB_undesigned_model
class Experiment(str, Enum):
    CGB = "CGB"
    CGB_undesigned = "CGB_undesigned"


def main(experiment: Experiment = typer.Argument(..., help="Dataset to use"),
         sample_count: int = typer.Option(10, help='How many times to retry the whole experiment'),
         ):
    #log experiment and GPU in mlflow
    mlflow.set_experiment(experiment.value)
    try:
        out = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        gpu_model = stdout.decode().strip()
        mlflow.log_param('GPU', gpu_model)
    except FileNotFoundError:
        pass
    class_map = {
        Experiment.CGB: CGB,
        Experiment.CGB_undesigned: CGB_undesigned_model,
    }
    #run benchmarks
    benchmark_class = class_map[experiment]
    benchmark = benchmark_class(sample_count)
    benchmark.run()


if __name__ == "__main__":
    typer.run(main)
