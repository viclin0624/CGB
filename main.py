import subprocess
from enum import Enum

import mlflow
import typer

from benchmarks.designed_model import BA4label
from benchmarks.trained_model import BA4label_unfixed_model
class Experiment(str, Enum):
    ba4label = "ba4label"
    ba4label_unfixed = "ba4label_unfixed"


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
        Experiment.ba4label: BA4label,
        Experiment.ba4label_unfixed: BA4label_unfixed_model,
    }
    #run benchmarks
    benchmark_class = class_map[experiment]
    benchmark = benchmark_class(sample_count)
    benchmark.run()


if __name__ == "__main__":
    typer.run(main)
