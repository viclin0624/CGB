import subprocess
from enum import Enum

import mlflow
import typer

from benchmarks.fixed_dgl import BA4label
class Experiment(str, Enum):
    ba4label = "ba4label"


def main(experiment: Experiment = typer.Argument(..., help="Dataset to use"),
         sample_count: int = typer.Option(10, help='How many times to retry the whole experiment'),
         num_layers: int = typer.Option(4, help='Number of layers in the GNN model'),
         concat_features: bool = typer.Option(True,
                                              help='Concat embeddings of each convolutional layer for final fc layers'),
         conv_type: str = typer.Option('GraphConv',
                                       help="Convolution class. Can be GCNConv or GraphConv"),
         ):
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
    }
    benchmark_class = class_map[experiment]
    benchmark = benchmark_class(sample_count, num_layers, concat_features, conv_type)
    benchmark.run()


if __name__ == "__main__":
    typer.run(main)