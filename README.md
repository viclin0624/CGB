# dgl-gnn-exp

## Quickly Start
If you want to run EXP1 in paper for testing explanation methods, please run:
```
sh run_fixed.sh
```
If you want to run EXP2 in paper for training unfixed models, please run:
```
sh run_unfixed.sh
```
Note: default train 30 models, after that run files named EXP3_xxxx in ./benchmarks/ for next step.

## Files Description
* benchmarks: main files are benchmark_dgl.py, designed_model.py and trained_model.py, other files are some experiment files as file names.

* method: main file is explain_methods_dgl.py, other files are implementation of GNNExplainer and PGMExplainer.

* mlruns: some auto generate files in experiments.

* model: main file is model_dgl.py defining fixed model and unfixed model.

## Set hyperparameters
* benchmarks/benchmark_dgl.py: use which explanation methods and learning rate when train models

* benchmarks/designed_model.py: number of nodes and m in dataset, whether summarize the explanation results

* benchmarks/trained_model.py: besides above, some train and test hyperparameters

* explain_methods_dgl.py: hyperparameters of explanation methods

## Results
```
mlflow ui
```
You can use this command to see results in browser