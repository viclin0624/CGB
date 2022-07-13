# dgl-gnn-exp

## Quickly Start
If you want to run EXP2 in paper for testing interpretation methods, please run:
```
sh run_fixed.sh
```
If you want to run EXP3 in paper for training unfixed models, please run:
```
sh run_unfixed.sh
```
Note: default train 30 models.

## Files Description
* benchmarks: main files are benchmark_dgl.py, fixed_dgl.py and unfixed_dgl.py, other files are some experiment files as file names.

* method: main file is explain_methods_dgl.py, other files are implementation of GNNExplainer and PGMExplainer.

* mlruns: some auto generate files in experiments.

* model: main file is model_dgl.py defining fixed model and unfixed model.
