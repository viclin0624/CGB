# CGB: Controllable Graph Benchmark for Evaluation of Graph Neural Network Explanation Methods

## Quickly Start
If you want to run Experiment 1 in paper for testing explanation methods, please run:
```
sh run_designed.sh
```
If you want to run Experiment 2 in paper for training models, please run:
```
sh run_trained.sh
```
Note: default train 30 models, after that run files named EXP2_xxxx in ./benchmarks/ as next step.

## Files Description
* benchmarks: main files are benchmark_dgl.py, designed_model.py and trained_model.py, other files are some experiment files as file names. result_models is saved trained models that are generated in Experiment 2 and model_cut shows edge mask distribution of these model in Experiment 2. 

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