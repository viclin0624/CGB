# CGB: Controllable Graph Benchmark for Evaluation of Graph Neural Network Explanation Methods

## Quickly Start
We use python 3.8.0, pytorch 1.7.1 and dgl 0.7.2. You can use this command to install packages. 

(Notice: This command install cpu version of dgl. If you want to run experiments on gpu, it needs to manually install from https://docs.dgl.ai/en/0.7.x/install/index.html and reset device in codes. You can see details in the last section.)
```
pip install -r requirements.txt 
```
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

* model: main file is model_dgl.py defining designed model and trained model.

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

## Run time
For experiment 1, we run every explanation methods 10 times ("ITERATION_NUM_OF_SUMMARY" in designed_model.py) for aggregating explanation results and run total experiment 10 times to get average of accuracy with default hyperparameters. 

* One experiments for PGMExplainer: 90 minutes

* One experiments for GNNExplainer: 6 minutes

* One experiments for SA: 30 seconds

* One experiments for Random: 22 seconds

* One experiments for IG: 19 minutes (about 7 minutes on GPU)

You can select to run which explanation methods in /benchmarks/benchmark_dgl.py. Total experiment 1 will cost about 19h30m.

For experiment 2, we first train model, run IG 1 times ("ITERATION_NUM_OF_SUMMARY" in trained_model.py) without aggregating explanation results and run total experiment 10 times to get average of accuracy with default hyperparameters. 

* Train one model: 3 minutes

* Explain one model on 100 samples: 2 minutes

Total 30 model will cost about 2h30m.
## Run experiments on GPU
To adapt to different machines, we modify codes to CPU version. If you want to run these codes on GPU, dgl with appropriate cuda version is needed and you should reset the device in benchmarks/benchmark_dgl.py line 46 and method/explain_methods_dgl.py line 18. If you want to run the benchmarks/vis_2models.py, the line 99 also need to be modified.
