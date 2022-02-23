
import sys
sys.path.append('..')
from model.models_dgl import FixedNet2
from build_graph import build_graph
import dgl
import torch
import numpy as np
from collections import Counter
import typer
from tqdm import tqdm


