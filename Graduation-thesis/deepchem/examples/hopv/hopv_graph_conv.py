"""
Script that trains graph-conv models on HOPV dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc
from hopv_datasets import load_hopv

# Load HOPV dataset
hopv_tasks, hopv_datasets, transformers = load_hopv(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = hopv_datasets

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean, mode="regression"),
    dc.metrics.Metric(
        dc.metrics.mean_absolute_error, np.mean, mode="regression")
]

# Number of features on conv-mols
n_feat = 75
# Batch size of models
batch_size = 50
graph_model = dc.nn.SequentialGraph(n_feat)
graph_model.add(dc.nn.GraphConv(64, n_feat, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphPool())
graph_model.add(dc.nn.GraphConv(64, 64, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphPool())
# Gather Projection
graph_model.add(dc.nn.Dense(128, 64, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))

model = dc.models.MultitaskGraphRegressor(
    graph_model,
    len(hopv_tasks),
    n_feat,
    batch_size=batch_size,
    learning_rate=1e-3,
    learning_rate_decay_time=1000,
    optimizer_type="adam",
    beta1=.9,
    beta2=.999)

# Fit trained model
model.fit(train_dataset, nb_epoch=25)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
