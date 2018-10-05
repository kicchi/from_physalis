#coding: utf-8

import tensorflow as tf
import deepchem as dc
import numpy as np

graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
loader = dc.data.data_loader.CSVLoader( tasks = ['solubility'], smiles_field = "smiles", id_field = "name", featurizer = graph_featurizer )
dataset = loader.featurize( './solubility.csv' )

spliter = dc.splits.splitters.RandomSplitter()
trainset, testset = splitter.train_split( dataset )

hp = dc.molnet.preset_hyper_parameters
param = hp.hps['graphconvreg']
print (param)


#n_atoms = 5
n_feat = 75
batch_size = 32

graph_model = dc.nn.SequentialGraph(n_feat)

graph_model.add(dc.nn.GraphConv(int(param['n_filters' ]), n_feata, activation = 'relu'))
graph_model.add(dc.nn.BatchNormalization(eplison = 1e-5, mode = 1))
graph_model.add(dc.nn.GraphPool())

#Gather Projection
graph_model.add(dc.nn.BatchNormalization(eplison = 1e-5, mode = 1))
graph_model.add(dc.nn.GraphGater(batch_size,acivation = 'linear'))

with tf.Session() as sess:
	model_graphconv = dc.models.MultitaskGraphRegressor(graph_mode,
	1,
	n_feat,
	batch_size = batch_size,
	learning_rate = param['learning_rate'],
	optimizer_type = 'adam',
	beta1 = .9, beta2 = .999)
	model_graphconv.fit(trainset, nb_epoch = 10)

	


