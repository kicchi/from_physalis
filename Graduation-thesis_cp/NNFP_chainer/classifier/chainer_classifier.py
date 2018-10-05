#coding: utf-8
import numpy as np
import numpy.random as npr
#import cupy as cp #GPUを使うためのnumpy
import chainer 
from chainer import cuda, Function, Variable, optimizers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

from NNFP import load_data, smiles_from_SDF
from NNFP import normalize_array
from NNFP import Deep_neural_network
from NNFP import Finger_print



task_params = {'target_name' : 'measured log solubility in mols per litre',
				'data_file'  : 'delaney.csv'}

N_train = 1
N_val   = 1
N_test  = 1

model_params = dict(fp_length = 50,      
					fp_depth = 4,       #NNの層と、FPの半径は同じ
					conv_width = 20,    #必要なパラメータはこれだけ（？）
					h1_size = 100,      #最上位の中間層のサイズ
					L2_reg = np.exp(-2))

train_params = dict(num_iters = 100,
					batch_size = 50,
					init_scale = np.exp(-4),
					step_size = np.exp(-6))


	
class Main(Chain):
	def __init__(self, model_params):
		super(Main, self).__init__(
			fp = Finger_print.FP(model_params),
			dnn = Deep_neural_network.DNN(model_params),
		)
	
	def __call__(self, x, y):
		y = Variable(np.array(y, dtype=np.int32))
		pred = self.prediction(x)
		'''mse → sigomoid cross entropy'''
		return F.sigmoid_cross_entropy(pred, y)



	def prediction(self, x):
		x = Variable(x)
		finger_print = self.fp(x)
		prob = self.dnn(finger_print)
		
		pred = Variable(np.empty((0, 1), dtype=np.float32))
		'''convert to 0,1'''
		for i in range(len(prob)):
			if prob[i]._data[0] > 0.5:
				pred = F.concat((pred, Variable(np.array([[1]], dtype=np.float32))), axis=0)
			else:
				pred = F.concat((pred, Variable(np.array([[0]], dtype=np.float32))), axis=0)
		return pred

	def mse(self, x, y, undo_norm, test_data=False):
		acc = 0.0	
		y = Variable(np.array(y, dtype=np.int32))
		pred = self.prediction(x)
		'''accuracy'''
		if test_data:
			for i in range(len(pred)):
				if int(y[i].data[0]) == int(pred[i]._data[0][0]):
					acc = acc + 1.0

		'''mse → sigomoid cross entropy'''
		return F.sigmoid_cross_entropy(pred, y), acc / float(len(pred))
	
def train_nn(model, train_smiles, train_targets, seed=0,
				validation_smiles=None, validation_raw_targets=None):

	num_print_examples = N_train
	_, undo_norm = normalize_array(train_targets)
	training_curve = []
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(0.01))	
	
	num_epoch = 200
	num_data = len(train_smiles)
	batch_size = 30
	x = train_smiles
	y = train_targets
	sff_idx = npr.permutation(num_data)
	for epoch in range(num_epoch):
		for idx in range(0,num_data, batch_size):
			batched_x = x[sff_idx[idx:idx+batch_size
				if idx + batch_size < num_data else num_data]]
			batched_y = y[sff_idx[idx:idx+batch_size
				if idx + batch_size < num_data else num_data]]
			model.zerograds()
			loss = model(batched_x, batched_y)
			loss.backward()
			optimizer.update()
		if epoch % 10 == 0:
			train_preds, _ = model.mse(train_smiles, train_targets, undo_norm)
			cur_loss = loss
			training_curve.append(cur_loss)
			print "Iteration", epoch, "loss", cur_loss._data[0], \
				"train RMSE", (train_preds._data[0]),
			if validation_smiles is not None:
				validation_preds, _ = model.mse(validation_smiles, validation_raw_targets, undo_norm)
				print  "Validation RMSE", epoch, ":", (validation_preds._data[0])

	
	return model, training_curve, undo_norm

def main():
	print "Loading data..."
	
	'''2 class data''' 
	sdf_train, sdf_val, sdf_test = smiles_from_SDF('mutag.sdf', (N_train, N_val, N_test))	
	x_trains, y_trains = sdf_train
	x_vals, y_vals = sdf_val
	x_tests, y_tests = sdf_test
	def run_conv_experiment():
		'''Initialize model'''
		NNFP = Main(model_params) 
		optimizer = optimizers.Adam()
		optimizer.setup(NNFP)
		'''Learn'''
		trained_NNFP, conv_training_curve, undo_norm = \
			train_nn(NNFP, 
					 x_trains, y_trains,  
					 validation_smiles=x_vals, 
					 validation_raw_targets=y_vals)
		return trained_NNFP.mse(x_tests, y_tests, undo_norm, True)

	print "Starting neural fingerprint experiment..."
	test_loss_neural, accuracy = run_conv_experiment()
	print 
	print  "Neural test RMSE", test_loss_neural._data[0]
	print "accuracy", accuracy

if __name__ == '__main__':
	main()
