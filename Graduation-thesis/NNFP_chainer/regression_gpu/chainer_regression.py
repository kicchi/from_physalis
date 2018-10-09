#coding: utf-8
import math
import time
#import numpy as np
import numpy.random as npr
import cupy as cp #GPUを使うためのnumpy
import chainer 
from chainer import cuda, Function, Variable, optimizers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

from NNFP import load_data 
from NNFP import result_plot 
from NNFP import normalize_array
from NNFP import Deep_neural_network
from NNFP import Finger_print


task_params = {'target_name' : 'measured log solubility in mols per litre',
				'data_file'  : 'delaney.csv'}

N_train = 70
N_val   = 1
N_test  = 10


model_params = dict(fp_length = 50,      
					fp_depth = 4,       #NNの層と、FPの半径は同じ
					conv_width = 20,    #必要なパラメータはこれだけ（？）
					h1_size = 100,      #最上位の中間層のサイズ
					L2_reg = cp.exp(-2))

train_params = dict(num_iters = 100,
					batch_size = 50,
					init_scale = cp.exp(-4),
					step_size = cp.exp(-6))

	
class Main(Chain):
	def __init__(self, model_params):
		super(Main, self).__init__(
			fp = Finger_print.FP(model_params),
			dnn = Deep_neural_network.DNN(model_params),
		)
	
	def __call__(self, x, y):
		t = time.time()
		y = Variable(cp.array(y, dtype=cp.float32))
		print("variable : ", time.time() - t)
		pred = self.prediction(x)
		return F.mean_squared_error(pred, y)

	def prediction(self, x):
		x = Variable(cuda.to_cpu(x))
		finger_print = self.fp(x)
		pred = self.dnn(finger_print)
		return pred

	def mse(self, x, y, undo_norm):
		y = Variable(cp.array(y, dtype=cp.float32))
		pred = undo_norm(self.prediction(x))
		return F.mean_squared_error(pred, y)
	
def train_nn(model, train_smiles, train_raw_targets, seed=0,
				validation_smiles=None, validation_raw_targets=None):

	num_print_examples = N_train
	train_targets, undo_norm = normalize_array(train_raw_targets)
	training_curve = []
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))	
	
	num_epoch = 100
	num_data = len(train_smiles)
	batch_size = 50
	x = train_smiles
	y = train_targets
	sff_idx = npr.permutation(num_data)
	TIME = time.time()
	for epoch in range(num_epoch):
		epoch_time = time.time()
		for idx in range(0,num_data, batch_size):
			batched_x = x[sff_idx[idx:idx+batch_size
				if idx + batch_size < num_data else num_data]]
			batched_y = y[sff_idx[idx:idx+batch_size
				if idx + batch_size < num_data else num_data]]
			update_time	 = time.time()
			model.zerograds()
			loss = model(batched_x, batched_y)
			loss.backward()
			optimizer.update()
			print("UPDATE TIME : ",  time.time() - update_time)
		#print "epoch ", epoch, "loss", loss._data[0]
		if epoch % 10 == 0:
			print_time = time.time()
			train_preds = model.mse(train_smiles, train_raw_targets, undo_norm)
			cur_loss = loss._data[0]
			training_curve.append(cur_loss)
			print("PRINT TIME : ",  time.time() - print_time)
			print("Iteration", epoch, "loss", math.sqrt(cur_loss), \
				"train RMSE", math.sqrt((train_preds._data[0])))
			if validation_smiles is not None:
				validation_preds = model.mse(validation_smiles, validation_raw_targets, undo_norm)
				print("Validation RMSE", epoch, ":", math.sqrt((validation_preds._data[0])))
		print("1 EPOCH TIME : ", time.time() - epoch_time)
		#print loss

		
	return model, training_curve, undo_norm

def main():
	print("Loading data...")
	traindata, valdata, testdata = load_data(
		task_params['data_file'], (N_train, N_val, N_test),
		input_name = 'smiles', target_name = task_params['target_name'])
	x_trains, y_trains = traindata
	x_vals, y_vals = valdata
	x_tests, y_tests = testdata
	x_trains = cp.reshape(x_trains, (N_train, 1))
	y_trains = cp.reshape(y_trains, (N_train, 1)).astype(cp.float32)
	x_vals = cp.reshape(x_vals, (N_val, 1))
	y_vals = cp.reshape(y_vals, (N_val, 1)).astype(cp.float32)
	x_tests = cp.reshape(x_tests, (N_test, 1))
	y_tests = cp.reshape(y_tests, (N_test, 1)).astype(cp.float32)

	def run_conv_experiment():
		'''Initialize model'''
		NNFP = Main(model_params) 
		optimizer = optimizers.Adam()
		optimizer.setup(NNFP)

		gpu_device = 0
		cuda.get_device(gpu_device).use()
		NNFP.to_gpu(gpu_device)
		#xp = cuda.cupy
		'''Learn'''
		trained_NNFP, conv_training_curve, undo_norm = \
			train_nn(NNFP, 
					 x_trains, y_trains,  
					 validation_smiles=x_vals, 
					 validation_raw_targets=y_vals)
		return math.sqrt(trained_NNFP.mse(x_tests, y_tests, undo_norm)._data[0]), conv_training_curve

	print("Starting neural fingerprint experiment...")
	test_loss_neural, conv_training_curve = run_conv_experiment()
	print() 
	print("Neural test RMSE", test_loss_neural)
	#result_plot(conv_training_curve, train_params)

if __name__ == '__main__':
	main()
