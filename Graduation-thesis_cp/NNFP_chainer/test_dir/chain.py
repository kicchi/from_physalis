#coding: utf-8
import numpy as np
import numpy.random as npr
#import cupy as cp #GPUを使うためのnumpy
import chainer 
from chainer import cuda, Function, gradient_check, \
	Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets
from chainer.training import extensions
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
X = X.astype(np.float32)
Y = iris.target
Y = Y.flatten().astype(np.int32)

train ,test= datasets.split_dataset_random(chainer.datasets.TupleDataset(X,Y),100)
train_iter = chainer.iterators.SerialIterator(train, 10)
test_iter = chainer.iterators.SerialIterator(test, 1,repeat=False, shuffle=False)


class Model01(chainer.Chain):
	def __init__(self):
		super(Model01,self).__init__(
			l1 = L.Linear(4,100),
			l2 = L.Linear(100,100),
			l3 = L.Linear(100,4))

	def __call__(self,x):    
		h = F.sigmoid(self.l1(x))
		h = F.relu(self.l2(h))
		return self.l3(h)

class Model02(chainer.Chain):
	def __init__(self):
		super(Model02,self).__init__(
			l1 = L.Linear(4,100),
			l2 = L.Linear(100,100),
			l3 = L.Linear(100,4))

	def __call__(self,x):    
		h = F.sigmoid(self.l1(x))
		h = (self.l2(h))
		return self.l3(h)

class IrisModel_1(chainer.Chain):
	def __init__(self):
		super(IrisModel_1,self).__init__(
			l11 = L.Linear(4,100),
			l12 = L.Linear(100,100),
			l13 = L.Linear(100,4),

			l21 = L.Linear(4,100),
			l22 = L.Linear(100,100),
			l23 = L.Linear(100,4),
			
		)

	def __call__(self,x):    
		h = F.sigmoid(self.l11(x))
		h = F.relu(self.l12(h))
		h = self.l13(h)
		h = F.sigmoid(self.l21(x))
		h = self.l22(h)
		return self.l23(h)

class IrisModel_2(chainer.Chain):
	def __init__(self):
		super(IrisModel_2,self).__init__(
			m1 = Model01(),
			m2 = Model02(),
		)

	def __call__(self,x):    
		return self.m2(self.m1(x))

class IrisModel_3(chainer.Chain):
	def __init__(self):
		super(IrisModel_3,self).__init__(
		)

	def __call__(self,x):    
		return x

model = L.Classifier(IrisModel_2())
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
	
print model.__dict__
#import pdb;pdb.set_trace()

updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (30, 'epoch'), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())


trainer.run()




