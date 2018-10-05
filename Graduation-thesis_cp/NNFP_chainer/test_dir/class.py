import chainer
from chainer import Chain
import chainer.links as L

def build_class():
	class Test(Chain):
		def __inti__(self):
			super(Test, self).__init__(
				h1 = L.Linear(1,1),
			)
			#with self.init_scope():
				#setattr(self,'hoge', 123)
				#setattr(self,'hoge foo', 456)
				
		def print_func(self):
			print "hello"

	return Test()

test = build_class()
test.print_func()
print type(test)
print test.__dict__.keys()
print test._params
