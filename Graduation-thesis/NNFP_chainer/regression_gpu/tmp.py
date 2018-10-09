import time
import numpy 
import cupy
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
CPU_TIME = []
GPU_TIME = []
x = numpy.linspace(1,10,10)

for i in range(1,11):
	N = 100 * i
	N2 = N * N
	xp = numpy
	#配列の次元
	
	#配列のセット
	a = xp.arange( N2 ).reshape( N , N )
	start = time.time()
	#内積の計算
	a = a.dot( a )
	end = time.time()
	#内積の計算時間を出力する
	CPU_TIME.append(end-start)
	print ( end - start )
	
	
	xp = cupy
	
	a = xp.arange( N2 ).reshape( N , N )
	start = time.time()
	#内積の計算
	a = a.dot( a )
	end = time.time()
	#内積の計算時間を出力する
	GPU_TIME.append(end-start)
	print ( end - start )

ax.plot(x,CPU_TIME)
ax.plot(x,GPU_TIME)
fig.show()
fig.savefig('time_test.png')
