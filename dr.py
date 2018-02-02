import numpy as np
import pandas as pd
import sys
caffe_root = '/home/xionghao/caffe/'  #caffe的根目录
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_cpu()

#设置网络配置文件和模型文件
model_def = '/home/xionghao/caffe/examples/kaggle/lenet.prototxt'
model_weights = '/home/xionghao/caffe/examples/kaggle/lenet_iter_10000.caffemodel'

#初始化网络
net = caffe.Net(model_def,model_weights,caffe.TEST)     

#读取测试数据
test=pd.read_csv('/home/xionghao/caffe/examples/kaggle/test.csv').values
n=np.shape(test)[0]

testId = range(1, n+1)
testLabel=[]

for i in xrange(n):
	#归一化
    input=test[i,:]/255.0
    #将1*784的向量调整为28*28的矩阵
	input=input.reshape((1,1,28,28))
	#输入数据
    net.blobs['data'].data[...] = input
    #前向传播一次，从返回的prob中选取最大可能性的分类
	output = net.forward()['prob'][0].argmax()
    testLabel.append(output)
	print output

dataframe = pd.DataFrame({'ImageId': testId, 'Label': testLabel})
dataframe.to_csv("submissions.csv", index=False)

