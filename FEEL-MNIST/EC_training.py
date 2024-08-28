from Minist_fc import *
def Client(ID,training_data,test_data,sizes,_weights,_biases):
    print('EC node training with ID',ID)
    fc = FCN(sizes, _biases, _weights)
    # 设置迭代次数，Epoch=5,mini-batch大小为20，学习率为3，并且设置测试集，即每一轮训练完成之后，都对模型进行一次评估。
    # 这里的参数可以根据实际情况进行修改
    _weights1, _biases1, training_cost = fc.SGD(training_data, 5, 20, 4, test_data=test_data)
    return _weights1, _biases1, training_cost