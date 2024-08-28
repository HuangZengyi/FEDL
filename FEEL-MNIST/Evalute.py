import numpy as np
def sigmoid(z):
    """
    Sigmoid激活函数定义
    """
    return 1.0/(1.0 + np.exp(-z))

def feedforward(a,_biases,_weights):
    """
    前向计算，返回神经网络的输出。公式如下:
    output = sigmoid(w*x+b)
    以[784,30,10]为例，权重向量大小分别为[30x784, 10x30]，偏置向量大小分别为[30x1, 10x1]
    输入向量为 784x1.
    矩阵的计算过程为：
        30x784 * 784x1 = 30x1
        30x1 + 30x1 = 30x1

        10x30 * 30x1 = 10x1
        10x1 + 10x1 = 10x1
        故最后的输出是10x1的向量，即代表了10个数字。
    :param a: 神经网络的输入
    """
    for b, w in zip(_biases, _weights):
        # a = polynomial_app(np.dot(w, a) + b)
        a = sigmoid(np.dot(w, a) + b)

    return a

def evaluate(test_data,_biases,_weights):
    """
    返回神经网络对测试数据test_data的预测结果，并且计算其中识别正确的个数
    因为神经网络的输出是一个10x1的向量，我们需要知道哪一个神经元被激活的程度最大，
    因此使用了argmax函数以获取激活值最大的神经元的下标，那就是网络输出的最终结果。
    """
    test_results = [(np.argmax(feedforward(x,_biases,_weights)), y)
                    for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)
