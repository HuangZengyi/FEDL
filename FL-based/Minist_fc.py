import random
import numpy as np
import mnist_loader

def polynomial_app(z):
    return z**2

def polynomial_app_prime(z):
    return 2*z

def sigmoid(z):
    """
    Sigmoid激活函数定义
    """
    return 1.0/(1.0 + np.exp(-z))
def sigmoid_prime(z):
    """
    Sigmoid函数的导数,关于Sigmoid函数的求导可以自行搜索。
    """
    return sigmoid(z)*(1-sigmoid(z))
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        # 均方差误差函数
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        # 均方差误差函数的导数，其中激活函数为sigmoid
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        交叉熵损失函数定义 C = -1/n ∑(y*lna + (1-y)ln(1-a))
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)

class FCN(object):
    """
    全连接网络的纯手工实现
    """

    def __init__(self, sizes, _biases, _weights, cost=QuadraticCost):
        self.sizes = sizes
        self.cost = cost
        self._num_layers = len(sizes)
        self._biases = _biases
        self._weights = _weights
        # print(self._biases[0].shape)
        # print(self._biases[1].shape)
        # print(self._weights[0].shape)
        #print(len(self._weights))
        # print(self._weights[1].shape)

    def feedforward(self, a):
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
        for b, w in zip(self._biases, self._weights):
            #a = polynomial_app(np.dot(w, a) + b)
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0, test_data=None):
        """
        使用小批量随机梯度下降来训练网络
        :param training_data: training data 是一个元素为(x, y)元祖形式的列表，代表了训练数据的输入和输出。
        :param epochs: 训练轮次
        :param mini_batch_size: 小批量训练样本数据集大小
        :param eta: 学习率
        :param test_data: 如果test_data被指定，那么在每一轮迭代完成之后，都对测试数据集进行评估，计算有多少样本被正确识别了。但是这会拖慢训练速度。
        :return:
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # 在每一次迭代之前，都将训练数据集进行随机打乱，然后每次随机选取若干个小批量训练数据集
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            # 每次训练迭代周期中要使用完全部的小批量训练数据集
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            cost = self.total_cost(training_data, lmbda)
            # 如果test_data被指定，那么在每一轮迭代完成之后，都对测试数据集进行评估，计算有多少样本被正确识别了
        return self._weights,self._biases, cost


    def update_mini_batch(self, mini_batch, eta):
        """
        通过小批量随机梯度下降以及反向传播来更新神经网络的权重和偏置向量
        :param mini_batch: 随机选择的小批量
        :param eta: 学习率
        """
        nabla_b = [np.zeros(b.shape) for b in self._biases]
        nabla_w = [np.zeros(w.shape) for w in self._weights]
        for x,y in mini_batch:
            # 反向传播算法，运用链式法则求得对b和w的偏导
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #print(len(delta_nabla_w[1]))
            #print(x)
            # 对小批量训练数据集中的每一个求得的偏导数进行累加
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 使用梯度下降得出的规则来更新权重和偏置向量
        self._weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self._weights, nabla_w)]
        self._biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self._biases, nabla_b)]

    def backprop(self, x, y):
        """
        反向传播算法，计算损失对w和b的梯度
        :param x: 训练数据x
        :param y: 训练数据x对应的标签
        :return: Return a tuple ``(nabla_b, nabla_w)`` representing the
                gradient for the cost function C_x.  ``nabla_b`` and
                ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
                to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self._biases]
        nabla_w = [np.zeros(w.shape) for w in self._weights]
        # 前向传播，计算网络的输出
        activation = x
        # 一层一层存储全部激活值的列表
        activations = [x]
        # 一层一层第存储全部的z向量，即带权输入
        zs = []
        #print(y)
        for b, w in zip(self._biases, self._weights):
            # 利用 z = wt*x+b 依次计算网络的输出
            z = np.dot(w, activation) + b
            zs.append(z)
            # 将每个神经元的输出z通过激活函数sigmoid
            #activation = polynomial_app(z)
            activation = sigmoid(z)
            # 将激活值放入列表中暂存
            activations.append(activation)
        # 反向传播过程

        # 首先计算输出层的误差delta L

        #delta = self.cost_derivative(activations[-1], y) * polynomial_app_prime(zs[-1])
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # 反向存储 损失函数C对b的偏导数
        nabla_b[-1] = delta
        # 反向存储 损失函数C对w的偏导数
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 从第二层开始，依次计算每一层的神经元的偏导数
        for l in range(2, self._num_layers):
            z = zs[-l]
            #sp = polynomial_app(z)
            sp = sigmoid(z)
            # 更新得到前一层的误差delta
            delta = np.dot(self._weights[-l + 1].transpose(), delta) * sp
            # 保存损失喊出C对b的偏导数，它就等于误差delta
            nabla_b[-l] = delta
            # 根据第4个方程，计算损失函数C对w的偏导数
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        # 返回每一层神经元的对b和w的偏导数
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        返回神经网络对测试数据test_data的预测结果，并且计算其中识别正确的个数
        因为神经网络的输出是一个10x1的向量，我们需要知道哪一个神经元被激活的程度最大，
        因此使用了argmax函数以获取激活值最大的神经元的下标，那就是网络输出的最终结果。
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self._weights)
        return cost

    def cost_derivative(self, output_activations, y):
        """
        返回损失函数对a的的偏导数，损失函数定义 C = 1/2*||y(x)-a||^2
        求导的结果为：
            C' = y(x) - a
        """
        #print(len(y),output_activations.shape)
        return (output_activations - y)



